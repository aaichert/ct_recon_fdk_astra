import os
import sys
import json
import time
import gc
import tempfile
import numpy as np
import nrrd
from PIL import Image
import tifffile
import astra
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from tqdm import tqdm

def parse_ompl_matrices(ompl_path):
    """Parse 3x4 projection matrices from OMPL file."""
    matrices = []
    with open(ompl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.strip("[]")
            rows = line.split(";")
            mat = []
            for r in rows:
                mat.append([float(x) for x in r.split()])
            matrices.append(np.array(mat))
    return matrices

def decompose_to_astra_vector(P_opt, Nx, Ny, Nz, W, H):
    """Decompose projection matrix to ASTRA cone_vec format with coordinate swap."""
    cx_vox = (Nx - 1) / 2.0
    cy_vox = (Ny - 1) / 2.0
    cz_vox = (Nz - 1) / 2.0
    
    # Translate from centered coordinates to voxel indices
    T_center = np.array([
        [1.0, 0.0, 0.0, cx_vox],
        [0.0, 1.0, 0.0, cz_vox],
        [0.0, 0.0, 1.0, cy_vox],
        [0.0, 0.0, 0.0, 1.0]
    ])
    P_centered = P_opt @ T_center
    
    # Decompose using ProjectiveGeometry23
    P_obj = ProjectionMatrix(P_centered, image_size=(W, H), pixel_spacing=1.0)
    K, R, t, appears_flipped = P_obj.decomposition(imageVPointsUp=False)
    
    C_hom = P_obj.getCenterOfProjection().flatten()
    C = C_hom[:3] / (C_hom[3] if abs(C_hom[3]) > 1e-12 else 1.0)
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    r1 = R[0, :]
    r2 = R[1, :]
    r3 = R[2, :]
    
    S = C
    d_dist = 1.0
    D_pp = S + d_dist * r3
    u = (d_dist / fx) * r1
    v = (d_dist / fy) * r2
    
    u_mid = (W - 1) / 2.0
    v_mid = (H - 1) / 2.0
    D = D_pp + (u_mid - cx) * u + (v_mid - cy) * v
    
    # Swap Z and Y to map [x, z, y] coordinates to [x, y, z] expected by ASTRA
    def swap_zy(vec):
        return np.array([vec[0], vec[2], vec[1]])
    
    S_astra = swap_zy(S)
    D_astra = swap_zy(D)
    u_astra = swap_zy(u)
    v_astra = swap_zy(v)
    
    return np.concatenate([S_astra, D_astra, u_astra, v_astra])

def load_image(path):
    """Load a single 2D projection image, handles NRRD, TIFF (via tifffile) or PIL-supported formats."""
    path = str(path)
    if path.lower().endswith('.nrrd'):
        img, header = nrrd.read(path)
        img = np.squeeze(img)
        if img.ndim == 2:
            return img.T # Swap axes to match (H, W) ordering from (W, H)
        return img
    elif path.lower().endswith(('.tif', '.tiff')):
        return tifffile.imread(path).astype(np.float32)
    else:
        return np.array(Image.open(path), dtype=np.float32)

def preprocess_image(img, convert_to_line_integral, I0_val=None):
    """Preprocess projection image(s) (normalization and log attenuation conversion)."""
    img = img.astype(np.float32)
    if convert_to_line_integral:
        if I0_val is None:
            I0_val = float(img.max())
            if I0_val <= 0:
                I0_val = 1.0
        img = -np.log(np.clip(img / I0_val, 1e-6, 1.0))
    return img

def main(config_path=None):        
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: reconstruct <config.json>")
            sys.exit(1)
        config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Parse downsample config
    downsample_cfg = config.get("downsample")
    if downsample_cfg:
        skip = int(downsample_cfg[0])
        downscale_factor = float(downsample_cfg[1])
    else:
        skip = 1
        downscale_factor = 1.0
        
    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_dir = config.get("data_dir", "")
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(config_dir, data_dir))
    
    def resolve(path):
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(data_dir, path))

    def resolve_relative_to_config(path):
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))
        
    # 1. Parse Geometry Info
    if "ompl_file" in config:
        P_list = parse_ompl_matrices(resolve_relative_to_config(config["ompl_file"]))
    else:
        print("Error: Config must contain 'ompl_file'.")
        sys.exit(1)
        
    P_list = P_list[::skip]
        
    # Apply optional image and world transforms: H @ P @ T
    H_mat = np.array(config.get("image_transform", np.eye(3).tolist()))
    T_mat = np.array(config.get("world_transform", np.eye(4).tolist()))
    if not np.allclose(H_mat, np.eye(3)) or not np.allclose(T_mat, np.eye(4)):
        print("Applying optional image_transform and/or world_transform to projection matrices...")
        P_list = [H_mat @ P @ T_mat for P in P_list]
        
    # Apply downscaling homography H to P_list
    if abs(downscale_factor - 1.0) >= 1e-5:
        H_downscale = np.array([
            [downscale_factor, 0.0, 0.0],
            [0.0, downscale_factor, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P_list = [H_downscale @ P for P in P_list]
        
    # Apply model_matrix transform if provided
    if "model_matrix" in config:
        M = np.array(config["model_matrix"])
        # Swap Y and Z coordinates to map centered voxel coords correctly
        P_perm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        M_swapped = M @ P_perm
        P_list = [P @ M_swapped for P in P_list]
        
    # 2. Get Volume dimensions
    if "voxel_dimensions" in config:
        Nx, Ny, Nz = config["voxel_dimensions"]
    else:
        print("Error: Config must contain 'voxel_dimensions' ([Nx, Ny, Nz]).")
        sys.exit(1)
        
    # 3. Load / Process Images
    if "image_files" not in config or not config["image_files"]:
        print("Error: Config must contain 'image_files'.")
        sys.exit(1)
        
    image_paths = [resolve(f) for f in config["image_files"]]
    
    convert_to_att = config.get("convert_to_line_integral", False)
    I0_val = config.get("I0", None)
    
    # Check if we have a single file containing a stack of projections
    if len(image_paths) == 1 and image_paths[0].lower().endswith('.nrrd'):
        print(f"Loading single NRRD projection stack: {image_paths[0]}")
        data, header = nrrd.read(image_paths[0])
        print(f"Loaded raw NRRD projection stack of shape {data.shape}, data type: {data.dtype} ({(data.nbytes / (1024**3)):.3f} GB)")
        # Transpose from (W, H, num_views) to (num_views, H, W)
        projs = np.transpose(data, (2, 1, 0))
        projs = projs[::skip]
        
        # Apply downscale factor if specified
        if abs(downscale_factor - 1.0) >= 1e-5:
            import scipy.ndimage
            print(f"Downscaling projection stack by factor {downscale_factor}...")
            first_scaled = scipy.ndimage.zoom(projs[0], downscale_factor, order=1)
            new_H, new_W = first_scaled.shape
            projs_scaled = np.zeros((len(projs), new_H, new_W), dtype=projs.dtype)
            projs_scaled[0] = first_scaled
            for idx in range(1, len(projs)):
                projs_scaled[idx] = scipy.ndimage.zoom(projs[idx], downscale_factor, order=1)
            projs = projs_scaled
            
        num_views, H, W = projs.shape
        
        # Apply image_transform to projection images if specified
        if not np.allclose(H_mat, np.eye(3)):
            print("Applying image_transform to projection stack...")
            if np.allclose(H_mat, np.array([[-1.0, 0.0, W - 1.0], [0.0, -1.0, H - 1.0], [0.0, 0.0, 1.0]])):
                projs = projs[:, ::-1, ::-1]
            elif np.allclose(H_mat, np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])):
                projs = projs[:, :, ::-1]
            elif np.allclose(H_mat, np.array([[1.0, 0.0, 0.0], [0.0, -1.0, H - 1.0], [0.0, 0.0, 1.0]])):
                projs = projs[:, ::-1, :]
            else:
                from skimage.transform import warp, ProjectiveTransform
                tform = ProjectiveTransform(matrix=np.linalg.inv(H_mat))
                for i in range(num_views):
                    projs[i] = warp(projs[i], tform, order=1, mode="constant", cval=0.0)
                    
        if convert_to_att and I0_val is None:
            I0_val = float(projs[0].max())
            if I0_val <= 0:
                I0_val = 1.0
                
        print("Preprocessing projections in memory...")
        projs_att = preprocess_image(projs, convert_to_att, I0_val)
        # Convert to ASTRA sinogram shape (H, num_views, W)
        p_meas_sino = np.transpose(projs_att, (1, 0, 2))
        temp_filename = None
    else:
        # Multiple images (load on the fly to avoid out-of-memory errors)
        image_paths = image_paths[::skip]
        print(f"Processing {len(image_paths)} individual projection images...")
        projs = None
        num_views = len(image_paths)
        
        # Load first image to get dimensions
        first_img = load_image(image_paths[0])
        print(f"Loaded first individual projection image of shape {first_img.shape}, data type: {first_img.dtype}")
        if abs(downscale_factor - 1.0) >= 1e-5:
            import scipy.ndimage
            first_img = scipy.ndimage.zoom(first_img, downscale_factor, order=1)
            print(f"Downscaled first individual projection image to: {first_img.shape}")
        H, W = first_img.shape
        
        # Determine I0_val from the first projection if not specified
        if convert_to_att and I0_val is None:
            I0_val = float(first_img.max())
            if I0_val <= 0:
                I0_val = 1.0
            print(f"Estimated I0_val: {I0_val}")
            
        use_memmap = config.get("use_memmap", False)
        if use_memmap:
            # Create a temporary memmap file to store the preprocessed sinogram on disk
            temp_file = tempfile.NamedTemporaryFile(suffix=".bin", dir=data_dir, delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            print(f"Allocating memory-mapped sinogram on disk: {temp_filename}")
            p_meas_sino = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=(H, num_views, W))
        else:
            print("Allocating sinogram in RAM...")
            p_meas_sino = np.zeros((H, num_views, W), dtype=np.float32)
            temp_filename = None
        
        for i, path in enumerate(tqdm(image_paths, desc="Loading projections")):
            img = load_image(path)
            if abs(downscale_factor - 1.0) >= 1e-5:
                import scipy.ndimage
                img = scipy.ndimage.zoom(img, downscale_factor, order=1)
            if not np.allclose(H_mat, np.eye(3)):
                if np.allclose(H_mat, np.array([[-1.0, 0.0, W - 1.0], [0.0, -1.0, H - 1.0], [0.0, 0.0, 1.0]])):
                    img = img[::-1, ::-1]
                elif np.allclose(H_mat, np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])):
                    img = img[:, ::-1]
                elif np.allclose(H_mat, np.array([[1.0, 0.0, 0.0], [0.0, -1.0, H - 1.0], [0.0, 0.0, 1.0]])):
                    img = img[::-1, :]
                else:
                    from skimage.transform import warp, ProjectiveTransform
                    tform = ProjectiveTransform(matrix=np.linalg.inv(H_mat))
                    img = warp(img, tform, order=1, mode="constant", cval=0.0)
            img_att = preprocess_image(img, convert_to_att, I0_val)
            p_meas_sino[:, i, :] = img_att
        p_meas_sino.flush()
        
    if len(P_list) != num_views:
        print(f"Error: Number of projection matrices ({len(P_list)}) does not match number of views ({num_views}).")
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        sys.exit(1)
        
    # 4. Decompose matrices to ASTRA vectors
    sino_gb = (H * num_views * W * 4) / (1024 ** 3)
    vol_gb = (Nx * Ny * Nz * 4) / (1024 ** 3)
    print(f"Projection Summary: {num_views} projections, each {W}x{H} pixels, data type: {p_meas_sino.dtype}")
    print(f"Sinogram Memory Size: {sino_gb:.3f} GB")
    print(f"Reconstruction Volume Dimensions: {Nx}x{Ny}x{Nz} voxels")
    print(f"Reconstruction Volume Memory Size: {vol_gb:.3f} GB")
    
    print("Decomposing projection matrices to ASTRA vectors...")
    vectors = np.zeros((num_views, 12))
    for i in range(num_views):
        vectors[i] = decompose_to_astra_vector(P_list[i], Nx, Ny, Nz, W, H)
        
    # 5. Perform FDK reconstruction
    vol_geom = astra.create_vol_geom(Ny, Nx, Nz)
    proj_geom = astra.create_proj_geom("cone_vec", H, W, vectors)
    
    proj_id = astra.data3d.create("-sino", proj_geom, p_meas_sino)
    
    # Clean up memory/references since ASTRA has its copy
    del p_meas_sino
    if 'projs' in locals():
        del projs
    if 'projs_att' in locals():
        del projs_att
    if 'data' in locals():
        del data
    gc.collect()
        
    rec_id = astra.data3d.create("-vol", vol_geom)
    
    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    
    filter_type = config.get("filter_type", "ram-lak").lower()
    cfg["FilterType"] = filter_type
    
    filters = {
        "ram-lak": "default high-pass, sharpest features but high noise",
        "shepp-logan": "slightly softer high-pass, good balance of resolution and noise",
        "cosine": "soft kernel, reduces high-frequency noise",
        "hamming": "softer kernel, smooth details with lower noise",
        "hann": "very soft kernel, smooths features and significantly reduces noise"
    }
    print("Running FDK reconstruction:")
    for key, desc in filters.items():
        prefix = "*" if key == filter_type else " "
        print(f"  {prefix} {key}: {desc}")
    
    alg_id = astra.algorithm.create(cfg)
    t0 = time.time()
    astra.algorithm.run(alg_id)
    print(f"FDK reconstruction completed in {time.time() - t0:.3f} seconds.")
    
    # Retrieve reconstructed volume directly
    vol_ref = astra.data3d.get(rec_id)
    # Clip and copy in a single operation
    vol = np.clip(vol_ref, 0.0, None)
    del vol_ref

    # 5b. Apply cylinder mask if requested
    mask_val = config.get("mask_cylinder", 0.0)
    # Support both old boolean format and new float format
    if isinstance(mask_val, bool):
        scale_factor = 1.0 if mask_val else 0.0
    else:
        try:
            scale_factor = float(mask_val)
        except (ValueError, TypeError):
            scale_factor = 0.0

    if scale_factor > 0.0:
        print(f"Computing circumscribed cylinder mask (scale factor: {scale_factor})...")

        from recon_coverage import ReconstructionVolumeEstimator
        
        # ReconstructionVolumeEstimator needs the original projection matrices in world coordinates
        P_original = parse_ompl_matrices(resolve_relative_to_config(config["ompl_file"]))
        P_original = P_original[::skip]
        if abs(downscale_factor - 1.0) >= 1e-5:
            H_downscale = np.array([
                [downscale_factor, 0.0, 0.0],
                [0.0, downscale_factor, 0.0],
                [0.0, 0.0, 1.0]
            ])
            P_original = [H_downscale @ P for P in P_original]
        
        detector_size = (W, H)
        
        # Select at most 18 equally spaced projection matrices to compute the estimator efficiently
        n_proj = len(P_original)
        if n_proj > 18:
            indices = np.round(np.linspace(0, n_proj - 1, 18)).astype(int)
            P_original_subset = [P_original[i] for i in indices]
        else:
            P_original_subset = P_original
            
        estimator = ReconstructionVolumeEstimator(
            P_original_subset,
            detector_size=detector_size
        )
        vols = estimator.estimate()
        cyl = vols.cylinder_cc
        
        # Voxel grid dimensions (in ASTRA, shape is (Ny, Nx, Nz))
        Ny, Nx, Nz = vol.shape
        y_grid = np.arange(Ny, dtype=np.float32)
        x_grid = np.arange(Nx, dtype=np.float32)
        z_grid = np.arange(Nz, dtype=np.float32)
        
        # Prepare 2D coordinates for fast slice-by-slice broadcasting
        x_2d = x_grid[:, None]  # Shape (Nx, 1)
        z_2d = z_grid[None, :]  # Shape (1, Nz)
        
        M = np.array(config.get("model_matrix", np.eye(4).tolist()), dtype=np.float32)
        
        p0 = np.array(cyl.point0, dtype=np.float32)
        p1 = np.array(cyl.point1, dtype=np.float32)
        radius = float(cyl.radius)

        # Scale cylinder from isocenter
        _, isocenter = estimator.circular_scan_heuristics()
        isocenter = np.array(isocenter, dtype=np.float32)
        p0 = isocenter + scale_factor * (p0 - isocenter)
        p1 = isocenter + scale_factor * (p1 - isocenter)
        radius = scale_factor * radius
        
        v = p1 - p0
        v_len = np.linalg.norm(v)
        v_hat = v / v_len
        
        # Pre-compute constant terms independent of y
        dx_base = M[0, 0] * z_2d + M[0, 1] * x_2d + M[0, 3] - p0[0]
        dy_base = M[1, 0] * z_2d + M[1, 1] * x_2d + M[1, 3] - p0[1]
        dz_base = M[2, 0] * z_2d + M[2, 1] * x_2d + M[2, 3] - p0[2]
        
        radius_sq = radius * radius
        
        # Slice-by-slice computation loop
        for y_idx in range(Ny):
            y_val = y_grid[y_idx]
            
            # Compute coordinates for this slice
            dx = dx_base + M[0, 2] * y_val
            dy = dy_base + M[1, 2] * y_val
            dz = dz_base + M[2, 2] * y_val
            
            # Vector math for distance to cylinder axis
            t = dx * v_hat[0] + dy * v_hat[1] + dz * v_hat[2]
            d_sq = (dx*dx + dy*dy + dz*dz) - t*t
            
            # Apply mask to the current slice in-place
            mask = d_sq > radius_sq
            vol[y_idx, mask] = 0.0
            
        print("Cylinder mask applied successfully.")

    # Cleanup ASTRA resources immediately to free GPU and host memory
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    astra.clear()
    gc.collect()
    
    # Clean up temp file
    if temp_filename and os.path.exists(temp_filename):
        try:
            os.remove(temp_filename)
            print("Deleted temporary sinogram file.")
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_filename}: {e}")
            
    # 6. Save reconstructed volume in X, Y, Z order
    vol_xyz = np.transpose(vol, (2, 1, 0))
    output_path = resolve_relative_to_config(config.get("output_file", "reconstruction.nrrd"))
    print(f"Saving final volume to: {output_path}")
    if output_path.lower().endswith(".nrrd"):
        header = {'encoding': 'raw'}
        if "model_matrix" in config:
            M = np.array(config["model_matrix"])
            # The top-left 3x3 of model_matrix represents the space directions
            space_dirs = M[:3, :3].tolist()
            # The last (translation) column of model_matrix represents the space origin
            origin = M[:3, 3].tolist()
            
            header['space dimension'] = 3
            header['space directions'] = space_dirs
            header['space origin'] = origin
            
        nrrd.write(output_path, vol_xyz, header=header)
    elif output_path.lower().endswith(".npz"):
        np.savez_compressed(output_path, volume=vol_xyz)
    else:
        np.save(output_path, vol_xyz)

if __name__ == "__main__":
    main()
