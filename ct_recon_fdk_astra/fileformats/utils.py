import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix

def get_camera_center(P):
    """
    Returns the 4D homogeneous coordinate of the camera center (nullspace of 3x4 P).
    """
    _, _, vh = np.linalg.svd(P)
    c = vh[-1, :]
    return c / c[3]

def get_camera_direction(P):
    """
    Returns the unit principal ray/camera direction vector.
    """
    direction = P[2, :3]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return direction / norm

def camera_look_at(K, eye, center, up=np.array([0.0, 1.0, 0.0])):
    """
    Constructs a 3x4 projection matrix using intrinsic matrix K, eye, center, and up vector.
    """
    z = eye - center
    z_norm = np.linalg.norm(z)
    if z_norm > 1e-8:
        z = z / z_norm
    else:
        z = np.array([0.0, 0.0, 1.0])
        
    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm > 1e-8:
        x = x / x_norm
    else:
        x = np.array([1.0, 0.0, 0.0])
        
    y = np.cross(z, x)
    
    R = np.vstack([x, y, z])
    t = -R @ eye
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt

def ctCircularTrajectoryToParameters(Ps, detector_pixel_spacing=1.0):
    """
    Estimates parameters of a circular trajectory as defined in the C++ ProjTable class.
    """
    n = len(Ps)
    if n < 3:
        return {}
        
    source_positions = []
    A = np.zeros((n, 4))
    for i, P in enumerate(Ps):
        P_mat = P.P if isinstance(P, ProjectionMatrix) else P
        C = get_camera_center(P_mat)
        source_positions.append(C)
        norm_C3 = np.linalg.norm(C[:3])
        A[i, :] = C / (norm_C3 if norm_C3 > 1e-8 else 1.0)
        
    # Linear plane fit to source trajectory
    _, _, vh = np.linalg.svd(A)
    plane_fit = vh[-1, :]
    rotation_plane = plane_fit / (np.linalg.norm(plane_fit[:3]) if np.linalg.norm(plane_fit[:3]) > 1e-8 else 1.0)
    
    P0 = Ps[0]
    P1 = Ps[n // 4]
    P2 = Ps[n // 2]
    
    def get_norm_P(P):
        P_mat = P.P if isinstance(P, ProjectionMatrix) else P
        norm_val = np.linalg.norm(P_mat[2, :3])
        return P_mat / (norm_val if norm_val > 1e-8 else 1.0)
        
    V0 = get_camera_direction(get_norm_P(P0))
    V1 = get_camera_direction(get_norm_P(P1))
    V2 = get_camera_direction(get_norm_P(P2))
    
    plane_normal = rotation_plane[:3]
    plane_proj = np.eye(3) - np.outer(plane_normal, plane_normal)
    
    # Determine direction of rotation and correct normal axis
    proj_V1 = plane_proj @ V1
    proj_V2 = plane_proj @ V2
    cross_prod = np.cross(proj_V1, proj_V2)
    if np.dot(plane_normal, cross_prod) < 0:
        rotation_plane = -rotation_plane
        plane_normal = rotation_plane[:3]
        
    V0_in_plane = plane_proj @ V0
    V0_norm = np.linalg.norm(V0_in_plane)
    if V0_norm > 1e-8:
        V0_in_plane /= V0_norm
    else:
        V0_in_plane = np.array([1.0, 0.0, 0.0])
        
    primary_angles = []
    secondary_angles = []
    source_to_iso_center_distances = []
    source_to_detector_distances = []
    
    for i in range(n):
        P_curr = get_norm_P(Ps[i])
        Ci = source_positions[i]
        sid = np.linalg.norm(Ci[:3])
        source_to_iso_center_distances.append(sid)
        
        Vi = get_camera_direction(P_curr)
        Vi_in_plane = plane_proj @ Vi
        Vi_norm = np.linalg.norm(Vi_in_plane)
        if Vi_norm > 1e-8:
            Vi_in_plane /= Vi_norm
        else:
            Vi_in_plane = np.array([1.0, 0.0, 0.0])
            
        # Primary angle of rotation
        pa_cos = np.dot(V0_in_plane, Vi_in_plane)
        pa_cos = np.clip(pa_cos, -1.0, 1.0)
        primary_angle = np.arccos(pa_cos)
        if np.dot(np.cross(V0_in_plane, Vi_in_plane), plane_normal) < 0:
            primary_angle = 2.0 * np.pi - primary_angle
        primary_angle = np.degrees(primary_angle)
        primary_angles.append(primary_angle)
        
        # Secondary angle of rotation
        cos_angle = np.dot(Vi, Vi_in_plane)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        secondary_angle = np.arccos(cos_angle)
        if np.dot(plane_normal, np.cross(Vi, Vi_in_plane)) < 0:
            secondary_angle = -secondary_angle
        secondary_angle = np.degrees(secondary_angle)
        secondary_angles.append(secondary_angle)
        
        # Source to detector distance (sdd)
        P_obj = Ps[i] if isinstance(Ps[i], ProjectionMatrix) else ProjectionMatrix(P_curr, pixel_spacing=detector_pixel_spacing)
        K, _, _, _ = P_obj.decomposition()
        sdd = detector_pixel_spacing * 0.5 * (np.abs(K[0,0]) + np.abs(K[1,1]))
        source_to_detector_distances.append(sdd)
        
    return {
        "plane_of_rotation": rotation_plane,
        "primary_angles": primary_angles,
        "secondary_angles": secondary_angles,
        "source_to_iso_center_distances": source_to_iso_center_distances,
        "source_to_detector_distances": source_to_detector_distances,
        "source_positions": source_positions
    }

def makeCircularTrajectory(n_proj, sid, sdd, n_u, n_v, max_angle, pixel_spacing):
    """
    Creates a perfect circular trajectory as defined in the C++ ProjTable class.
    """
    f = sdd / pixel_spacing
    K = np.array([
        [f, 0.0, n_u * 0.5],
        [0.0, f, n_v * 0.5],
        [0.0, 0.0, 1.0]
    ])
    
    T_rot_x = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    Ps = []
    for i in range(n_proj):
        primary_angle = np.radians(i * (max_angle / n_proj))
        eye = np.array([sid * np.cos(primary_angle), 0.0, sid * np.sin(primary_angle)])
        center = np.array([0.0, 0.0, 0.0])
        P_mat = camera_look_at(K, eye, center, up=np.array([0.0, 1.0, 0.0]))
        P_mat = P_mat @ T_rot_x
        P_mat /= np.linalg.norm(P_mat[2, :3])
        Ps.append(ProjectionMatrix(P_mat, image_size=(n_u, n_v), pixel_spacing=pixel_spacing))
    return Ps
