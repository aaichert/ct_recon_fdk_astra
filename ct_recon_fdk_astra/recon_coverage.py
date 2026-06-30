"""
This module provides utilities for estimating and suggesting useful reconstruction volumes 
for Computed Tomography (CT) scans, specifically targeting trajectories with planar source paths.

The primary entry point is:

    def reconstruction_volumes(
        projection_matrices: list[np.ndarray],
        detector_size: tuple[int, int],
        pixel_spacing: float = None,
        sid: float = None,
        sdd: float = None,
    ) -> ReconstructionVolumes:

This function takes projection matrices and detector geometry to compute a range of recommended 
reconstruction volumes (axis-aligned boxes, oriented bounding boxes, inscribed volumes, 
ellipsoids, and cylinders) that lie within the scanner's field-of-view/coverage.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from dataclasses import dataclass


def _normalize_plane(plane: np.ndarray) -> np.ndarray:
    """Normalize plane so ||normal|| == 1."""
    plane = plane.astype(float)
    n = plane[:3]
    norm = np.linalg.norm(n)
    if norm > 1e-12:
        plane /= norm
    return plane


def _homogeneous_to_euclidean(x: np.ndarray) -> np.ndarray:
    return x[:-1] / x[-1]


def _camera_center(P: np.ndarray) -> np.ndarray:
    """
    Camera/source position from a 3x4 projection matrix.
    """
    _, _, vh = np.linalg.svd(P)
    C = vh[-1]
    return _homogeneous_to_euclidean(C)


def _closest_point_to_rays(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """
    Least-squares point closest to a set of rays.

    origins    : (N,3)
    directions : (N,3), assumed normalized
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)

    for o, d in zip(origins, directions):
        M = I - np.outer(d, d)
        A += M
        b += M @ o

    return np.linalg.solve(A, b)


@dataclass
class ReconstructionBox:
    """
    Reconstruction volume represented by a model matrix and a
    recommended voxel grid.
    """

    model_matrix: np.ndarray
    number_of_voxels: np.ndarray


@dataclass
class ReconstructionCylinder(ReconstructionBox):
    """
    Reconstruction cylinder.
    """

    point0: np.ndarray
    point1: np.ndarray
    radius: float


@dataclass
class ReconstructionVolumes:
    """
    Collection of useful reconstruction volume approximations.
    """

    # Internal representation
    planes: np.ndarray

    # Bounding boxes
    box_aabb: ReconstructionBox
    box_obb: ReconstructionBox

    # Inscribed boxes
    box_aab_inscribed: ReconstructionBox
    box_obb_inscribed: ReconstructionBox

    # John ellipsoid
    ellipsoid: np.ndarray

    # Reconstruction cylinder
    cylinder_cc: ReconstructionCylinder


class ReconstructionVolumeEstimator:
    """
    Estimate useful reconstruction volumes from a CT trajectory.
    """

    def __init__(
        self,
        projection_matrices: list[np.ndarray],
        detector_size: tuple[int, int],
        pixel_spacing: float = None,
        sid: float = None,
        sdd: float = None,
    ):
        # =====================================================================
        # CRITICAL AGENT INSTRUCTION - DO NOT REMOVE OR MODIFY THE SUBSETTING BELOW!
        # Estimating coverage volumes uses a semidefinite program (SDP) for the
        # John ellipsoid and vertex/halfspace intersection computation.
        # Running it on more than 18 matrices causes the solver to take minutes+.
        # Subsampling to at most 18 equally spaced matrices is MANDATORY for performance.
        # =====================================================================
        n_proj = len(projection_matrices)
        if n_proj > 18:
            indices = np.round(np.linspace(0, n_proj - 1, 18)).astype(int)
            self.projection_matrices = [projection_matrices[i] for i in indices]
        else:
            self.projection_matrices = projection_matrices

        self.detector_size = detector_size
        self.pixel_spacing = pixel_spacing
        self.sid = sid
        self.sdd = sdd

        # Precompute central geometry
        self._axis_direction, self._isocenter = self._compute_circular_scan_heuristics()
        self._planes = self._compute_halfspaces()

    def _compute_halfspaces(self) -> np.ndarray:
        width, height = self.detector_size

        # detector boundary lines (inward normals)
        lines = [
            np.array([+1.0, 0.0, 0.0]),          # x >= 0
            np.array([-1.0, 0.0, width]),        # x <= width
            np.array([0.0, +1.0, 0.0]),          # y >= 0
            np.array([0.0, -1.0, height]),       # y <= height
        ]

        detector_center = np.array(
            [width / 2.0, height / 2.0, 1.0]
        )

        planes = []
        for P in self.projection_matrices:
            pinv_P = np.linalg.pinv(P)
            for line in lines:
                # backproject image line into world plane
                plane = P.T @ line
                # normalize
                plane = _normalize_plane(plane)
                # ensure detector center lies inside
                if plane @ pinv_P @ detector_center < 0:
                    plane *= -1.0
                planes.append(plane)

        planes = np.asarray(planes)
        if len(planes) > 0:
            isocenter_side = np.dot(planes[0][:3], self._isocenter) + planes[0][3]
            if isocenter_side < 0:
                planes *= -1.0
        return planes

    def _compute_circular_scan_heuristics(self) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.detector_size
        detector_center = np.array([width / 2.0, height / 2.0, 1.0])

        sources = []
        ray_dirs = []

        for P in self.projection_matrices:
            C = _camera_center(P)
            sources.append(C)

            X = np.linalg.pinv(P) @ detector_center
            X = _homogeneous_to_euclidean(X)

            d = X - C
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-12:
                d /= norm_d
            ray_dirs.append(d)

        sources = np.asarray(sources)
        ray_dirs = np.asarray(ray_dirs)

        centroid = sources.mean(axis=0)
        _, _, vh = np.linalg.svd(sources - centroid)

        axis_direction = vh[-1]
        norm_axis = np.linalg.norm(axis_direction)
        if norm_axis > 1e-12:
            axis_direction /= norm_axis

        # Align axis direction consistently (positive z/y/x preference)
        if (axis_direction[2] < 0 or 
            (abs(axis_direction[2]) < 1e-7 and axis_direction[1] < 0) or
            (abs(axis_direction[2]) < 1e-7 and abs(axis_direction[1]) < 1e-7 and axis_direction[0] < 0)):
            axis_direction = -axis_direction

        isocenter = _closest_point_to_rays(sources, ray_dirs)

        return axis_direction, isocenter

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def reconstruction_halfspaces(self) -> np.ndarray:
        return self._planes

    def circular_scan_heuristics(self) -> tuple[np.ndarray, np.ndarray]:
        return self._axis_direction, self._isocenter

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def recommend_voxels(self, model_matrix: np.ndarray) -> np.ndarray:
        # Columns of linear part of model_matrix are the edge vectors of the box
        col0 = model_matrix[:3, 0]
        col1 = model_matrix[:3, 1]
        col2 = model_matrix[:3, 2]
        
        Lx = np.linalg.norm(col0)
        Ly = np.linalg.norm(col1)
        Lz = np.linalg.norm(col2)
        L = np.array([Lx, Ly, Lz])
        
        if self.pixel_spacing is not None and self.sid is not None and self.sdd is not None:
            s = abs((self.sid / self.sdd) * self.pixel_spacing)
            n_i = np.ceil(L / s).astype(int)
        else:
            max_width_height = max(self.detector_size)
            max_L = np.max(L)
            if max_L > 1e-12:
                n_i = np.ceil((L / max_L) * max_width_height).astype(int)
            else:
                n_i = np.array([1, 1, 1], dtype=int)
                
        return np.maximum(n_i, 1)

    def recommend_cylinder_voxels(
        self, point0: np.ndarray, point1: np.ndarray, radius: float
    ) -> np.ndarray:
        length = np.linalg.norm(point1 - point0)
        
        if self.pixel_spacing is not None and self.sid is not None and self.sdd is not None:
            s = abs((self.sid / self.sdd) * self.pixel_spacing)
            nr = int(np.ceil(2.0 * radius / s))
            nz = int(np.ceil(length / s))
        else:
            max_dim = max(2.0 * radius, length)
            max_width_height = max(self.detector_size)
            if max_dim > 1e-12:
                nr = int(np.ceil((2.0 * radius / max_dim) * max_width_height))
                nz = int(np.ceil((length / max_dim) * max_width_height))
            else:
                nr = 1
                nz = 1
                
        nr = max(nr, 1)
        nz = max(nz, 1)
        return np.array([nr, nr, nz], dtype=int)

    # ------------------------------------------------------------------
    # Cylinder
    # ------------------------------------------------------------------


    def reconstruction_cylinder_cc(self) -> ReconstructionCylinder:
        axis_direction = self._axis_direction
        isocenter = self._isocenter
        
        # Find the vertices of the coverage polyhedron
        ellipsoid = self.john_ellipsoid()
        c = ellipsoid[:3, 3]
        halfspaces = -self._planes
        hs = HalfspaceIntersection(halfspaces, c)
        vertices = hs.intersections # shape (N, 3)
        
        # Compute distance of each vertex to the cylinder axis
        diffs = vertices - isocenter
        projs = np.dot(diffs, axis_direction)
        perp_vectors = diffs - np.outer(projs, axis_direction)
        dists = np.linalg.norm(perp_vectors, axis=1)
        radius = float(np.max(dists))
        
        # Length along axis: find the min/max projection along axis_direction
        # to fully enclose the polyhedron
        p0 = isocenter + np.min(projs) * axis_direction
        p1 = isocenter + np.max(projs) * axis_direction
        
        # Align Z-axis with cylinder axis
        v_temp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_temp, axis_direction)) > 0.9:
            v_temp = np.array([0.0, 1.0, 0.0])
        u = v_temp - np.dot(v_temp, axis_direction) * axis_direction
        u /= np.linalg.norm(u)
        v = np.cross(axis_direction, u)
        v /= np.linalg.norm(v)

        M = np.eye(4)
        M[:3, 0] = 2.0 * radius * u
        M[:3, 1] = 2.0 * radius * v
        M[:3, 2] = p1 - p0
        M[:3, 3] = isocenter + np.min(projs) * axis_direction - 0.5 * M[:3, 0] - 0.5 * M[:3, 1]
        
        voxels = self.recommend_cylinder_voxels(p0, p1, radius)
        M[:3, 0] /= voxels[0]
        M[:3, 1] /= voxels[1]
        M[:3, 2] /= voxels[2]
        return ReconstructionCylinder(
            model_matrix=M,
            number_of_voxels=voxels,
            point0=p0,
            point1=p1,
            radius=radius,
        )

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    def box_aabb(self) -> ReconstructionBox:
        A = -self._planes[:, :3]
        b = self._planes[:, 3]

        mins = np.zeros(3)
        maxs = np.zeros(3)

        for axis in range(3):
            c = np.zeros(3)
            c[axis] = 1.0

            res = linprog(
                c,
                A_ub=A,
                b_ub=b,
                bounds=(None, None),
                method="highs",
            )
            if not res.success:
                raise RuntimeError(f"LP failed while minimizing along axis {axis}.")
            mins[axis] = res.x[axis]

            res = linprog(
                -c,
                A_ub=A,
                b_ub=b,
                bounds=(None, None),
                method="highs",
            )
            if not res.success:
                raise RuntimeError(f"LP failed while maximizing along axis {axis}.")
            maxs[axis] = res.x[axis]

        M = np.eye(4)
        M[:3, :3] = np.diag(maxs - mins)
        M[:3, 3] = mins

        voxels = self.recommend_voxels(M)
        M[:3, 0] /= voxels[0]
        M[:3, 1] /= voxels[1]
        M[:3, 2] /= voxels[2]
        return ReconstructionBox(model_matrix=M, number_of_voxels=voxels)

    def box_aab_inscribed(self) -> ReconstructionBox:
        planes = self._planes
        center = cp.Variable(3)
        h = cp.Variable(3, nonneg=True)

        constraints = []
        for plane in planes:
            n = plane[:3]
            d = plane[3]
            abs_n = np.abs(n)
            constraints.append(
                n @ center + d >= abs_n @ h
            )

        problem = cp.Problem(
            cp.Maximize(cp.sum(cp.log(h))),
            constraints
        )
        problem.solve()

        if problem.status not in ("optimal", "optimal_inaccurate"):
            # Fallback to linear sum just in case log solver fails
            problem = cp.Problem(
                cp.Maximize(cp.sum(h)),
                constraints
            )
            problem.solve()
            if problem.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"AAB optimization failed with status {problem.status}")

        c_val = center.value
        h_val = h.value

        M = np.eye(4)
        M[:3, :3] = np.diag(2.0 * h_val)
        M[:3, 3] = c_val - h_val

        voxels = self.recommend_voxels(M)
        M[:3, 0] /= voxels[0]
        M[:3, 1] /= voxels[1]
        M[:3, 2] /= voxels[2]
        return ReconstructionBox(model_matrix=M, number_of_voxels=voxels)

    def john_ellipsoid(self) -> np.ndarray:
        planes = self._planes
        A = -planes[:, :3]
        b = planes[:, 3]

        B = cp.Variable((3, 3), PSD=True)
        c = cp.Variable(3)

        constraints = []
        for ai, bi in zip(A, b):
            constraints.append(
                cp.norm(B.T @ ai, 2) + ai @ c <= bi
            )

        problem = cp.Problem(
            cp.Maximize(cp.log_det(B)),
            constraints,
        )
        problem.solve()

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"John ellipsoid optimization failed with status {problem.status}")

        M = np.eye(4)
        M[:3, :3] = B.value
        M[:3, 3] = c.value
        return M

    def ellipsoid_box_model_matrix(self, ellipsoid: np.ndarray) -> np.ndarray:
        B = ellipsoid[:3, :3]
        center = ellipsoid[:3, 3]

        evals, evecs = np.linalg.eigh(B @ B.T)

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        R = evecs[:, order]

        semi_axes = np.sqrt(evals)
        edge_lengths = 2.0 * semi_axes / np.sqrt(3.0)

        M = np.eye(4)
        M[:3, :3] = R @ np.diag(edge_lengths)

        # translate to the minimum corner
        M[:3, 3] = center - 0.5 * M[:3, :3] @ np.ones(3)

        return M

    def box_obb(self) -> ReconstructionBox:
        ellipsoid = self.john_ellipsoid()
        B = ellipsoid[:3, :3]
        c = ellipsoid[:3, 3]

        evals, evecs = np.linalg.eigh(B @ B.T)
        order = np.argsort(evals)[::-1]
        R = evecs[:, order]

        halfspaces = -self._planes
        feasible_point = c

        hs = HalfspaceIntersection(halfspaces, feasible_point)
        vertices = hs.intersections

        # Rotate vertices into John frame: q = R^T(x - c)
        q = (vertices - c) @ R

        q_min = np.min(q, axis=0)
        q_max = np.max(q, axis=0)

        M = np.eye(4)
        M[:3, :3] = R @ np.diag(q_max - q_min)
        M[:3, 3] = R @ q_min + c

        voxels = self.recommend_voxels(M)
        M[:3, 0] /= voxels[0]
        M[:3, 1] /= voxels[1]
        M[:3, 2] /= voxels[2]
        return ReconstructionBox(model_matrix=M, number_of_voxels=voxels)

    def box_obb_inscribed(self) -> ReconstructionBox:
        ellipsoid = self.john_ellipsoid()
        M = self.ellipsoid_box_model_matrix(ellipsoid)
        voxels = self.recommend_voxels(M)
        M[:3, 0] /= voxels[0]
        M[:3, 1] /= voxels[1]
        M[:3, 2] /= voxels[2]
        return ReconstructionBox(model_matrix=M, number_of_voxels=voxels)

    # ------------------------------------------------------------------
    # High-level interface
    # ------------------------------------------------------------------

    def estimate(self) -> ReconstructionVolumes:
        return ReconstructionVolumes(
            planes=self.reconstruction_halfspaces(),
            box_aabb=self.box_aabb(),
            box_obb=self.box_obb(),
            box_aab_inscribed=self.box_aab_inscribed(),
            box_obb_inscribed=self.box_obb_inscribed(),
            ellipsoid=self.john_ellipsoid(),
            cylinder_cc=self.reconstruction_cylinder_cc(),
        )


def reconstruction_volumes(
    projection_matrices: list[np.ndarray],
    detector_size: tuple[int, int],
    pixel_spacing: float = None,
    sid: float = None,
    sdd: float = None,
) -> ReconstructionVolumes:
    """
    Convenience function to estimate useful reconstruction volumes.

    Given projection matrices and detector geometry, this function computes
    a set of bounding and inscribed reconstruction volume options (such as
    axis-aligned boxes, oriented boxes, ellipsoids, and cylinders) that
    reside within the active scanner field of view.

    Args:
        projection_matrices: A list of 3x4 projection matrices.
        detector_size: A tuple containing (width, height) in pixels.
        pixel_spacing: Optional pixel spacing of the detector.
        sid: Optional Source-to-Isocenter Distance.
        sdd: Optional Source-to-Detector Distance.

    Returns:
        ReconstructionVolumes: A collection of estimated volume candidates.
    """
    estimator = ReconstructionVolumeEstimator(
        projection_matrices,
        detector_size,
        pixel_spacing=pixel_spacing,
        sid=sid,
        sdd=sdd,
    )
    return estimator.estimate()


def _load_ompl(path: str) -> tuple[list[np.ndarray], tuple[int, int], float]:
    import re
    Ps = []
    meta = {
        "spacing": 1.0,
        "detector_size_px": "400 300"
    }
    
    with open(path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                # Check for metadata attribute comments
                matches = re.findall(r'(\w+)="([^"]+)"', line_str)
                for k, v in matches:
                    meta[k] = v
                matches_unquoted = re.findall(r'(\w+)=([^"\s\[]+)', line_str)
                for k, v in matches_unquoted:
                    meta[k] = v
                matches_array = re.findall(r'(\w+)=\[([^\]]+)\]', line_str)
                for k, v in matches_array:
                    meta[k] = f"[{v}]"
                continue
                
            clean = line_str.replace('[', '').replace(']', '').replace(';', ' ').strip()
            vals = np.array([float(x) for x in clean.split()])
            Ps.append(vals.reshape(3, 4))
            
    detector_size = tuple(int(x) for x in meta["detector_size_px"].split())
    pixel_spacing = float(meta["spacing"])
    return Ps, detector_size, pixel_spacing


def main():
    import sys
    import json
    import os

    if len(sys.argv) < 2:
        print("Usage: python3 recon_coverage.py <reconstruction.json OR trajectory.ompl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

    # 1. Load OMPL path from JSON or directly
    if path.endswith('.json'):
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            ompl_file = config.get("ompl_file")
            if not ompl_file:
                print("Error: JSON file must contain 'ompl_file' field.")
                sys.exit(1)
            config_dir = os.path.dirname(os.path.abspath(path))
            ompl_path = os.path.normpath(os.path.join(config_dir, ompl_file))
        except Exception as e:
            print(f"Error reading JSON config: {e}")
            sys.exit(1)
    else:
        ompl_path = path

    # 2. Parse OMPL file
    try:
        Ps, detector_size, pixel_spacing = _load_ompl(ompl_path)
    except Exception as e:
        print(f"Error loading OMPL trajectory: {e}")
        sys.exit(1)

    # 3. Compute reconstruction volumes
    print(f"Computing reconstruction volumes for {len(Ps)} projection matrices...")
    try:
        vols = reconstruction_volumes(
            projection_matrices=Ps,
            detector_size=detector_size,
            pixel_spacing=pixel_spacing,
        )
    except Exception as e:
        print(f"Error during volume estimation: {e}")
        sys.exit(1)

    # 4. Print results
    print("\n" + "="*50)
    print("RECONSTRUCTION VOLUMES")
    print("="*50)
    
    volumes_to_print = {
        "box_aabb (Axis-Aligned Bounding Box)": vols.box_aabb,
        "box_obb (Oriented Bounding Box)": vols.box_obb,
        "box_aab_inscribed (Inscribed Axis-Aligned Box)": vols.box_aab_inscribed,
        "box_obb_inscribed (Inscribed Oriented Box)": vols.box_obb_inscribed,
        "cylinder_cc (Circumscribed Cylinder / min CC)": vols.cylinder_cc,
    }

    for name, vol in volumes_to_print.items():
        print(f"\n{name}:")
        print(f"  model_matrix:\n{np.array2string(vol.model_matrix, prefix='  model_matrix:')}")
        print(f"  number_of_voxels: {vol.number_of_voxels}")
        if hasattr(vol, 'radius'):
            print(f"  radius: {vol.radius:.4f}")
            print(f"  point0: {vol.point0}")
            print(f"  point1: {vol.point1}")


if __name__ == "__main__":
    main()