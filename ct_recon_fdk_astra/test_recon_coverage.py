import unittest
import numpy as np
import os
from recon_coverage import (
    reconstruction_volumes,
    _load_ompl,
    ReconstructionVolumeEstimator
)
from scipy.spatial import HalfspaceIntersection

class TestReconCoverageGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate path to fullscan_90views_600x400.ompl
        cls.ompl_path = os.path.join("example_data", "fullscan_90views_600x400.ompl")
        if not os.path.exists(cls.ompl_path):
            raise unittest.SkipTest(f"Test data not found at {cls.ompl_path}")

        # Load projection matrices and geometry
        cls.Ps, cls.detector_size, cls.pixel_spacing = _load_ompl(cls.ompl_path)
        cls.vols = reconstruction_volumes(
            projection_matrices=cls.Ps,
            detector_size=cls.detector_size,
            pixel_spacing=cls.pixel_spacing
        )
        
        # Instantiate estimator directly to access helper functions/internal state
        cls.estimator = ReconstructionVolumeEstimator(
            cls.Ps,
            cls.detector_size,
            pixel_spacing=cls.pixel_spacing
        )
        
        # Vertices of the coverage polyhedron (intersection of halfspaces)
        cls.halfspaces = -cls.estimator._planes
        cls.center = cls.estimator.john_ellipsoid()[:3, 3]
        hs = HalfspaceIntersection(cls.halfspaces, cls.center)
        cls.polyhedron_vertices = hs.intersections

    def test_basic_geometry_dimensions(self):
        # Verify sizes and shapes of results
        self.assertEqual(len(self.Ps), 90)
        self.assertEqual(self.detector_size, (600, 400))
        self.assertAlmostEqual(self.pixel_spacing, 0.66, places=2)
        
        # Verify matrix shapes
        for vol in [self.vols.box_aabb, self.vols.box_obb, self.vols.box_aab_inscribed, self.vols.box_obb_inscribed, self.vols.cylinder_cc]:
            self.assertEqual(vol.model_matrix.shape, (4, 4))
            self.assertEqual(vol.number_of_voxels.shape, (3,))
            self.assertTrue((vol.number_of_voxels > 0).all())

    def test_inscribed_vs_bounding_volume_consistency(self):
        # Calculate volume of boxes as product of diagonal components/eigenvalues of scale part
        def box_volume(M, number_of_voxels):
            # Since M maps [0, N]^3 to millimeters, we must multiply the single voxel volume
            # by the total voxel count.
            return abs(np.linalg.det(M[:3, :3])) * np.prod(number_of_voxels)

        vol_aabb = box_volume(self.vols.box_aabb.model_matrix, self.vols.box_aabb.number_of_voxels)
        vol_aab_inscribed = box_volume(self.vols.box_aab_inscribed.model_matrix, self.vols.box_aab_inscribed.number_of_voxels)
        vol_obb = box_volume(self.vols.box_obb.model_matrix, self.vols.box_obb.number_of_voxels)
        vol_obb_inscribed = box_volume(self.vols.box_obb_inscribed.model_matrix, self.vols.box_obb_inscribed.number_of_voxels)

        # Inscribed boxes must be smaller than circumscribed bounding boxes
        self.assertLessEqual(vol_aab_inscribed, vol_aabb)
        self.assertLessEqual(vol_obb_inscribed, vol_obb)

    def test_cylinder_axis_alignment(self):
        # For the cylinder, the Z-axis (third column of the model matrix)
        # must be collinear with the estimated trajectory rotation axis.
        axis_direction = self.estimator._axis_direction
        
        for cyl in [self.vols.cylinder_cc]:
            z_axis = cyl.model_matrix[:3, 2]
            z_axis_dir = z_axis / np.linalg.norm(z_axis)
            
            # Check collinearity: absolute dot product should be close to 1
            dot_product = abs(np.dot(z_axis_dir, axis_direction))
            self.assertAlmostEqual(dot_product, 1.0, places=5)

    def test_inscribed_halfspace_containment(self):
        # Inscribed volumes must lie entirely within the halfspaces (planes) defined by the trajectory.
        # Check all vertices of the inscribed boxes and inscribed cylinder endpoints.
        planes = self.estimator._planes # each row represents a plane equation n_x x + n_y y + n_z z + d >= 0
        
        # Helper to get the 8 corners of a box given its model matrix and voxel grid dims
        def get_box_corners(M, number_of_voxels):
            nx, ny, nz = number_of_voxels
            corners_voxel = np.array([
                [0, 0, 0, 1],
                [nx, 0, 0, 1],
                [0, ny, 0, 1],
                [0, 0, nz, 1],
                [nx, ny, 0, 1],
                [nx, 0, nz, 1],
                [0, ny, nz, 1],
                [nx, ny, nz, 1]
            ]).T
            corners_world = M @ corners_voxel
            return corners_world[:3, :].T

        # Test AAB Inscribed
        for pt in get_box_corners(self.vols.box_aab_inscribed.model_matrix, self.vols.box_aab_inscribed.number_of_voxels):
            for plane in planes:
                dist = plane[:3] @ pt + plane[3]
                # With numerical tolerances, the distance must be non-negative
                self.assertGreaterEqual(dist, -1e-4)

        # Test OBB Inscribed
        for pt in get_box_corners(self.vols.box_obb_inscribed.model_matrix, self.vols.box_obb_inscribed.number_of_voxels):
            for plane in planes:
                dist = plane[:3] @ pt + plane[3]
                self.assertGreaterEqual(dist, -1e-4)



    def test_circumscribed_polyhedron_coverage(self):
        # Bounding / Circumscribed volumes must completely enclose the polyhedron of intersection
        # (meaning all vertices of the polyhedron lie inside the bounding volume bounds).
        
        # Helper to check if a point lies inside a box defined by model matrix M and voxel counts
        def is_point_inside_box(pt, M, number_of_voxels):
            # Transform point to voxel space: q = M^-1 * [pt; 1]
            q = np.linalg.solve(M, np.append(pt, 1.0))
            # In voxel space, the box is [0, Nx] x [0, Ny] x [0, Nz]
            return (q[:3] >= -1e-4).all() and (q[:3] <= number_of_voxels + 1e-4).all()

        # Check that all polyhedron vertices are inside box_aabb
        for pt in self.polyhedron_vertices:
            self.assertTrue(is_point_inside_box(pt, self.vols.box_aabb.model_matrix, self.vols.box_aabb.number_of_voxels))

        # Check that all polyhedron vertices are inside box_obb
        for pt in self.polyhedron_vertices:
            self.assertTrue(is_point_inside_box(pt, self.vols.box_obb.model_matrix, self.vols.box_obb.number_of_voxels))

        # Check that all polyhedron vertices are inside the cylinder_cc
        cyl_cc = self.vols.cylinder_cc
        p0 = cyl_cc.point0
        p1 = cyl_cc.point1
        r = cyl_cc.radius
        
        axis = p1 - p0
        length = np.linalg.norm(axis)
        axis_dir = axis / length
        
        for pt in self.polyhedron_vertices:
            # Vector from point0 to point
            v = pt - p0
            # Project onto axis
            proj_len = np.dot(v, axis_dir)
            # The projection must lie between 0 and length (along the height of the cylinder)
            self.assertGreaterEqual(proj_len, -1e-4)
            self.assertLessEqual(proj_len, length + 1e-4)
            
            # Distance to axis (radial distance) must be <= radius
            perp_vec = v - proj_len * axis_dir
            radial_dist = np.linalg.norm(perp_vec)
            self.assertLessEqual(radial_dist, r + 1e-4)

    def test_john_quadric_projection(self):
        # 1. Compute John's ellipsoid using the estimator
        ellipsoid = self.estimator.john_ellipsoid()
        self.assertEqual(ellipsoid.shape, (4, 4))
        
        # 2. Convert ellipsoid (B, c) to quadric Q (symmetric 4x4)
        B_val = ellipsoid[:3, :3]
        c_val = ellipsoid[:3, 3]
        Sigma_inv = np.linalg.inv(B_val @ B_val.T)
        Q = np.zeros((4, 4))
        Q[:3, :3] = Sigma_inv
        Q[:3, 3] = -Sigma_inv @ c_val
        Q[3, :3] = -c_val.T @ Sigma_inv
        Q[3, 3] = c_val.T @ Sigma_inv @ c_val - 1.0
        
        # Verify symmetry of Q
        np.testing.assert_allclose(Q, Q.T, atol=1e-7)
        
        # 3. Project to a 2D conic using the first projection matrix P
        P = self.Ps[0]
        P_mat = P
        
        Q_inv = np.linalg.inv(Q)
        C_inv = P_mat @ Q_inv @ P_mat.T
        
        # C_inv must be invertible (non-singular) for a valid conic
        self.assertGreater(abs(np.linalg.det(C_inv)), 1e-12)
        
        C = np.linalg.inv(C_inv)
        # Verify symmetry of C
        np.testing.assert_allclose(C, C.T, atol=1e-7)

    def test_quadric_plane_intersection(self):
        # 1. Compute John's ellipsoid and quadric Q
        ellipsoid = self.estimator.john_ellipsoid()
        B_val = ellipsoid[:3, :3]
        c_val = ellipsoid[:3, 3]
        Sigma_inv = np.linalg.inv(B_val @ B_val.T)
        Q = np.zeros((4, 4))
        Q[:3, :3] = Sigma_inv
        Q[:3, 3] = -Sigma_inv @ c_val
        Q[3, :3] = -c_val.T @ Sigma_inv
        Q[3, 3] = c_val.T @ Sigma_inv @ c_val - 1.0
        
        # 2. Get plane of rotation normal and isocenter from estimator
        axis_direction, isocenter = self.estimator.circular_scan_heuristics()
        
        # 3. Construct basis matrix H_plane
        a = axis_direction / np.linalg.norm(axis_direction)
        v = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, v)) > 0.9:
            v = np.array([0.0, 1.0, 0.0])
        r1 = np.cross(v, a)
        r1 = r1 / np.linalg.norm(r1)
        r2 = np.cross(a, r1)
        r2 = r2 / np.linalg.norm(r2)
        
        H = np.zeros((4, 3))
        H[:3, 0] = r1
        H[:3, 1] = r2
        H[:3, 2] = isocenter
        H[3, 2] = 1.0
        
        # 4. Compute local conic C_plane
        C_plane = H.T @ Q @ H
        self.assertEqual(C_plane.shape, (3, 3))
        np.testing.assert_allclose(C_plane, C_plane.T, atol=1e-7)
        
        # 5. Project to screen under a viewport-like projection matrix P (not on the plane)
        # Construct a camera looking at the isocenter from above the plane
        P_viewport = np.array([
            [1.0, 0.0, 0.0, -isocenter[0]],
            [0.0, 1.0, 0.0, -isocenter[1]],
            [0.0, 0.0, 1.0, -isocenter[2] + 1000.0]
        ])
        P_eff = P_viewport @ H
        
        C_plane_inv = np.linalg.inv(C_plane)
        C_screen_inv = P_eff @ C_plane_inv @ P_eff.T
        self.assertGreater(abs(np.linalg.det(C_screen_inv)), 1e-12)
        
        C_screen = np.linalg.inv(C_screen_inv)
        np.testing.assert_allclose(C_screen, C_screen.T, atol=1e-7)
        
        # 6. Verify that projection matrix on the plane (e.g. scan camera) is degenerate
        P_scan = self.Ps[0]
        P_eff_scan = P_scan @ H
        C_screen_inv_scan = P_eff_scan @ C_plane_inv @ P_eff_scan.T
        self.assertLess(abs(np.linalg.det(C_screen_inv_scan)), 1e-7)

    def test_draw_cylinder_math(self):
        # 1. Cylinder parameters
        point0 = np.array([0.0, 0.0, 0.0])
        point1 = np.array([0.0, 0.0, 100.0])
        radius = 25.0
        
        # 2. Orthonormal basis vectors
        v_dir = point1 - point0
        len_v = np.linalg.norm(v_dir)
        axis = v_dir / len_v
        
        tmp = np.array([1.0, 0.0, 0.0])
        u1 = np.cross(tmp, axis)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(axis, u1)
        u2 /= np.linalg.norm(u2)
        
        # 3. Viewport-like projection matrix (camera center at [200.0, 0.0, -500.0])
        P_viewport = np.array([
            [1.0, 0.0, 0.0, -200.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 500.0]
        ])
        
        # 4. Project conics
        C_local_inv = np.diag([1.0, 1.0, -1.0 / (radius**2)])
        for p_center in [point0, point1]:
            H = np.zeros((4, 3))
            H[:3, 0] = u1
            H[:3, 1] = u2
            H[:3, 2] = p_center
            H[3, 2] = 1.0
            
            P_eff = P_viewport @ H
            C_screen_inv = P_eff @ C_local_inv @ P_eff.T
            self.assertGreater(abs(np.linalg.det(C_screen_inv)), 1e-12)
            
        # 5. Silhouette angles
        from recon_coverage import _camera_center
        C_cam = _camera_center(P_viewport)
        V = C_cam - point0
        A = np.dot(u1, V)
        B = np.dot(u2, V)
        r_val = np.sqrt(A**2 + B**2)
        
        self.assertGreaterEqual(r_val, radius)
        phi = np.arctan2(B, A)
        alpha = np.arccos(radius / r_val)
        
        # Check silhouette points on cylinder surface
        for theta in [phi + alpha, phi - alpha]:
            dx = radius * np.cos(theta) * u1 + radius * np.sin(theta) * u2
            X = point0 + dx
            normal = np.cos(theta) * u1 + np.sin(theta) * u2
            ray = C_cam - X
            self.assertAlmostEqual(np.dot(normal, ray), 0.0, places=5)
            
        # 6. Test actual draw_cylinder call
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gui"))
        from reconstruction_gui import draw_cylinder
        svg_out = draw_cylinder(P_viewport, point0, point1, radius, stroke="orange")
        self.assertIn("<ellipse", svg_out)
        self.assertIn("<line", svg_out)

if __name__ == "__main__":
    unittest.main()
