# Reconstruction Volume Estimation from CT Projection Matrices

## Overview

This module computes several useful approximations of the reconstructable volume of a CT trajectory from nothing more than

- a list of `3×4` projection matrices
- the detector size `(width, height)`.

The central idea is that every detector boundary backprojects into a plane. The intersection of all detector halfspaces forms a convex polyhedron representing the visible reconstruction volume.

Everything else (cylinders, boxes, ellipsoids, ...) is computed from this single half-space representation.

---

# Architecture

```text
Projection matrices
        │
        ▼
 reconstruction_halfspaces()
        │
        ▼
    Convex polyhedron
 (H-representation only)
        │
        ├─────────────► circular_scan_heuristics()
        │                     │
        │                     ▼
        │              reconstruction_cylinder_cc()
        │
        ├─────────────► box_aabb()
        │
        ├─────────────► box_aab_inscribed()
        │
        ├─────────────► john_ellipsoid()
        │                     │
        │                     ▼
        │         ellipsoid_box_model_matrix()
        │                     │
        │                     ▼
        │          box_obb_inscribed()
        │
        └─────────────► box_obb()
```

---

# Internal Representation

The fundamental representation is

$$
n_xx+n_yy+n_zz+d\ge0.
$$

Every plane is stored as

```python
plane = np.array([nx, ny, nz, d])
```

Normals always point **towards the inside**.

The reconstruction region is

$$
\Omega
=
\bigcap_i
\left\{
x\mid
n_i^Tx+d_i\ge0
\right\}.
$$

No vertices are stored.

---

# Step 1 — Construct Halfspaces

For every projection matrix

$$
P\in\mathbb R^{3\times4},
$$

construct the four detector boundary lines

```text
x = 0
x = width
y = 0
y = height
```

represented as homogeneous image lines

$$
l=(a,b,c)^T.
$$

Backproject each line

$$
\pi=P^Tl.
$$

Normalize

$$
\|n\|=1.
$$

Flip the sign if necessary so that the detector center lies inside.

Store

```python
planes.shape == (4*N, 4)
```

---

# Step 2 — Circular Scan Heuristics

Estimate the rotation axis.

## Source Positions

The camera center is the nullspace of the projection matrix:

$$
C_i=\operatorname{null}(P_i).
$$

Normalize homogeneous coordinates.

---

## Axis Direction

Fit a plane through all source positions.

Compute the centroid

$$
\bar C.
$$

Subtract the centroid and perform an SVD.

The right singular vector corresponding to the smallest singular value is

$$
a,
$$

which is the estimated rotation axis direction.

Normalize

$$
\|a\|=1.
$$

---

## Isocenter

For every projection, backproject the detector center

$$
\left(\frac{w}{2},\frac{h}{2},1\right)^T
$$

using

$$
X_i=\operatorname{pinv}(P_i)x.
$$

Construct the detector-center ray

$$
C_i+t(X_i-C_i).
$$

Estimate the point closest to all rays by solving

$$
\left(
\sum_i
(I-d_id_i^T)
\right)x
=
\sum_i
(I-d_id_i^T)C_i,
$$

where

$$
d_i=\frac{X_i-C_i}{\|X_i-C_i\|}.
$$

This point is the estimated isocenter.

---

# Step 3 — Reconstruction Cylinder

Given

- axis point
- axis direction

compute

$$
n_\perp
=
n-(n\cdot a)a.
$$

The distance from the axis to a plane is

$$
d
=
\frac{n^Tc+d}
{\|n_\perp\|}.
$$

Ignore planes parallel to the axis.

The cylinder radius is

$$
r
=
\min_i d_i.
$$

Return

```python
(point0,
 point1,
 radius)
```

where

```python
point0 = center - L * axis
point1 = center + L * axis
```

for an arbitrary visualization length.

---

# Step 4 — Axis-Aligned Bounding Box (AABB)

Solve six linear programs

```text
min x
max x

min y
max y

min z
max z
```

subject to

$$
Ax\le b.
$$

Construct a `4×4` model matrix mapping

$$
[0,1]^3
$$

to world coordinates.

---

# Step 5 — Axis-Aligned Inscribed Box (AAB)

Unknowns

```text
center
hx
hy
hz
```

For every plane

$$
n^Tc+d
\ge
|n_x|h_x
+
|n_y|h_y
+
|n_z|h_z.
$$

Objective

maximize

$$
h_x+h_y+h_z
$$

or preferably

$$
\log(h_x)+
\log(h_y)+
\log(h_z).
$$

Construct the corresponding `4×4` model matrix.

---

# Step 6 — John Ellipsoid

Represent the ellipsoid as

$$
E=
\left\{
Bu+c
\mid
\|u\|\le1
\right\}.
$$

Unknowns

- symmetric positive definite matrix

$$
B
$$

- center

$$
c.
$$

Optimization

maximize

$$
\log\det(B)
$$

subject to

$$
\|B^Ta_i\|_2
+
a_i^Tc
\le
b_i.
$$

Return the affine transform

```python
M[:3,:3] = B
M[:3,3]  = c
```

which maps the unit sphere into world coordinates.

---

# Step 7 — Largest Box Inside the John Ellipsoid

Diagonalize

$$
BB^T
=
RDR^T.
$$

The semi-axis lengths are

$$
a_i=\sqrt{D_{ii}}.
$$

The largest rectangular box inside an ellipsoid has half-lengths

$$
\frac{a_i}{\sqrt3}.
$$

Construct the corresponding `4×4` model matrix mapping

$$
[0,1]^3
$$

into world coordinates.

This is

```text
box_obb_inscribed
```

---

# Step 8 — Oriented Bounding Box (OBB)

Convert the half-space representation into vertices using

```python
scipy.spatial.HalfspaceIntersection
```

Rotate every vertex into the John frame

$$
q
=
R^T(x-c).
$$

Compute

```text
min(q)
max(q)
```

Build the AABB in this rotated coordinate system.

Transform back into world coordinates.

This yields

```text
box_obb
```

---

# Returned Object

```python
@dataclass
class ReconstructionVolumes:

    #
    # Internal representation
    #
    planes: np.ndarray

    #
    # Boxes
    #
    box_aabb: np.ndarray
    box_aab_inscribed: np.ndarray

    box_obb: np.ndarray
    box_obb_inscribed: np.ndarray

    #
    # Ellipsoid
    #
    ellipsoid: np.ndarray

    #
    # Cylinder
    #
    cylinder_p0: np.ndarray
    cylinder_p1: np.ndarray
    cylinder_radius: float
```

---

# Guiding Philosophy

The module should expose **only geometric primitives**.

Internally, everything is derived from the half-space representation.

The half-space representation is the single source of truth.

Projection matrices are used only once to construct the halfspaces.

Every returned geometric primitive is represented by either

- a **4×4 model matrix**, or
- for cylinders,

```python
(point0,
 point1,
 radius)
```

No meshes, explicit vertex lists, or triangulations should be exposed unless explicitly requested.

# Extension: Recommended Reconstruction Sampling

This section extends the API by also estimating a **reasonable voxel grid** for each reconstruction volume.

The guiding principle is that a reconstruction volume should always consist of

- a geometric description (model matrix or cylinder)
- a recommended discretization (number of voxels)

These two concepts are intentionally separated.

---

# Motivation

A reconstruction volume alone specifies

- position
- orientation
- physical extent

but not

- reconstruction resolution.

Conversely, the detector determines an approximate sampling density, but not the physical region that should be reconstructed.

Both pieces of information are required before allocating a reconstruction volume.

---

# Representation

Every box-like reconstruction volume is represented as

```python
model_matrix
```

mapping

$$
[0,1]^3
$$

into world coordinates,

together with

```python
number_of_voxels = (n_x, n_y, n_z)
```

describing the recommended reconstruction grid.

The voxel counts are **not** part of the default affine transform.

---

# Mapping Voxel Indices to Millimeters

To obtain a transformation matrix $M_{\text{voxel}}$ that maps discrete voxel coordinates $(i, j, k) \in [0, n_x] \times [0, n_y] \times [0, n_z]$ directly to physical coordinates in millimeters, the base `model_matrix` $M$ must be scaled about the $(0,0,0)$ voxel corner:

$$
M_{\text{voxel}}
=
M \cdot \operatorname{diag}\left(\frac{1}{n_x}, \frac{1}{n_y}, \frac{1}{n_z}, 1\right)
$$

In Python, this corresponds to:

```python
# Scale first three columns by 1 / number_of_voxels
M_voxel = box.model_matrix.copy()
nx, ny, nz = box.number_of_voxels
M_voxel[:3, 0] /= nx
M_voxel[:3, 1] /= ny
M_voxel[:3, 2] /= nz
```

---

# Recommended Sampling

The objective is **not** to compute the optimal sampling, but rather a robust default that closely matches the detector sampling.

The preferred strategy is

1. Estimate the detector pixel spacing at the isocenter.
2. Compute the physical edge lengths of the reconstruction volume.
3. Divide the edge lengths by the estimated voxel size.

For a box with edge lengths

$$
L_x,L_y,L_z,
$$

and an estimated voxel size

$$
s,
$$

the recommended voxel counts become

$$
n_i
=
\left\lceil
\frac{L_i}{s}
\right\rceil.
$$

---

# Estimating the Native Voxel Size

Whenever detector calibration is available,

$$
s
=
\frac{SID}{SDD}
\cdot
p,
$$

where

- $SID$ is the source-isocenter distance,
- $SDD$ is the source-detector distance,
- $p$ is the detector pixel spacing.

This approximates the native sampling of the acquisition.

---

# Calibration-Free Fallback

If detector calibration is unavailable,

the voxel counts are scaled such that the longest edge approximately receives

$$
\max(width,height)
$$

voxels.

In other words,

$$
n_i
=
\left\lceil
\frac{L_i}
{\max(L)}
\cdot
\max(width,height)
\right\rceil.
$$

This produces approximately isotropic voxels while preserving the detector resolution.

---

# Cylinder Sampling

The reconstruction cylinder is represented as

- two points defining the axis
- radius
- recommended voxel counts

The suggested discretization is

$$
(n_r,n_r,n_z),
$$

where

$$
n_r
=
\left\lceil
\frac{2r}{s}
\right\rceil,
$$

and

$$
n_z
=
\left\lceil
\frac{L}{s}
\right\rceil.
$$

---

# Design Philosophy

The module should always return both

- geometry
- sampling

This allows downstream reconstruction algorithms to immediately allocate a volume without introducing additional heuristics.

---

# Python Interfaces

```python
from dataclasses import dataclass

import numpy as np


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

    #
    # Internal representation
    #
    planes: np.ndarray

    #
    # Bounding boxes
    #
    box_aabb: ReconstructionBox
    box_obb: ReconstructionBox

    #
    # Inscribed boxes
    #
    box_aab_inscribed: ReconstructionBox
    box_obb_inscribed: ReconstructionBox

    #
    # John ellipsoid
    #
    ellipsoid: np.ndarray

    #
    # Reconstruction cylinder
    #
    cylinder_cc: ReconstructionCylinder


class ReconstructionVolumeEstimator:
    """
    Estimate useful reconstruction volumes from a CT trajectory.
    """

    def __init__(
        self,
        projection_matrices: list[np.ndarray],
        detector_size: tuple[int, int],
    ):
        pass

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def reconstruction_halfspaces(self) -> np.ndarray:
        pass

    def circular_scan_heuristics(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def recommend_voxels(
        self,
        model_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    def recommend_cylinder_voxels(
        self,
        point0: np.ndarray,
        point1: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        pass

    # ------------------------------------------------------------------
    # Cylinder
    # ------------------------------------------------------------------

    def reconstruction_cylinder_cc(
        self,
    ) -> ReconstructionCylinder:
        pass

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    def box_aabb(
        self,
    ) -> ReconstructionBox:
        pass

    def box_aab_inscribed(
        self,
    ) -> ReconstructionBox:
        pass

    def john_ellipsoid(
        self,
    ) -> np.ndarray:
        pass

    def ellipsoid_box_model_matrix(
        self,
        ellipsoid: np.ndarray,
    ) -> np.ndarray:
        pass

    def box_obb(
        self,
    ) -> ReconstructionBox:
        pass

    def box_obb_inscribed(
        self,
    ) -> ReconstructionBox:
        pass

    # ------------------------------------------------------------------
    # High-level interface
    # ------------------------------------------------------------------

    def estimate(
        self,
    ) -> ReconstructionVolumes:
        pass
```

# Future Extensions

The half-space representation naturally supports additional reconstruction primitives without changing the architecture.

Potential future additions include

- minimum enclosing sphere
- maximum inscribed sphere
- minimum enclosing cylinder
- John-Loewner ellipsoids
- convex hull export
- mesh generation
- support function evaluation
- collision queries
- clipping against arbitrary regions of interest

The half-space representation remains the single source of truth from which all geometric primitives are derived.