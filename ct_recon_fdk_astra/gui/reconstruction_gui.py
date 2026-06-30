import os
import sys
import json
import numpy as np

# -----------------------------------------------------------------------------
# Monkey-patch svg_source_detector to evaluate draw_on_detector locally
# -----------------------------------------------------------------------------
import ProjectiveGeometry23.svg_utils
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
import ProjectiveGeometry23.utils as pgu
from svg_snip.Composer import Group, Composer
import svg_snip.Elements3D as e3d

def fix_8digit_hex_svg(svg_code):
    import re
    # Match any stroke or fill color with an 8-digit hex code
    pattern = r'(fill|stroke)="#([0-9a-fA-F]{6})([0-9a-fA-F]{2})"'
    def repl(match):
        attr = match.group(1)
        color = match.group(2)
        alpha_hex = match.group(3)
        alpha_val = int(alpha_hex, 16) / 255.0
        return f'{attr}="#{color}" {attr}-opacity="{alpha_val:.3f}"'
    return re.sub(pattern, repl, svg_code)

def safe_point(P, X, **kwargs):
    x = P @ pgu.cvec(X)
    if abs(x[2][0]) <= 1e-5:
        return ""
    return e3d.point(P, X, **kwargs)

def safe_line(P, X1, X2, **kwargs):
    x1 = P @ pgu.cvec(X1)
    x2 = P @ pgu.cvec(X2)
    if abs(x1[2][0]) <= 1e-5 or abs(x2[2][0]) <= 1e-5:
        return ""
    return e3d.line(P, X1, X2, **kwargs)

def safe_arrow(P, X1, X2, **kwargs):
    x1 = P @ pgu.cvec(X1)
    x2 = P @ pgu.cvec(X2)
    if abs(x1[2][0]) <= 1e-5 or abs(x2[2][0]) <= 1e-5:
        return ""
    return e3d.arrow(P, X1, X2, **kwargs)

def safe_text(P, X, **kwargs):
    x = P @ pgu.cvec(X)
    if abs(x[2][0]) <= 1e-5:
        return ""
    return e3d.text(P, X, **kwargs)

def draw_ellipsoid(P, Q, **kwargs):
    """
    Project 3D ellipsoid Q (4x4 symmetric matrix) onto 2D image plane via projection matrix P (3x4)
    and draw it as a 2D conic using e2d.conic.
    """
    try:
        Q_inv = np.linalg.inv(Q)
        C_inv = P @ Q_inv @ P.T
        if abs(np.linalg.det(C_inv)) < 1e-12:
            return ""
        C = np.linalg.inv(C_inv)
        import svg_snip.Elements as e2d
        return e2d.conic(C, **kwargs)
    except Exception as e:
        return f"<!-- draw_ellipsoid failed: {e} -->"

def draw_quadric_plane_intersection(P, Q, H_plane, **kwargs):
    """
    Render the intersection of 3D quadric Q and a plane defined by basis matrix H_plane
    under the camera projection matrix P (3x4).
    """
    try:
        C_plane = H_plane.T @ Q @ H_plane
        P_eff = P @ H_plane
        C_plane_inv = np.linalg.inv(C_plane)
        C_screen_inv = P_eff @ C_plane_inv @ P_eff.T
        if abs(np.linalg.det(C_screen_inv)) < 1e-12:
            return ""
        C_screen = np.linalg.inv(C_screen_inv)
        import svg_snip.Elements as e2d
        return e2d.conic(C_screen, **kwargs)
    except Exception as e:
        return f"<!-- draw_quadric_plane_intersection failed: {e} -->"

def draw_translated_ellipse(P, C_plane, H_translated, **kwargs):
    """
    Project 3D ellipse C_plane on plane H_translated to screen coordinate plane conic under camera projection matrix P.
    """
    try:
        P_eff = P @ H_translated
        C_plane_inv = np.linalg.inv(C_plane)
        C_screen_inv = P_eff @ C_plane_inv @ P_eff.T
        if abs(np.linalg.det(C_screen_inv)) < 1e-12:
            return ""
        C_screen = np.linalg.inv(C_screen_inv)
        import svg_snip.Elements as e2d
        return e2d.conic(C_screen, **kwargs)
    except Exception as e:
        return f"<!-- draw_translated_ellipse failed: {e} -->"

def draw_cylinder(P, point0, point1, radius, **kwargs):
    """
    Project a 3D cylinder (defined by point0, point1, radius) to 2D under projection P.
    Draws the two projected circular lids as conics and the two silhouette lines.
    """
    try:
        import svg_snip.Elements as e2d
        import ProjectiveGeometry23.utils as pgu
        
        v_dir = point1 - point0
        len_v = np.linalg.norm(v_dir)
        if len_v < 1e-12:
            return ""
        axis = v_dir / len_v
        
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u1 = np.cross(tmp, axis)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(axis, u1)
        u2 /= np.linalg.norm(u2)
        
        C_local_inv = np.diag([1.0, 1.0, -1.0 / (radius**2)])
        
        lids_svg = []
        p_mid = 0.5 * (point0 + point1)
        for p_center in [point0, point1, p_mid]:
            H = np.zeros((4, 3))
            H[:3, 0] = u1
            H[:3, 1] = u2
            H[:3, 2] = p_center
            H[3, 2] = 1.0
            
            P_eff = P @ H
            C_screen_inv = P_eff @ C_local_inv @ P_eff.T
            if abs(np.linalg.det(C_screen_inv)) > 1e-12:
                C_screen = np.linalg.inv(C_screen_inv)
                lids_svg.append(e2d.conic(C_screen, **kwargs))
                
        from recon_coverage import _camera_center
        C_cam = _camera_center(P)
        
        V = C_cam - point0
        A = np.dot(u1, V)
        B = np.dot(u2, V)
        r_val = np.sqrt(A**2 + B**2)
        
        lines_svg = []
        if r_val >= radius:
            phi = np.arctan2(B, A)
            alpha = np.arccos(radius / r_val)
            for theta in [phi + alpha, phi - alpha]:
                dx = radius * np.cos(theta) * u1 + radius * np.sin(theta) * u2
                X0_hom = np.append(point0 + dx, 1.0)
                X1_hom = np.append(point1 + dx, 1.0)
                
                line_str = safe_line(P, X0_hom, X1_hom, **kwargs)
                if line_str:
                    lines_svg.append(line_str)
                    
        return "".join(lids_svg) + "".join(lines_svg)
    except Exception as e:
        return f"<!-- draw_cylinder failed: {e} -->"

# Register safe_arrow in Composer to inject the marker definitions (arrowheads)
import svg_snip.Elements as _elements
Composer.declared_shapes[safe_arrow] = Composer.declared_shapes[_elements.arrow]

def patched_svg_source_detector(P, projection: ProjectionMatrix, draw_on_detector=None, composer=None, **kwargs):
    sdg = SourceDetectorGeometry(projection)
    C = pgu.cvec(sdg.source_position)
    O = sdg.detector_origin
    U = pgu.cvec(sdg.axis_direction_Upx) * projection.image_size[0]
    V = pgu.cvec(sdg.axis_direction_Vpx) * projection.image_size[1]

    group = Group("Source Detector Geometry")

    # Handle projection tracking on detector if geometry function is passed
    if draw_on_detector is not None:
        T_detector = sdg.central_projection_3d
        group.add(draw_on_detector, P=P@T_detector, **kwargs)

    # Check if the source position C is in the null space of the viewport camera P
    x_c = P @ C
    show_source_and_frustum = abs(x_c[2][0]) > 1e-5

    if show_source_and_frustum:
        # Source position
        group.add(e3d.point, P=P, X=C, r=1, fill="black")

    # Detector frame plane and directional axes
    group.add(e3d.polygon, P=P, Xs=[O, O+U, O+V+U, O+V], fill="#00000020", stroke="black", stroke_back="black")
    if kwargs.get('show_axis_labels', True):
        group.add(e3d.arrow, P=P, X1=O, X2=O + U, stroke="magenta")
        group.add(e3d.arrow, P=P, X1=O, X2=O + V, stroke="cyan")
        group.add(e3d.text, P=P, X=O + U * 1.05, content="U", fill="magenta", font_size="12px", font_family="sans-serif")
        group.add(e3d.text, P=P, X=O + V * 1.05, content="V", fill="cyan", font_size="12px", font_family="sans-serif")
        det_label = "virtual detector" if kwargs.get('is_virtual', False) else "detector"
        group.add(e3d.text, P=P, X=O, content=det_label, fill="black", font_size="12px", font_family="sans-serif")

    if show_source_and_frustum:
        # Projection Frustum boundary lines
        group.add(e3d.line, P=P, X1=C, X2=O, stroke="black", stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+V, stroke="black", stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+U, stroke="black", stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+V+U, stroke="black", stroke_width=1.5)

    # Optional descriptive annotations
    if kwargs.get('show_axis_labels', True):
        if show_source_and_frustum and 'label_source' in kwargs:
            group.add(e3d.text, P=P, X=C, content=kwargs['label_source'])
        if 'label_detector' in kwargs:
            group.add(e3d.text, P=P, X=O, content=kwargs['label_detector'])
        
    # Evaluate group immediately to prevent downstream matrix override and fix alpha locally
    svg_code, used_funcs = group(composer=composer, **kwargs)
    return fix_8digit_hex_svg(svg_code), used_funcs

def patched_volume(P, shape, model_matrix=np.eye(4), color_axes=True, lighting=True, composer=None, **kwargs):
    # Call the original volume function without axes
    group = e3d.volume(P, shape=shape, model_matrix=model_matrix, color_axes=False, lighting=lighting, **kwargs)
    
    if color_axes:
        # Draw coordinate axes with stroke_width=2
        X, Y, Z = shape[2], shape[1], shape[0]
        corners_voxel = [
            np.array([0, 0, 0, 1]),
            np.array([X, 0, 0, 1]),
            np.array([0, Y, 0, 1]),
            np.array([0, 0, Z, 1]),
        ]
        corners = [model_matrix @ c for c in corners_voxel]
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[1], stroke='red', stroke_width=2)
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[2], stroke='green', stroke_width=2)
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[3], stroke='blue', stroke_width=2)
        text_kwargs = kwargs.copy()
        text_kwargs.pop("stroke", None)
        text_kwargs.pop("stroke_width", None)
        text_kwargs.pop("fill", None)
        group.add(e3d.text, P=P, X=corners[0], content="volume", fill="black", stroke="none", stroke_width=0, font_size="12px", font_family="sans-serif", **text_kwargs)

    # Evaluate immediately to prevent downstream matrix override and fix alpha locally.
    # Strip style kwargs from the group wrapper to prevent style inheritance on children.
    group_kwargs = {k: v for k, v in kwargs.items() if k not in ["fill", "stroke", "stroke_width"]}
    svg_code, used_funcs = group(composer=composer, **group_kwargs)
    return fix_8digit_hex_svg(svg_code), used_funcs

def patched_trajectory(P, disp_src, active_idx, num_views, composer=None, **kwargs):
    group = Group("Source Trajectory")
    
    part1_idxs = np.arange(0, active_idx)
    part2_idxs = np.arange(active_idx + 1, num_views)
    
    def draw_part_trajectory(g, idxs):
        if len(idxs) < 2:
            return
        if len(idxs) > 90:
            sampled_idxs = idxs[np.round(np.linspace(0, len(idxs) - 1, 90)).astype(int)]
        else:
            sampled_idxs = idxs
        for i in range(len(sampled_idxs) - 1):
            g.add(safe_line, X1=disp_src[sampled_idxs[i]], X2=disp_src[sampled_idxs[i + 1]], stroke="#00adb5", stroke_width=1.2)
            
    draw_part_trajectory(group, part1_idxs)
    draw_part_trajectory(group, part2_idxs)
    
    svg_code, used_funcs = group(composer=composer, P=P)
    return fix_8digit_hex_svg(svg_code), used_funcs
# Apply the patches to the library at runtime
ProjectiveGeometry23.svg_utils.svg_source_detector = patched_svg_source_detector
# -----------------------------------------------------------------------------

# Dynamic format discovery
try:
    import sys
    import os
    _dir = os.path.dirname(os.path.abspath(__file__))
    _reconstruct_dir = os.path.dirname(_dir)
    if _reconstruct_dir not in sys.path:
        sys.path.insert(0, _reconstruct_dir)
    from fileformats import discover_formats
except ImportError:
    try:
        from reconstruct.fileformats import discover_formats
    except ImportError:
        def discover_formats():
            return {}, {}

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTextEdit, QPushButton, QLabel, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QStatusBar, QFileDialog, QGroupBox, QScrollArea, QFrame,
    QMessageBox, QComboBox, QLineEdit, QMenu, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QDialog, QDialogButtonBox, QProgressDialog, QGridLayout
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread, QTimer
from PyQt6.QtGui import QFont, QMouseEvent, QWheelEvent, QAction

# Optional xray-epipolar-consistency (parameterization support)
try:
    import xray_epipolar_consistency.parameterization as _ecc_param
    HAS_ECC = True
except ImportError:
    HAS_ECC = False

# Keep alive references for dynamically launched GUIs
_calib_gui_keepalive = None

# Import ProjectiveGeometry23 components
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
from ProjectiveGeometry23.homography import rotation_x, rotation_z, scale
from ProjectiveGeometry23.svg_utils import svg_coordinate_frame, svg_source_detector
import svg_snip.Elements as e2d
import svg_snip.Elements3D as e3d
import re

def load_ompl(path):
    """
    Loads projection matrices from a text file with one matrix per line.
    Supports parsing metadata comments starting with '#>' or '#'.
    """
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
                
            # Parse matrix row: e.g., [1 0 0 0; 0 1 0 0; 0 0 1 0] or just values
            clean = line_str.replace('[', '').replace(']', '').replace(';', ' ').strip()
            vals = np.array([float(x) for x in clean.split()])
            Ps.append(vals.reshape(3, 4))
   
    return [
        ProjectionMatrix(
            P,
            image_size=[int(x) for x in meta["detector_size_px"].split()],
            pixel_spacing=float(meta["spacing"])
        )
        for P in Ps
    ]


def resolve_relative(config_path, relative_path):
    if not relative_path:
        return ""
    if os.path.isabs(relative_path):
        return relative_path
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
    else:
        config_dir = os.getcwd()
    return os.path.normpath(os.path.join(config_dir, relative_path))

def format_config_json(config):
    import math
    
    def format_sig_figs(val, sig_figs=4):
        if not isinstance(val, (int, float)):
            return val
        if val == 0:
            return 0.0
        try:
            mag = math.floor(math.log10(abs(val)))
            scale = 10 ** (sig_figs - 1 - mag)
            rounded = round(val * scale) / scale
            return float(f"{rounded:.12g}")
        except:
            return val

    # Deep copy config dict/list logic
    config_copy = {}
    for k, v in config.items():
        config_copy[k] = v

    # Format model_matrix
    model_matrix_str = None
    if "model_matrix" in config_copy:
        M = config_copy["model_matrix"]
        try:
            M_formatted = [[format_sig_figs(val, 4) for val in row] for row in M]
            matrix_lines = []
            for row in M_formatted:
                row_str = ", ".join(f"{val}" for val in row)
                matrix_lines.append(f"        [{row_str}]")
            model_matrix_str = "[\n" + ",\n".join(matrix_lines) + "\n    ]"
            config_copy["model_matrix"] = "__MODEL_MATRIX_PLACEHOLDER__"
        except:
            pass

    # Format image_transform
    image_transform_str = None
    if "image_transform" in config_copy:
        M = config_copy["image_transform"]
        try:
            M_formatted = [[format_sig_figs(val, 4) for val in row] for row in M]
            matrix_lines = []
            for row in M_formatted:
                row_str = ", ".join(f"{val}" for val in row)
                matrix_lines.append(f"        [{row_str}]")
            image_transform_str = "[\n" + ",\n".join(matrix_lines) + "\n    ]"
            config_copy["image_transform"] = "__IMAGE_TRANSFORM_PLACEHOLDER__"
        except:
            pass

    # Format world_transform
    world_transform_str = None
    if "world_transform" in config_copy:
        M = config_copy["world_transform"]
        try:
            M_formatted = [[format_sig_figs(val, 4) for val in row] for row in M]
            matrix_lines = []
            for row in M_formatted:
                row_str = ", ".join(f"{val}" for val in row)
                matrix_lines.append(f"        [{row_str}]")
            world_transform_str = "[\n" + ",\n".join(matrix_lines) + "\n    ]"
            config_copy["world_transform"] = "__WORLD_TRANSFORM_PLACEHOLDER__"
        except:
            pass

    # Format voxel_dimensions
    voxel_dims_str = None
    if "voxel_dimensions" in config_copy:
        dims = config_copy["voxel_dimensions"]
        try:
            voxel_dims_str = f"[{int(dims[0])}, {int(dims[1])}, {int(dims[2])}]"
            config_copy["voxel_dimensions"] = "__VOXEL_DIMENSIONS_PLACEHOLDER__"
        except:
            pass

    # Serialize to JSON with standard indent
    json_str = json.dumps(config_copy, indent=4)

    # Restore placeholders
    if model_matrix_str is not None:
        json_str = json_str.replace('"__MODEL_MATRIX_PLACEHOLDER__"', model_matrix_str)
    if image_transform_str is not None:
        json_str = json_str.replace('"__IMAGE_TRANSFORM_PLACEHOLDER__"', image_transform_str)
    if world_transform_str is not None:
        json_str = json_str.replace('"__WORLD_TRANSFORM_PLACEHOLDER__"', world_transform_str)
    if voxel_dims_str is not None:
        json_str = json_str.replace('"__VOXEL_DIMENSIONS_PLACEHOLDER__"', voxel_dims_str)

    return json_str

def downscale_image(img, factor):
    if factor <= 1.0:
        return img
    H, W = img.shape
    new_H = max(1, int(round(H / factor)))
    new_W = max(1, int(round(W / factor)))
    
    import scipy.ndimage
    zoom_y = new_H / H
    zoom_x = new_W / W
    
    downscaled = scipy.ndimage.zoom(img, (zoom_y, zoom_x), order=1)
    
    if np.issubdtype(img.dtype, np.integer):
        return np.round(downscaled).astype(img.dtype)
    else:
        return downscaled.astype(img.dtype)

class ExportSubsampledDialog(QDialog):
    def __init__(self, parent=None, image_paths=None):
        super().__init__(parent)
        self.image_paths = image_paths or []
        self.setWindowTitle("Export Sub-sampled Dataset")
        self.resize(450, 210)
        
        layout = QVBoxLayout(self)
        
        # Grid layout for inputs
        grid = QGridLayout()
        layout.addLayout(grid)
        
        # 1. Skip factor
        grid.addWidget(QLabel("Skip every n-th image:"), 0, 0)
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(1, 1000)
        self.spin_skip.setValue(2)
        grid.addWidget(self.spin_skip, 0, 1)
        
        # 2. Downscale factor
        grid.addWidget(QLabel("Image downscale factor:"), 1, 0)
        self.spin_factor = QDoubleSpinBox()
        self.spin_factor.setRange(1.0, 100.0)
        self.spin_factor.setDecimals(2)
        self.spin_factor.setSingleStep(0.1)
        self.spin_factor.setValue(2.0)
        grid.addWidget(self.spin_factor, 1, 1)
        
        # 3. Export directory
        grid.addWidget(QLabel("Export Directory:"), 2, 0)
        dir_layout = QHBoxLayout()
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText("Select empty or new directory...")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.edit_dir)
        dir_layout.addWidget(self.btn_browse)
        grid.addLayout(dir_layout, 2, 1)

        # 4. Estimated Scan Size
        grid.addWidget(QLabel("Estimated scan size:"), 3, 0)
        self.lbl_scan_size = QLabel("Estimating...")
        grid.addWidget(self.lbl_scan_size, 3, 1)
        
        # Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Detect bytes per pixel and datatype
        self.bytes_per_pixel = 4 # default to float32
        self.dtype_str = "float32"
        self.detect_image_dtype()

        # Connect signals
        self.spin_skip.valueChanged.connect(self.update_size_estimate)
        self.spin_factor.valueChanged.connect(self.update_size_estimate)
        self.update_size_estimate()
        
    def detect_image_dtype(self):
        if not self.image_paths:
            return
        try:
            import numpy as np
            from PIL import Image
            import nrrd
            first_path = self.image_paths[0]
            if not os.path.exists(first_path):
                return
            if first_path.lower().endswith('.nrrd'):
                img, _ = nrrd.read(first_path)
                self.bytes_per_pixel = img.dtype.itemsize
                self.dtype_str = str(img.dtype)
            else:
                with Image.open(first_path) as pil_img:
                    img = np.array(pil_img)
                    self.bytes_per_pixel = img.dtype.itemsize
                    self.dtype_str = str(img.dtype)
        except Exception as e:
            print(f"[DEBUG] Failed to detect dtype of first image: {e}")

    def update_size_estimate(self):
        skip = self.spin_skip.value()
        factor = self.spin_factor.value()
        
        p_list = []
        if self.parent() and hasattr(self.parent(), 'P_list') and self.parent().P_list:
            p_list = self.parent().P_list
            
        if not p_list:
            self.lbl_scan_size.setText("No trajectory loaded.")
            return
            
        indices = list(range(0, len(p_list), skip))
        total_bytes = 0
        w_out, h_out = 0, 0
        
        for idx in indices:
            p = p_list[idx]
            w = max(1, int(round(p.image_size[0] / factor)))
            h = max(1, int(round(p.image_size[1] / factor)))
            total_bytes += w * h * self.bytes_per_pixel
            if idx == 0:
                w_out, h_out = w, h
                
        size_gb = total_bytes / (1024 ** 3)
        num_images = len(indices)
        
        self.lbl_scan_size.setText(
            f"{size_gb:.3f} GB ({num_images} images, {w_out} x {h_out} px, {self.dtype_str})"
        )
        
    def browse_directory(self):
        dialog = QFileDialog(self, "Select Export Directory")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                self.edit_dir.setText(selected[0])
            
    def validate_and_accept(self):
        path = self.edit_dir.text().strip()
        if not path:
            QMessageBox.warning(self, "Validation Error", "Please select an export directory.")
            return
            
        if os.path.exists(path):
            if os.listdir(path):
                QMessageBox.warning(self, "Validation Error", "The selected export directory must be empty.")
                return
        
        self.accept()

# QSS Premium Dark Mode Theme Stylesheet - Removed for Vanilla look
DARK_THEME = ""

# -----------------------------------------------------------------------------
# Reconstruction Thread & Console Window for real-time subprocess-free execution
# -----------------------------------------------------------------------------
# ReconstructionThread and ConsoleOutputWindow removed. ProcessConsoleWindow is used instead.


# ---------------------------------------------------------------------------
# Floating Parameterization Window (requires xray-epipolar-consistency)
# ---------------------------------------------------------------------------
class ParameterizationWindow(QWidget):
    """Floating window for editing a single Parameterization object.
    Two-way sync between raw JSON editor and parameter table.
    Only instantiated when HAS_ECC is True."""

    paramsChanged = pyqtSignal()

    # Non-Chain, non-abstract classes available for selection
    _STATIONARY = ['DetectorShift', 'DetectorOrientation', 'Distance',
                   'GantryAngle', 'RotationAxis', 'ObjectPose', 'Turntable']
    _TIMEVARIANT = ['LinearDrift', 'ContinuousMotion']  # Jitter excluded: per-view params can't be applied to a single matrix

    def __init__(self, parent_app):
        super().__init__()
        self.app = parent_app
        self._obj = None               # current ParameterizationBase instance
        self._initialized_obj = None   # deepcopy with prior_knowledge set from full P_list
        self._sampled_P_views = None   # list of N apply_to_trajectory results for overlay
        self._updating = False         # re-entrancy guard for two-way sync

        self.setWindowTitle('Parameterization')
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(900, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ---- Class selector row ------------------------------------------
        sel_row = QHBoxLayout()
        sel_row.setSpacing(6)

        sel_row.addWidget(QLabel('Class:'))
        self._cls_combo = QComboBox()
        self._cls_combo.addItems(self._STATIONARY + self._TIMEVARIANT)
        self._cls_combo.currentTextChanged.connect(self._on_class_changed)
        sel_row.addWidget(self._cls_combo)

        self._lbl_child = QLabel('Child:')
        sel_row.addWidget(self._lbl_child)
        self._child_combo = QComboBox()
        self._child_combo.addItems(self._STATIONARY)
        sel_row.addWidget(self._child_combo)

        self._lbl_ncp = QLabel('Ctrl pts:')
        sel_row.addWidget(self._lbl_ncp)
        self._ncp_spin = QSpinBox()
        self._ncp_spin.setRange(1, 999)
        self._ncp_spin.setValue(4)
        self._ncp_spin.setFixedWidth(60)
        sel_row.addWidget(self._ncp_spin)

        btn_create = QPushButton('Create')
        btn_create.clicked.connect(self._create_object)
        sel_row.addWidget(btn_create)
        sel_row.addStretch()
        root.addLayout(sel_row)

        # ---- Splitter: JSON editor | parameter table ----------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Left: JSON editor
        json_panel = QWidget()
        json_layout = QVBoxLayout(json_panel)
        json_layout.setContentsMargins(0, 0, 0, 0)
        json_layout.setSpacing(2)
        json_layout.addWidget(QLabel('JSON'))
        self._json_edit = QTextEdit()
        self._json_edit.setFont(QFont('Monospace', 9))
        self._json_edit.setPlaceholderText('Create an object above to see its JSON…')
        self._json_edit.textChanged.connect(self._on_json_changed)
        json_layout.addWidget(self._json_edit)
        splitter.addWidget(json_panel)

        # Right: parameter table
        tbl_panel = QWidget()
        tbl_layout = QVBoxLayout(tbl_panel)
        tbl_layout.setContentsMargins(0, 0, 0, 0)
        tbl_layout.setSpacing(2)
        tbl_layout.addWidget(QLabel('Parameters'))
        # Columns: Active | Name | Value | Min | Max | Rnd
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ['Active', 'Parameter', 'Value', 'Min', 'Max', ''])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        _col_widths = [46, 150, 95, 90, 90, 28]
        for col, w in enumerate(_col_widths):
            self._table.setColumnWidth(col, w)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.verticalHeader().hide()
        # Cap the right panel so it never grows wider than the table columns +
        # a small margin for the vertical scrollbar; the JSON editor gets the rest.
        _tbl_max_w = sum(_col_widths) + 22
        tbl_panel.setMaximumWidth(_tbl_max_w)
        tbl_layout.addWidget(self._table)
        splitter.addWidget(tbl_panel)
        splitter.setSizes([9999, _tbl_max_w])  # JSON side takes all available space

        # ---- Bottom button bar -------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_load = QPushButton('Load JSON…')
        btn_load.clicked.connect(self._load_json)
        btn_row.addWidget(btn_load)

        btn_save = QPushButton('Save JSON…')
        btn_save.clicked.connect(self._save_json)
        btn_row.addWidget(btn_save)

        btn_row.addStretch()

        btn_rnd_all = QPushButton('Randomize All')
        btn_rnd_all.clicked.connect(self._randomize_all)
        btn_row.addWidget(btn_rnd_all)

        btn_row.addWidget(QLabel('Random Samples for Visualization:'))
        self._spin_nsamples = QSpinBox()
        self._spin_nsamples.setRange(1, 200)
        self._spin_nsamples.setValue(25)
        self._spin_nsamples.setFixedWidth(55)
        self._spin_nsamples.valueChanged.connect(self._rebuild_preview_cache)
        btn_row.addWidget(self._spin_nsamples)

        self._chk_preview = QCheckBox('Preview')
        self._chk_preview.toggled.connect(self._on_preview_toggled)
        btn_row.addWidget(self._chk_preview)

        btn_apply = QPushButton('Apply')
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self._apply)
        btn_row.addWidget(btn_apply)

        root.addLayout(btn_row)

        # Debounce timer for JSON→table sync
        self._json_timer = QTimer()
        self._json_timer.setSingleShot(True)
        self._json_timer.setInterval(500)
        self._json_timer.timeout.connect(self._sync_json_to_table)

        # Rebuild preview cache (and re-apply if Preview is on) when params change
        self.paramsChanged.connect(self._rebuild_preview_cache)

        # Active checkbox uses itemChanged — connect after building table
        self._table.itemChanged.connect(self._on_table_edited)

        # Init visibility
        self._on_class_changed(self._cls_combo.currentText())

    # ------------------------------------------------------------------
    # Class selector helpers
    # ------------------------------------------------------------------
    def _on_class_changed(self, cls_name):
        is_tv = cls_name in self._TIMEVARIANT
        self._lbl_child.setVisible(is_tv)
        self._child_combo.setVisible(is_tv)
        self._lbl_ncp.setVisible(is_tv)
        self._ncp_spin.setVisible(is_tv)

    def _get_cls(self, name):
        return getattr(_ecc_param, name)

    # ------------------------------------------------------------------
    # Create / populate
    # ------------------------------------------------------------------
    def _create_object(self):
        import xray_epipolar_consistency.parameterization as ep
        cls_name = self._cls_combo.currentText()
        try:
            if cls_name in self._TIMEVARIANT:
                child_cls = self._get_cls(self._child_combo.currentText())
                n_cp = self._ncp_spin.value()
                cls = self._get_cls(cls_name)
                self._obj = cls(referenced_class=child_cls, num_control_points=n_cp)
            else:
                self._obj = self._get_cls(cls_name)()
        except Exception as e:
            QMessageBox.critical(self, 'Create Error', str(e))
            return
        self._populate_table()
        self._sync_table_to_json()

    def _populate_table(self):
        if self._obj is None:
            return
        self._updating = True
        self._table.setRowCount(0)
        for name, info in self._obj.parameters.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setRowHeight(row, 26)

            # Col 0: Active checkbox
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(
                Qt.CheckState.Checked if info.get('opt', True)
                else Qt.CheckState.Unchecked)
            self._table.setItem(row, 0, chk)

            # Col 1: Name (read-only, tooltip = description)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setToolTip(info.get('description', ''))
            self._table.setItem(row, 1, name_item)

            lo, hi = info.get('range', (-1.0, 1.0))
            val = float(info.get('value', 0.0))

            # Col 2: Value — QLineEdit for scientific notation
            val_edit = QLineEdit(f'{val:.6g}')
            val_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            val_edit.editingFinished.connect(self._on_table_edited)
            self._table.setCellWidget(row, 2, val_edit)

            # Col 3: Min spinbox
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1e12, 1e12)
            min_spin.setDecimals(6)
            min_spin.setValue(lo)
            min_spin.setSingleStep(abs(hi - lo) / 100 if hi != lo else 0.1)
            min_spin.valueChanged.connect(self._on_table_edited)
            self._table.setCellWidget(row, 3, min_spin)

            # Col 4: Max spinbox
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1e12, 1e12)
            max_spin.setDecimals(6)
            max_spin.setValue(hi)
            max_spin.setSingleStep(abs(hi - lo) / 100 if hi != lo else 0.1)
            max_spin.valueChanged.connect(self._on_table_edited)
            self._table.setCellWidget(row, 4, max_spin)

            # Col 5: per-row Randomize
            btn_rnd = QPushButton('\u27f3')
            btn_rnd.setFixedWidth(26)
            btn_rnd.clicked.connect(lambda _, r=row: self._randomize_row(r))
            self._table.setCellWidget(row, 5, btn_rnd)

        self._updating = False

    # ------------------------------------------------------------------
    # Two-way sync
    # ------------------------------------------------------------------
    def _read_table_into_obj(self):
        """Read current table widget values back into self._obj.parameters."""
        if self._obj is None:
            return
        for row, name in enumerate(self._obj.parameters.keys()):
            chk = self._table.item(row, 0)
            val_w = self._table.cellWidget(row, 2)
            min_w = self._table.cellWidget(row, 3)
            max_w = self._table.cellWidget(row, 4)
            if chk:
                self._obj.parameters[name]['opt'] = (
                    chk.checkState() == Qt.CheckState.Checked)
            if val_w:
                try:
                    self._obj.parameters[name]['value'] = float(val_w.text())
                except ValueError:
                    pass
            if min_w and max_w:
                self._obj.parameters[name]['range'] = (
                    min_w.value(), max_w.value())

    def _sync_table_to_json(self):
        if self._updating or self._obj is None:
            return
        self._read_table_into_obj()
        self._updating = True
        self._json_edit.blockSignals(True)
        self._json_edit.setPlainText(
            json.dumps(self._obj.to_dict(), indent=2))
        self._json_edit.blockSignals(False)
        self._json_edit.setStyleSheet('')
        self._updating = False
        self.paramsChanged.emit()

    def _on_table_edited(self):
        if not self._updating:
            self._sync_table_to_json()

    def _on_json_changed(self):
        if not self._updating:
            self._json_timer.start()

    def _sync_json_to_table(self):
        if self._updating:
            return
        text = self._json_edit.toPlainText().strip()
        if not text:
            return
        try:
            d = json.loads(text)
            import xray_epipolar_consistency.parameterization as ep
            obj = ep.from_dict(d)
            self._obj = obj
            self._updating = True
            self._populate_table()
            self._updating = False
            self._json_edit.setStyleSheet('')
            self.paramsChanged.emit()
        except Exception:
            self._json_edit.setStyleSheet(
                'QTextEdit { border: 2px solid red; }')

    # ------------------------------------------------------------------
    # Randomize
    # ------------------------------------------------------------------
    def _randomize_row(self, row, *, _batch=False):
        """Set one row to a random value within its range.
        Pass _batch=True to suppress the JSON sync (caller will sync once)."""
        import random
        min_w = self._table.cellWidget(row, 3)
        max_w = self._table.cellWidget(row, 4)
        val_w = self._table.cellWidget(row, 2)
        if min_w and max_w and val_w:
            lo, hi = min_w.value(), max_w.value()
            val_w.setText(f'{random.uniform(lo, hi):.6g}')
            if not _batch:
                self._sync_table_to_json()

    def _randomize_all(self):
        if self._obj is None:
            return
        self._updating = True   # suppress per-row paramsChanged
        for row in range(self._table.rowCount()):
            chk = self._table.item(row, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                self._randomize_row(row, _batch=True)
        self._updating = False
        self._sync_table_to_json()  # single update for all rows

    # ------------------------------------------------------------------
    # Load / Save JSON
    # ------------------------------------------------------------------
    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Parameterization JSON', '',
            'JSON Files (*.json);;All Files (*)')
        if not path:
            return
        try:
            with open(path) as f:
                text = f.read()
            self._json_edit.setPlainText(text)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Parameterization JSON', '',
            'JSON Files (*.json);;All Files (*)')
        if not path:
            return
        try:
            with open(path, 'w') as f:
                f.write(self._json_edit.toPlainText())
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', str(e))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _rebuild_preview_cache(self):
        """Called on every paramsChanged. Initialises the obj with the full
        P_list once (sets prior_knowledge), pre-computes the preview P_list,
        and generates N random samples for the overlay.
        Heavy work happens here once; toggling Preview is instant."""
        import copy, random
        if self._obj is None or not self.app.P_list:
            return
        self._read_table_into_obj()

        # Initialise prior_knowledge from the full trajectory
        obj_init = copy.deepcopy(self._obj)
        obj_init.estimateTrajectoryParameters(self.app.P_list)
        self._initialized_obj = obj_init

        # Always pre-compute the preview P_list (current param values)
        self._preview_P_list_cache = obj_init.apply_to_trajectory(self.app.P_list)
        if self._chk_preview.isChecked():
            self.app.P_list_preview = self._preview_P_list_cache
            self.app.update_trajectory_coordinates()

        # Generate N randomised samples (active params only)
        # Values are written directly into parameters dict;
        # apply_stationary reads from there — no set_parameter_vector needed.
        n = self._spin_nsamples.value()
        self._sampled_P_views = []
        for _ in range(n):
            obj_s = copy.deepcopy(obj_init)  # prior_knowledge already set
            for name, info in obj_s.parameters.items():
                if info.get('opt', True):
                    lo, hi = info.get('range', (-1.0, 1.0))
                    info['value'] = random.uniform(lo, hi)
            self._sampled_P_views.append(
                obj_s.apply_to_trajectory(self.app.P_list))

        self.app.render_viewport()
        self.app._refresh_projection_windows()

    def _on_preview_toggled(self, checked):
        """Toggle preview instantly from cache — no recomputation."""
        cache = getattr(self, '_preview_P_list_cache', None)
        self.app.P_list_preview = cache if (checked and cache is not None) else None
        self.app.update_trajectory_coordinates()
        self.app.render_viewport()
        self.app._refresh_projection_windows()


    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply(self):
        if self._obj is None or not self.app.P_list:
            return
        self._read_table_into_obj()
        import copy
        if self._initialized_obj is None:
            obj_copy = copy.deepcopy(self._obj)
            obj_copy.estimateTrajectoryParameters(self.app.P_list)
            self._initialized_obj = obj_copy
        # Commit permanently to the original P_list
        self.app.P_list = self._initialized_obj.apply_to_trajectory(self.app.P_list)
        # Clear preview state
        self.app.P_list_preview = None
        self.app.update_trajectory_coordinates()
        self._initialized_obj = None
        self._chk_preview.blockSignals(True)
        self._chk_preview.setChecked(False)
        self._chk_preview.blockSignals(False)
        self.app.render_viewport()
        self.app.statusBar().showMessage('Parameterization applied.')

    # ------------------------------------------------------------------
    # API for ProjectionViewWindow overlay
    # ------------------------------------------------------------------
    def get_active_param_ranges(self):
        """Return list of (name, lo, hi) for each active (checked) parameter."""
        if self._obj is None:
            return []
        result = []
        for row, name in enumerate(self._obj.parameters.keys()):
            chk = self._table.item(row, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                min_w = self._table.cellWidget(row, 3)
                max_w = self._table.cellWidget(row, 4)
                if min_w and max_w:
                    result.append((name, min_w.value(), max_w.value()))
        return result

    def _get_values(self):
        """Return {name: value} for all parameters."""
        if self._obj is None:
            return {}
        self._read_table_into_obj()
        return {n: info['value'] for n, info in self._obj.parameters.items()}


def _apply_parameterization_obj(obj, P_list):
    """Apply a ParameterizationBase instance to a list of ProjectionMatrix objects.
    apply_to_trajectory takes and returns list[ProjectionMatrix]."""
    return obj.apply_to_trajectory(P_list)



# ---------------------------------------------------------------------------
# Projection View Window — shows scene from the current projection's POV
# ---------------------------------------------------------------------------
class ProjectionViewWindow(QWidget):
    """Resizable window showing the scene projected through one projection matrix.
    Multiple independent instances can be open simultaneously, each with its own
    view-index slider. Uses the same SVG Composer pipeline as the main viewport."""

    def __init__(self, parent_app, view_idx=0):
        super().__init__()
        self.app = parent_app
        self._view_idx = view_idx
        self._preview_min_val = None
        self._preview_max_val = None

        self.setWindowTitle(f"Projectoin View \u2014 C{view_idx}")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(640, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The SVG widget lives inside a plain container so we can letterbox it
        self._container = QWidget()
        self._container.setStyleSheet("background: #888;")
        outer.addWidget(self._container, stretch=1)

        self._svg = QSvgWidget(self._container)
        self._svg.setStyleSheet("background: white;")

        # Own view-index slider + spinbox (no connection to main window)
        ctrl = QWidget()
        ctrl.setFixedHeight(32)
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(6, 2, 6, 2)
        ctrl_layout.setSpacing(6)

        n = len(self.app.P_list)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, n - 1))
        self._slider.setValue(view_idx)

        self._spin = QSpinBox()
        self._spin.setRange(0, max(0, n - 1))
        self._spin.setValue(view_idx)
        self._spin.setFixedWidth(60)

        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._slider.valueChanged.connect(self._on_view_changed)

        ctrl_layout.addWidget(self._slider)
        ctrl_layout.addWidget(self._spin)
        
        self._chk_preview = QCheckBox("Preview Image Data")
        self._chk_preview.setChecked(False)
        
        image_paths = self.app.get_resolved_image_paths()
        has_matching_images = n > 0 and n == len(image_paths)
        self._chk_preview.setEnabled(has_matching_images)
        self._chk_preview.stateChanged.connect(self.refresh)
        ctrl_layout.addWidget(self._chk_preview)
        
        outer.addWidget(ctrl)

        # Defer first render so the container has its real size after layout
        QTimer.singleShot(0, self.refresh)

    def _on_view_changed(self, idx):
        self._view_idx = idx
        self.setWindowTitle(f"Projectoin View \u2014 C{idx}")
        self.refresh()

    def update_slider_range(self):
        """Called by the main app when P_list changes size."""
        n = len(self.app.P_list)
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        self._spin.setRange(0, max(0, n - 1))
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)
        
        self._preview_min_val = None
        self._preview_max_val = None
        
        image_paths = self.app.get_resolved_image_paths()
        has_matching_images = n > 0 and n == len(image_paths)
        self._chk_preview.blockSignals(True)
        self._chk_preview.setEnabled(has_matching_images)
        if not has_matching_images:
            self._chk_preview.setChecked(False)
        self._chk_preview.blockSignals(False)

    def _load_and_process_preview_image(self, path):
        ext = os.path.splitext(path.lower())[1]
        if ext in ('.tif', '.tiff'):
            import tifffile
            arr = np.asarray(tifffile.imread(path), dtype=np.float32)
        elif ext == '.nrrd':
            import nrrd
            arr, _ = nrrd.read(path)
            arr = np.asarray(arr, dtype=np.float32)
        elif ext in ('.png', '.pnj', '.jpg', '.jpeg'):
            from PIL import Image
            img = Image.open(path)
            arr = np.array(img.convert('F'), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported image format: {ext}")
            
        h, w = arr.shape[:2]
        if w > 1024 or h > 1024:
            scale = 1024.0 / max(w, h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            from PIL import Image
            img_temp = Image.fromarray(arr)
            img_resized = img_temp.resize((new_w, new_h), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32)
        return arr

    def refresh(self):
        if not self.app.P_list:
            return
        if self._view_idx >= len(self.app.P_list):
            self._view_idx = 0
        P_proj = self.app.P_list[self._view_idx]
        img_W, img_H = P_proj.image_size

        # Letterbox: fit detector into window maintaining aspect ratio
        win_W, win_H = self._container.width(), self._container.height()
        if win_W < 1 or win_H < 1:
            win_W, win_H = 640, 480
        scale = min(win_W / img_W, win_H / img_H)
        draw_W = int(img_W * scale)
        draw_H = int(img_H * scale)
        off_x = (win_W - draw_W) // 2
        off_y = (win_H - draw_H) // 2
        self._svg.setGeometry(off_x, off_y, draw_W, draw_H)

        preview_img = None
        if self._chk_preview.isChecked():
            image_paths = self.app.get_resolved_image_paths()
            if len(image_paths) != len(self.app.P_list):
                self._chk_preview.blockSignals(True)
                self._chk_preview.setChecked(False)
                self._chk_preview.setEnabled(False)
                self._chk_preview.blockSignals(False)
                QMessageBox.warning(self, "Invalid Image Data", f"Number of image files ({len(image_paths)}) does not match projection matrices ({len(self.app.P_list)}).")
            else:
                try:
                    curr_path = image_paths[self._view_idx]
                    # Verify format of current image
                    ext = os.path.splitext(curr_path.lower())[1]
                    if ext not in ('.tif', '.tiff', '.nrrd', '.png', '.pnj', '.jpg', '.jpeg'):
                        raise ValueError(f"unsupported image format: '{ext}'")
                        
                    if self._preview_min_val is None:
                        ref_idx = 1 if len(image_paths) >= 2 else 0
                        ref_path = image_paths[ref_idx]
                        ref_arr = self._load_and_process_preview_image(ref_path)
                        self._preview_min_val = float(np.min(ref_arr))
                        self._preview_max_val = float(np.max(ref_arr))
                        if self._preview_max_val == self._preview_min_val:
                            self._preview_max_val += 1.0
                            
                    curr_arr = self._load_and_process_preview_image(curr_path)
                    min_v = self._preview_min_val
                    max_v = self._preview_max_val
                    
                    if max_v != min_v:
                        curr_arr = (curr_arr - min_v) / (max_v - min_v)
                    else:
                        curr_arr = curr_arr - min_v
                    curr_arr = np.clip(curr_arr * 255.0, 0, 255).astype(np.uint8)
                    
                    from PIL import Image
                    preview_img = Image.fromarray(curr_arr)
                except Exception as e:
                    self._chk_preview.blockSignals(True)
                    self._chk_preview.setChecked(False)
                    self._chk_preview.setEnabled(False)
                    self._chk_preview.blockSignals(False)
                    QMessageBox.warning(self, "Preview Error", f"Failed to load preview image:\n{e}")

        # Build the scene using the shared helper
        # For the projection window we render at detector resolution
        svg_obj = self.app._build_scene_svg(img_W, img_H, P=P_proj.P, preview_pil_image=preview_img)
        self.app._add_dynamic_elements(
            svg_obj,
            P_view=P_proj.P,
            active_idx=self._view_idx,
            show_pyramid=False,
            show_current_source=False,
        )

        # Add U/V axis arrows at image center
        self._add_uv_axes(svg_obj, img_W, img_H)

        # Render main scene
        raw = svg_obj.render(P=P_proj.P)

        # Inject parameter range overlay as SVG fragments
        if (self.app._param_win is not None
                and self.app._param_win.isVisible()
                and self.app._param_win._sampled_P_views is not None):
            overlay = self._build_param_overlay_svg(img_W, img_H)
            if overlay:
                raw = raw.replace('</svg>', overlay + '\n</svg>', 1)

        fixed = ReconstructionGUIApp._fix_svg_alpha(raw)
        self._svg.load(fixed.encode('utf-8'))

    def _add_uv_axes(self, svg_obj, img_W, img_H):
        """Draw U (red) and V (blue) arrows at the image center.
        Arrow length = 10% of min(img_W, img_H) rounded UP to nearest power of 10."""
        import math
        raw_len = 0.10 * min(img_W, img_H)
        exp = math.ceil(math.log10(max(raw_len, 1.0)))
        axis_len = 10 ** exp  # image-space pixels

        cx, cy = img_W / 2.0, img_H / 2.0

        # U axis — magenta, points right (+u)
        svg_obj.add(
            e2d.arrow,
            x1=cx, y1=cy, x2=cx + axis_len, y2=cy,
            stroke='magenta', stroke_width=2,
        )
        svg_obj.add(e2d.text, x=cx + axis_len + 4, y=cy + 4,
                    content="U", fill='magenta', font_size='12px', font_family='sans-serif')

        # V axis — cyan, points down (+v)
        svg_obj.add(
            e2d.arrow,
            x1=cx, y1=cy, x2=cx, y2=cy + axis_len,
            stroke='cyan', stroke_width=2,
        )
        svg_obj.add(e2d.text, x=cx + 4, y=cy + axis_len + 12,
                    content="V", fill='cyan', font_size='12px', font_family='sans-serif')

    def _build_param_overlay_svg(self, img_W, img_H):
        """Return an SVG fragment showing N randomised parameter samples.
        The N full-trajectory results are cached in _sampled_P_views;
        this method is just a per-view cached lookup — no recomputation."""
        import re
        param_win = self.app._param_win
        sampled = param_win._sampled_P_views
        if not sampled or self.app.voxel_dimensions is None:
            return ''

        Nx, Ny, Nz = self.app.voxel_dimensions
        M = self.app.model_matrix

        # 9 reference points in world-space homogeneous coords
        ref_pts = []
        for ix in (0, Nx):
            for iy in (0, Ny):
                for iz in (0, Nz):
                    ref_pts.append(M @ np.array([ix, iy, iz, 1.0]))
        ref_pts.append(M @ np.array([Nx/2, Ny/2, Nz/2, 1.0]))

        def svg_inner(fragment):
            m = re.search(r'<svg[^>]*>(.*)</svg>', fragment, re.DOTALL)
            return m.group(1).strip() if m else ''

        parts = []
        for sample_traj in sampled:
            P_mod = sample_traj[self._view_idx]
            c = Composer((img_W, img_H))
            for X in ref_pts:
                c.add(e3d.point, X=X, r=2, fill='#555555')
            parts.append(svg_inner(c.render(P=P_mod.P)))

        return '\n'.join(parts)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    def closeEvent(self, event):
        try:
            self.app._projection_windows.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)


class ReconstructionVolumeWindow(QWidget):
    """
    Floating (always-on-top) window for editing the reconstruction volume.
    Owns no data itself — reads/writes through the parent ReconstructionGUIApp.
    """
    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.app = parent_app
        self._updating = False  # re-entrance guard
        self._rot_x = 0.0
        self._rot_y = 0.0
        self._rot_z = 0.0

        self.setWindowTitle("Reconstruction Volume")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(380, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # --- Top grid layout (Position, Number of Voxels) ---
        grid = QGridLayout()
        grid.setSpacing(6)

        # Position (corner)
        grid.addWidget(QLabel("Position (corner)"), 0, 0)
        self.spin_pos_x = QDoubleSpinBox()
        self.spin_pos_y = QDoubleSpinBox()
        self.spin_pos_z = QDoubleSpinBox()
        for i, sp in enumerate((self.spin_pos_x, self.spin_pos_y, self.spin_pos_z)):
            sp.setRange(-9999.0, 9999.0)
            sp.setDecimals(2)
            sp.setSingleStep(1.0)
            sp.setFixedWidth(72)
            grid.addWidget(sp, 0, i + 1)

        # Number of Voxels
        grid.addWidget(QLabel("Number of Voxels"), 1, 0)
        self.spin_vox_x = QSpinBox()
        self.spin_vox_y = QSpinBox()
        self.spin_vox_z = QSpinBox()
        for i, sp in enumerate((self.spin_vox_x, self.spin_vox_y, self.spin_vox_z)):
            sp.setRange(1, 8192)
            sp.setFixedWidth(72)
            grid.addWidget(sp, 1, i + 1)

        layout.addLayout(grid)

        # --- Voxel Size & Rescale ---
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Voxel Size"))
        self.spin_vox_size = QDoubleSpinBox()
        self.spin_vox_size.setRange(0.0001, 1000.0)
        self.spin_vox_size.setDecimals(4)
        self.spin_vox_size.setSingleStep(0.01)
        self.spin_vox_size.setFixedWidth(72)
        size_row.addWidget(self.spin_vox_size)
        
        size_row.addSpacing(15)
        
        size_row.addWidget(QLabel("Rescale:"))
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.01, 100.0)
        self.scale_factor_spin.setValue(0.5)
        self.scale_factor_spin.setSingleStep(0.1)
        self.scale_factor_spin.setFixedWidth(70)
        size_row.addWidget(self.scale_factor_spin)
        
        btn_scale = QPushButton("Rescale")
        btn_scale.clicked.connect(self._scale_action)
        size_row.addWidget(btn_scale)
        size_row.addStretch()
        layout.addLayout(size_row)

        # --- Rotation sliders ---
        self.slider_rot_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot_z = QSlider(Qt.Orientation.Horizontal)
        for lbl_text, sl in (("Rotation X", self.slider_rot_x),
                              ("Rotation Y", self.slider_rot_y),
                              ("Rotation Z", self.slider_rot_z)):
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl_text))
            sl.setRange(-1800, 1800)  # tenths of a degree
            sl.setValue(0)
            row.addWidget(sl)
            layout.addLayout(row)

        # --- Read-only model_matrix JSON ---
        layout.addWidget(QLabel("model_matrix (read-only):"))
        self.txt_matrix = QTextEdit()
        self.txt_matrix.setReadOnly(True)
        self.txt_matrix.setFont(QFont("Consolas", 8))
        self.txt_matrix.setFixedHeight(100)
        layout.addWidget(self.txt_matrix)

        # --- Automatic Suggestions Group Box ---
        group_auto = QGroupBox("Automatic")
        auto_layout = QVBoxLayout(group_auto)
        
        # Center button
        btn_center = QPushButton("Center")
        btn_center.clicked.connect(self.app.center_volume_action)
        auto_layout.addWidget(btn_center)
        
        # Grid layout for suggestion options
        grid_suggest = QGridLayout()
        grid_suggest.setSpacing(4)
        
        # Row 0: Column Headers
        lbl_empty = QLabel("")
        lbl_h_aa = QLabel("Axis-Aligned")
        lbl_h_aa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_h_ob = QLabel("Oriented Box")
        lbl_h_ob.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_h_cyl = QLabel("Cylinder")
        lbl_h_cyl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        grid_suggest.addWidget(lbl_empty, 0, 0)
        grid_suggest.addWidget(lbl_h_aa, 0, 1)
        grid_suggest.addWidget(lbl_h_ob, 0, 2)
        grid_suggest.addWidget(lbl_h_cyl, 0, 3)
        
        # Row 1: Inscribed Row
        lbl_inscribed = QLabel("Inscribed")
        lbl_inscribed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        btn_style = "QPushButton { padding: 2px 2px; }"
        
        btn_aab = QPushButton("AAB")
        btn_aab.setStyleSheet(btn_style)
        btn_aab.clicked.connect(lambda: self.app.suggest_volume_action("aab_inscribed"))
        
        btn_ob = QPushButton("OB")
        btn_ob.setStyleSheet(btn_style)
        btn_ob.clicked.connect(lambda: self.app.suggest_volume_action("obb_inscribed"))
        
        grid_suggest.addWidget(lbl_inscribed, 1, 0)
        grid_suggest.addWidget(btn_aab, 1, 1)
        grid_suggest.addWidget(btn_ob, 1, 2)
        
        # Row 2: Circumscribed Row
        lbl_circumscribed = QLabel("Circumscribed")
        lbl_circumscribed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        btn_aabb = QPushButton("AABB")
        btn_aabb.setStyleSheet(btn_style)
        btn_aabb.clicked.connect(lambda: self.app.suggest_volume_action("aabb"))
        
        btn_obb = QPushButton("OBB")
        btn_obb.setStyleSheet(btn_style)
        btn_obb.clicked.connect(lambda: self.app.suggest_volume_action("obb"))
        
        btn_min_cc = QPushButton("min CC")
        btn_min_cc.setStyleSheet("QPushButton { padding: 2px 2px; background-color: #00adb5; color: white; font-weight: bold; }")
        btn_min_cc.clicked.connect(lambda: self.app.suggest_volume_action("cylinder_cc"))
        
        grid_suggest.addWidget(lbl_circumscribed, 2, 0)
        grid_suggest.addWidget(btn_aabb, 2, 1)
        grid_suggest.addWidget(btn_obb, 2, 2)
        grid_suggest.addWidget(btn_min_cc, 2, 3)
        
        auto_layout.addLayout(grid_suggest)
        layout.addWidget(group_auto)

        # --- Bottom OK button ---
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.close)
        btn_ok.setDefault(True)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        # Wire signals AFTER all widgets exist
        self.spin_vox_x.valueChanged.connect(self._on_voxels_changed)
        self.spin_vox_y.valueChanged.connect(self._on_voxels_changed)
        self.spin_vox_z.valueChanged.connect(self._on_voxels_changed)
        self.spin_vox_size.valueChanged.connect(self._on_vox_size_changed)
        self.spin_pos_x.valueChanged.connect(self._on_position_changed)
        self.spin_pos_y.valueChanged.connect(self._on_position_changed)
        self.spin_pos_z.valueChanged.connect(self._on_position_changed)
        self.slider_rot_x.valueChanged.connect(lambda v: self._on_rotation_changed('x', v))
        self.slider_rot_y.valueChanged.connect(lambda v: self._on_rotation_changed('y', v))
        self.slider_rot_z.valueChanged.connect(lambda v: self._on_rotation_changed('z', v))

    # ------------------------------------------------------------------
    # Public: call this after the config is loaded / model_matrix changes
    # ------------------------------------------------------------------
    def refresh_from_config(self):
        """Populate all controls from the current editor JSON."""
        self._updating = True
        try:
            config = self._parse_config()
            if config is None:
                return

            dims = config.get("voxel_dimensions", [100, 100, 100])
            M = np.array(config.get("model_matrix", np.eye(4).tolist()))

            self.spin_vox_x.setValue(int(dims[0]))
            self.spin_vox_y.setValue(int(dims[1]))
            self.spin_vox_z.setValue(int(dims[2]))

            vox_size = np.linalg.norm(M[:3, 0])
            self.spin_vox_size.setValue(float(vox_size))

            self.spin_pos_x.setValue(float(M[0, 3]))
            self.spin_pos_y.setValue(float(M[1, 3]))
            self.spin_pos_z.setValue(float(M[2, 3]))

            # Reset rotation sliders to 0 — rotations are incremental
            self.slider_rot_x.setValue(0)
            self.slider_rot_y.setValue(0)
            self.slider_rot_z.setValue(0)
            self._rot_x = 0.0
            self._rot_y = 0.0
            self._rot_z = 0.0

            self._update_matrix_display(M)
        finally:
            self._updating = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_config(self):
        try:
            return json.loads(self.app.editor.toPlainText())
        except Exception:
            return None

    def _update_matrix_display(self, M):
        rows = []
        for row in M:
            vals = ", ".join(f"{v:.4g}" for v in row)
            rows.append(f"[{vals}]")
        self.txt_matrix.setPlainText("[\n  " + ",\n  ".join(rows) + "\n]")

    def _write_matrix_to_config(self, M):
        config = self._parse_config()
        if config is None:
            return
        config["model_matrix"] = M.tolist()
        self.app.editor.blockSignals(True)
        self.app.editor.setPlainText(format_config_json(config))
        self.app.editor.blockSignals(False)
        self.app.update_from_editor()
        self._update_matrix_display(M)

    def _write_matrix_silent(self, M):
        """Write M to the JSON editor and update the display, but do NOT
        call update_from_editor() — used by the rotation handler so that
        mid-drag updates don't reset the sliders or trigger a full reload."""
        config = self._parse_config()
        if config is None:
            return
        config["model_matrix"] = M.tolist()
        self.app.editor.blockSignals(True)
        self.app.editor.setPlainText(format_config_json(config))
        self.app.editor.blockSignals(False)
        # Update the app's cached model_matrix so render_viewport draws the new box
        self.app.model_matrix = M.copy()
        self._update_matrix_display(M)
        self.app.render_viewport()

    def _get_current_matrix(self):
        config = self._parse_config()
        if config is None:
            return None
        return np.array(config.get("model_matrix", np.eye(4).tolist()))

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------
    def _on_voxels_changed(self):
        if self._updating:
            return
        config = self._parse_config()
        if config is None:
            return
        config["voxel_dimensions"] = [
            self.spin_vox_x.value(),
            self.spin_vox_y.value(),
            self.spin_vox_z.value(),
        ]
        self.app.editor.blockSignals(True)
        self.app.editor.setPlainText(format_config_json(config))
        self.app.editor.blockSignals(False)
        self.app.update_from_editor()

    def _on_vox_size_changed(self):
        if self._updating:
            return
        M = self._get_current_matrix()
        if M is None:
            return
        
        # Calculate current voxel size as the norm of the first column of model_matrix
        old_size = np.linalg.norm(M[:3, 0])
        new_size = self.spin_vox_size.value()
        if abs(old_size) < 1e-12:
            # If the matrix was zero/degenerate, let's make it a standard identity scaling
            M[:3, :3] = np.eye(3) * new_size
            self._write_matrix_to_config(M)
            return
            
        f = new_size / old_size
        if abs(f - 1.0) < 1e-10:
            return
            
        # Scale about the volume center to keep it in place
        Nx, Ny, Nz = (self.spin_vox_x.value(),
                       self.spin_vox_y.value(),
                       self.spin_vox_z.value())
        center_vox = np.array([Nx / 2.0, Ny / 2.0, Nz / 2.0, 1.0])
        center_world = M @ center_vox
        
        M[:3, :3] = M[:3, :3] * f
        new_center = M @ center_vox
        M[:3, 3] += center_world[:3] - new_center[:3]
        
        self._updating = True
        self.spin_pos_x.setValue(float(M[0, 3]))
        self.spin_pos_y.setValue(float(M[1, 3]))
        self.spin_pos_z.setValue(float(M[2, 3]))
        self._updating = False
        
        self._write_matrix_to_config(M)

    def _on_position_changed(self):
        if self._updating:
            return
        M = self._get_current_matrix()
        if M is None:
            return
        M[0, 3] = self.spin_pos_x.value()
        M[1, 3] = self.spin_pos_y.value()
        M[2, 3] = self.spin_pos_z.value()
        self._write_matrix_to_config(M)

    def _on_rotation_changed(self, axis, slider_value):
        """Incremental rotation: apply inverse of old angle, then new angle."""
        if self._updating:
            return
        M = self._get_current_matrix()
        if M is None:
            return

        new_angle_deg = slider_value / 10.0
        if axis == 'x':
            old_angle_deg = self._rot_x
            self._rot_x = new_angle_deg
        elif axis == 'y':
            old_angle_deg = self._rot_y
            self._rot_y = new_angle_deg
        else:
            old_angle_deg = self._rot_z
            self._rot_z = new_angle_deg

        delta_rad = np.radians(new_angle_deg - old_angle_deg)
        if abs(delta_rad) < 1e-10:
            return

        # Rotate about volume center
        Nx, Ny, Nz = (self.spin_vox_x.value(),
                       self.spin_vox_y.value(),
                       self.spin_vox_z.value())
        center_vox = np.array([Nx / 2.0, Ny / 2.0, Nz / 2.0, 1.0])
        center_world = M @ center_vox

        c, s = np.cos(delta_rad), np.sin(delta_rad)
        if axis == 'x':
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 'y':
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])

        # Apply rotation to the 3x3 part of model_matrix
        M[:3, :3] = R @ M[:3, :3]
        # Recompute translation so the volume center stays fixed
        new_center = M @ center_vox
        M[:3, 3] += center_world[:3] - new_center[:3]

        self._updating = True
        self.spin_pos_x.setValue(float(M[0, 3]))
        self.spin_pos_y.setValue(float(M[1, 3]))
        self.spin_pos_z.setValue(float(M[2, 3]))
        self._updating = False

        self._write_matrix_silent(M)

    def _scale_action(self):
        self.app.scale_factor_spin = self.scale_factor_spin
        self.app.scale_volume_action()
        M = self._get_current_matrix()
        if M is not None:
            self._updating = True
            self.spin_pos_x.setValue(float(M[0, 3]))
            self.spin_pos_y.setValue(float(M[1, 3]))
            self.spin_pos_z.setValue(float(M[2, 3]))
            vox_size = np.linalg.norm(M[:3, 0])
            self.spin_vox_size.setValue(float(vox_size))
            self._updating = False
            self._update_matrix_display(M)

    def closeEvent(self, event):
        # Uncollapse left sidebar in parent app
        self.app.left_widget.setVisible(True)
        super().closeEvent(event)


class ViewportWidget(QSvgWidget):
    viewChanged = pyqtSignal()
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from ProjectiveGeometry23.homography import rotation_x, rotation_z
        self.default_R = rotation_x(-0.7)[:3, :3] @ rotation_z(0.0)[:3, :3]
        self.R_view = self.default_R.copy()
        
        self.yaw = 0.0
        self.pitch = -0.7
        
        self.default_s = 0.2
        self.s = self.default_s
        
        self.tx = 0.0
        self.ty = 0.0
        
        self.last_mouse_pos = None
        self.P_display = None
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def H_translation(self):
        return np.array([
            [1.0, 0.0, self.tx],
            [0.0, 1.0, self.ty],
            [0.0, 0.0, 1.0]
        ])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            self.P_display = ProjectionMatrix.perspective_look_at(
                eye=np.array([0, 0, 250]),
                center=np.array([0, 0, 0]),
                image_size=(w, h),
                fovy_rad=0.7
            )
        self.viewChanged.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.last_mouse_pos is not None:
            pos = event.position()
            diff = pos - self.last_mouse_pos
            self.last_mouse_pos = pos
            
            if event.buttons() & Qt.MouseButton.LeftButton:
                # Update pitch and yaw (turntable rotation around world Z-axis)
                self.yaw -= diff.x() * 0.01
                self.pitch -= diff.y() * 0.01
                
                from ProjectiveGeometry23.homography import rotation_x, rotation_z
                self.R_view = rotation_x(self.pitch)[:3, :3] @ rotation_z(self.yaw)[:3, :3]
                
                self.viewChanged.emit()
            elif event.buttons() & Qt.MouseButton.RightButton:
                # Update 2D translation (right drag controls horizontal and vertical translation)
                self.tx += diff.x()
                self.ty += diff.y()
                
                # Do not allow translation to be larger than half of the SVG size
                w, h = self.width(), self.height()
                max_tx = w / 2.0
                max_ty = h / 2.0
                self.tx = max(-max_tx, min(max_tx, self.tx))
                self.ty = max(-max_ty, min(max_ty, self.ty))
                
                self.viewChanged.emit()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.s *= 1.1
        elif delta < 0:
            self.s /= 1.1
        # Clamp scale factor
        self.s = max(0.001, min(100.0, self.s))
        self.viewChanged.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()


def print_highlighted_config(config_path):
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Truncate image_files if it is a list and contains more than 3 elements
        if "image_files" in config and isinstance(config["image_files"], list) and len(config["image_files"]) > 3:
            orig_images = config["image_files"]
            truncated_images = [orig_images[0], orig_images[1], "...", orig_images[-1]]
            config_copy = config.copy()
            config_copy["image_files"] = truncated_images
        else:
            config_copy = config
            
        json_str = json.dumps(config_copy, indent=4)
        
        try:
            from rich import print as rprint
            from rich.json import JSON
            rprint(JSON(json_str))
        except ImportError:
            print(json_str)
    except Exception as e:
        print(f"[WARNING] Could not print config JSON: {e}")


class SelectableStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel()
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.addWidget(self._label, 1)

    def showMessage(self, message, timeout=0):
        self._label.setText(message)
        super().showMessage("", timeout)

    def clearMessage(self):
        self._label.setText("")
        super().clearMessage()


class ChooseECCDirectoryDialog(QDialog):
    def __init__(self, parent, ecc_directories):
        from PyQt6.QtWidgets import QListWidget
        super().__init__(parent)
        self.setWindowTitle("Continue Calibration or Start New")
        self.setMinimumWidth(450)
        self.result_mode = None  # "continue" or "new"
        self.selected_dir = None
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel(
            "<b>Existing ECC Calibration folders were found:</b><br>"
            "Please select a folder to continue an existing calibration, or start a new one."
        ))
        
        self.list_widget = QListWidget()
        self.list_widget.addItems(ecc_directories)
        self.list_widget.currentRowChanged.connect(self.update_buttons)
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        
        self.btn_continue = QPushButton("Continue Selected")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.on_continue)
        btn_layout.addWidget(self.btn_continue)
        
        self.btn_new = QPushButton("Create New Correction")
        self.btn_new.clicked.connect(self.on_new)
        btn_layout.addWidget(self.btn_new)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        if len(ecc_directories) > 0:
            self.list_widget.setCurrentRow(0)
            
    def update_buttons(self, row):
        self.btn_continue.setEnabled(row >= 0)
        
    def on_continue(self):
        self.result_mode = "continue"
        self.selected_dir = self.list_widget.currentItem().text()
        self.accept()
        
    def on_new(self):
        self.result_mode = "new"
        self.accept()


class ReconstructionGUIApp(QMainWindow):
    def __init__(self, config_path=None, edit_config_only=False):
        super().__init__()
        self.setWindowTitle("X-Ray CT Reconstruction GUI")
        self.resize(1280, 800)
        self.edit_config_only = edit_config_only
        
        # Check if GeometryCorrectionGUI is available dynamically at instantiation
        self.has_calibration_gui = False
        try:
            from geometry_correction_gui import GeometryCorrectionGUI
            self.has_calibration_gui = True
        except ImportError:
            try:
                _dir = os.path.dirname(os.path.abspath(__file__))
                _reconstruct_root = os.path.dirname(_dir)
                _intenso_root = os.path.dirname(_reconstruct_root)
                _calib_dir = os.path.join(_intenso_root, "ct_calibration_correction_gui")
                if _calib_dir not in sys.path and os.path.exists(_calib_dir):
                    sys.path.insert(0, _calib_dir)
                from geometry_correction_gui import GeometryCorrectionGUI
                self.has_calibration_gui = True
            except ImportError:
                self.has_calibration_gui = False
        
        self.P_list = []
        self.P_list_preview = None    # set by ParameterizationWindow when Preview is on
        self.john_quadric = None
        self.cached_volumes = None
        self._trajectory_cache = None
        self._cached_sdg_maps = {}
        self.voxel_dimensions = None
        self.model_matrix = np.eye(4)
        self.current_config_path = None
        
        self.isocenter = np.array([0.0, 0.0, 0.0])
        self.rotation_axis = np.array([0.0, 0.0, 1.0])
        self.source_positions_hom = []
        self.T_align = np.eye(4)
        self.mean_sid = 0.0
        self.mean_sdd = 0.0
        self.unit = "mm"
        self.trajectory_dirty = False
        self.last_loaded_ompl_path = None
        self.last_rendered_svg = None
        
        self.init_ui()
        
        # Debounce timer for auto-updating view when JSON text changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500) # 500 ms delay
        self.update_timer.timeout.connect(self.update_from_editor)
        self.editor.textChanged.connect(self.update_timer.start)
        
        # Try loading configuration file or fall back to bundled example data
        if config_path:
            self.load_config(config_path)
        else:
            default_cfg = None
            try:
                import ct_recon_fdk_astra as _recon
                default_cfg = str(_recon.get_data_path(
                    "example_data", "fullscan_180views_600x400.json"
                ))
            except Exception:
                pass
            if default_cfg and os.path.exists(default_cfg):
                self.load_config(default_cfg)
            else:
                self.statusBar().showMessage("Ready. Load a configuration file to begin.")

    def init_ui(self):
        self._setup_menu_bar()
        # Main Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # 1. Left Editor Panel
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        
        lbl_editor = QLabel("Configuration JSON Editor")
        lbl_editor.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select config JSON file...")
        self.path_edit.setReadOnly(True)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_config_file)
        self.btn_reveal = QPushButton("📁")
        self.btn_reveal.setToolTip("Reveal config directory in file manager")
        self.btn_reveal.setFixedWidth(32)
        self.btn_reveal.clicked.connect(self.reveal_config_dir)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_browse)
        path_layout.addWidget(self.btn_reveal)
        
        if self.edit_config_only:
            self.path_edit.hide()
            self.btn_browse.hide()
            self.btn_reveal.hide()

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Paste or write your configuration JSON here...")
        
        btn_edit_volume = QPushButton("Edit Reconstruction Volume...")
        btn_edit_volume.clicked.connect(self.open_volume_window)

        if self.has_calibration_gui:
            self.btn_calibration = QPushButton("ECC Calibration Correction...")
            self.btn_calibration.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
            self.btn_calibration.clicked.connect(self.open_calibration_gui_action)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Reconstruction Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["ram-lak", "shepp-logan", "cosine", "hamming", "hann", "none"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)

        self.chk_use_memmap = QCheckBox("Use Disk Memory-Mapping (Low RAM)")
        self.chk_use_memmap.setChecked(True)
        self.chk_use_memmap.stateChanged.connect(self.on_use_memmap_changed)

        line_integral_layout = QHBoxLayout()
        self.chk_convert_to_line_integral = QCheckBox("Convert to Line Integral")
        self.chk_convert_to_line_integral.setChecked(False)
        self.chk_convert_to_line_integral.stateChanged.connect(self.on_convert_to_line_integral_changed)
        
        self.combo_downsample = QComboBox()
        self.combo_downsample.addItems([
            "None",
            "0.8",
            "0.5",
            "0.333",
            "skip 2 @ 0.8",
            "skip 2 @ 0.5",
            "skip 2 @ 0.333",
            "skip 3 @ 0.333",
            "skip 4 @ 0.25"
        ])
        self.combo_downsample.currentTextChanged.connect(self.on_downsample_changed)
        
        line_integral_layout.addWidget(self.chk_convert_to_line_integral)
        line_integral_layout.addWidget(QLabel("Downsample:"))
        line_integral_layout.addWidget(self.combo_downsample)
        line_integral_layout.addStretch()

        mask_cyl_layout = QHBoxLayout()
        lbl_mask_cyl = QLabel("Mask Cylinder scale (0.0=off):")
        self.spin_mask_cylinder = QDoubleSpinBox()
        self.spin_mask_cylinder.setRange(0.0, 10.0)
        self.spin_mask_cylinder.setSingleStep(0.001)
        self.spin_mask_cylinder.setDecimals(4)
        self.spin_mask_cylinder.setValue(0.0)
        self.spin_mask_cylinder.valueChanged.connect(self.on_mask_cylinder_changed)
        mask_cyl_layout.addWidget(lbl_mask_cyl)
        mask_cyl_layout.addWidget(self.spin_mask_cylinder)

        btn_layout = QHBoxLayout()
        if self.edit_config_only:
            self.btn_save_close = QPushButton("Save and Close")
            self.btn_save_close.clicked.connect(self.save_and_close)
            btn_layout.addWidget(self.btn_save_close)
        else:
            self.btn_save = QPushButton("Save")
            self.btn_save.clicked.connect(self.save_config)
            self.btn_reconstruct = QPushButton("Save and Reconstruct")
            self.btn_reconstruct.clicked.connect(self.run_reconstruction)
            self.btn_reconstruct.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
            btn_layout.addWidget(self.btn_save)
            btn_layout.addWidget(self.btn_reconstruct)
        
        left_layout.addWidget(lbl_editor)
        left_layout.addLayout(path_layout)
        left_layout.addWidget(self.editor)
        left_layout.addWidget(btn_edit_volume)
        left_layout.addLayout(filter_layout)
        left_layout.addWidget(self.chk_use_memmap)
        left_layout.addLayout(line_integral_layout)
        left_layout.addLayout(mask_cyl_layout)
        
        # 2. Center Viewport Panel
        self.viewport = ViewportWidget()
        self.viewport.viewChanged.connect(self.render_viewport)
        self.viewport.doubleClicked.connect(self.toggle_sidebars)
        
        # 3. Right Sidebar Panel
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 5, 6, 10)
        
        # File info panel
        grp_info = QGroupBox("File Info")
        info_layout = QVBoxLayout(grp_info)
        info_layout.setContentsMargins(6, 8, 6, 6)
        self.txt_file_info = QLabel()
        self.txt_file_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.txt_file_info.setText(
            "Number of views: 0\n"
            "Detector size: N/A\n"
            "Pixel spacing: N/A\n"
            "Mean SID: N/A\n"
            "Mean SDD: N/A\n"
            "Isocenter:\n  N/A\n"
            "Plane of rotation:\n  N/A"
        )
        info_layout.addWidget(self.txt_file_info)
        
        
        # scale_factor_spin is now owned by ReconstructionVolumeWindow;
        # keep a default instance so scale_volume_action() can always find it.
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.01, 100.0)
        self.scale_factor_spin.setValue(0.5)
        self.scale_factor_spin.setSingleStep(0.1)
        
        # Visualization Options Group
        grp_active = QGroupBox("Visualization")
        active_layout = QVBoxLayout(grp_active)
        active_layout.setContentsMargins(6, 8, 6, 6)
        
        slider_layout = QHBoxLayout()

        self.btn_play = QPushButton("\u25b6")
        self.btn_play.setFixedWidth(28)
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._toggle_play)
        slider_layout.addWidget(self.btn_play)

        self.active_view_slider = QSlider(Qt.Orientation.Horizontal)
        self.active_view_slider.setRange(0, 0)
        self.active_view_spin = QSpinBox()
        self.active_view_spin.setRange(0, 0)
        self.active_view_spin.setFixedWidth(60)
        
        self.active_view_slider.valueChanged.connect(self.active_view_spin.setValue)
        self.active_view_spin.valueChanged.connect(self.active_view_slider.setValue)
        self.active_view_slider.valueChanged.connect(self.render_viewport)
        
        slider_layout.addWidget(self.active_view_slider)
        slider_layout.addWidget(self.active_view_spin)
        active_layout.addLayout(slider_layout)

        self._play_timer = QTimer()
        self._play_timer.setInterval(100)
        self._play_timer.timeout.connect(self._advance_view)

        # Checkboxes above "View Mode:"
        checkbox_layout = QHBoxLayout()
        self.chk_axis_labels = QCheckBox("Axis labels")
        self.chk_axis_labels.setChecked(True)
        self.chk_axis_labels.toggled.connect(self.render_viewport)
        checkbox_layout.addWidget(self.chk_axis_labels)
        
        self.chk_trajectory = QCheckBox("Trajectory")
        self.chk_trajectory.setChecked(True)
        self.chk_trajectory.toggled.connect(self.render_viewport)
        checkbox_layout.addWidget(self.chk_trajectory)
        
        active_layout.addLayout(checkbox_layout)
        
        view_mode_layout = QHBoxLayout()
        view_mode_layout.addWidget(QLabel("View Mode:"))
        self.combo_view_follows = QComboBox()
        self.combo_view_follows.addItems(["Turntable", "Gantry", "Laminography"])
        self.combo_view_follows.setCurrentText("Turntable")
        self.combo_view_follows.currentTextChanged.connect(self.on_view_follows_changed)
        view_mode_layout.addWidget(self.combo_view_follows)
        
        self.combo_up_axis = QComboBox()
        self.combo_up_axis.addItems(["X", "Y", "Z"])
        self.combo_up_axis.setCurrentText("X")
        self.combo_up_axis.setEnabled(False)
        self.combo_up_axis.currentTextChanged.connect(self.render_viewport)
        view_mode_layout.addWidget(self.combo_up_axis)
        
        active_layout.addLayout(view_mode_layout)

        det_vis_layout = QHBoxLayout()
        det_vis_layout.addWidget(QLabel("Detector Mode:"))
        self.combo_det_vis = QComboBox()
        self.combo_det_vis.addItems(["Physical Location", "Virtual (Iso-Center)"])
        self.combo_det_vis.setCurrentText("Physical Location")
        self.combo_det_vis.currentTextChanged.connect(self.render_viewport)
        det_vis_layout.addWidget(self.combo_det_vis)
        
        active_layout.addLayout(det_vis_layout)
        
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("Region of Interest:"))
        self.combo_roi = QComboBox()
        self.combo_roi.addItems(["Hidden", "John's Quadric", "Circumscribed Cylinder"])
        self.combo_roi.setCurrentText("Circumscribed Cylinder")
        self.combo_roi.currentTextChanged.connect(self.render_viewport)
        roi_layout.addWidget(self.combo_roi)
        active_layout.addLayout(roi_layout)
        
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        zoom_label = QLabel("Zoom:")
        zoom_label.setFixedWidth(38)
        zoom_layout.addWidget(zoom_label)
        
        self.btn_zoom_sdd = QPushButton("Detectors")
        self.btn_zoom_sdd.clicked.connect(self.zoom_to_sdd)
        self.btn_zoom_trajectory = QPushButton("Sources")
        self.btn_zoom_trajectory.clicked.connect(self.zoom_to_trajectory)
        self.btn_zoom_volume = QPushButton("Volume")
        self.btn_zoom_volume.clicked.connect(self.zoom_to_recon_volume)
        
        btn_style = "QPushButton { padding: 3px 5px; }"
        self.btn_zoom_sdd.setStyleSheet(btn_style)
        self.btn_zoom_trajectory.setStyleSheet(btn_style)
        self.btn_zoom_volume.setStyleSheet(btn_style)
        
        zoom_layout.addWidget(self.btn_zoom_sdd)
        zoom_layout.addWidget(self.btn_zoom_trajectory)
        zoom_layout.addWidget(self.btn_zoom_volume)
        active_layout.addLayout(zoom_layout)
        
        # Active View Details Panel
        grp_details = QGroupBox("Active View Details")
        details_layout = QVBoxLayout(grp_details)
        details_layout.setContentsMargins(6, 8, 6, 6)
        self.txt_active_details = QLabel()
        self.txt_active_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.txt_active_details.setText(
            "Source Position:\n  N/A\n"
            "Principal Ray:\n  N/A\n"
            "Projection Matrix [mm] -> detector pixels:\nN/A"
        )
        details_layout.addWidget(self.txt_active_details)
        
        # Assemble Right Panel
        right_layout.addWidget(grp_info)
        right_layout.addWidget(grp_active)
        right_layout.addWidget(grp_details)
        
        # Spacer/Stretch to push buttons to the bottom
        right_layout.addStretch(1)
        
        if self.has_calibration_gui:
            right_layout.addWidget(self.btn_calibration)
            
        btn_layout_right = QHBoxLayout()
        if self.edit_config_only:
            btn_layout_right.addWidget(self.btn_save_close)
        else:
            btn_layout_right.addWidget(self.btn_save)
            btn_layout_right.addWidget(self.btn_reconstruct)
        right_layout.addLayout(btn_layout_right)
        
        self.right_scroll.setWidget(right_widget)

        # Volume window and parameterization window (created lazily)
        self._volume_win = None
        self._param_win = None
        self._projection_windows = []  # list of open ProjectionViewWindow instances
        
        # Add to Splitter
        main_splitter.addWidget(self.left_widget)
        main_splitter.addWidget(self.viewport)
        main_splitter.addWidget(self.right_scroll)
        
        # Prevent panels from collapsing completely or getting cut off
        # Left widget is resizable from 330, right widget is fixed at 305
        self.left_widget.setMinimumWidth(330)
        self.viewport.setMinimumWidth(200)
        self.right_scroll.setFixedWidth(305)
        
        # Set proportional sizing (Editor: 30%, Viewport: 50%, Sidebar: 20%)
        main_splitter.setSizes([330, 650, 305])
        
        # 4. Status Bar
        self.setStatusBar(SelectableStatusBar())

    def toggle_sidebars(self):
        is_visible = self.left_widget.isVisible() or self.right_scroll.isVisible()
        self.left_widget.setVisible(not is_visible)
        self.right_scroll.setVisible(not is_visible)

    def _toggle_play(self, checked):
        if checked:
            self.btn_play.setText("\u23f8")
            self._play_timer.start()
        else:
            self.btn_play.setText("\u25b6")
            self._play_timer.stop()

    def _advance_view(self):
        max_val = self.active_view_slider.maximum()
        if max_val <= 0:
            return
        cur = self.active_view_slider.value()
        self.active_view_slider.setValue(0 if cur >= max_val else cur + 1)

    def open_volume_window(self):
        if self._volume_win is None:
            self._volume_win = ReconstructionVolumeWindow(self)
        self._volume_win.refresh_from_config()
        self._volume_win.show()
        self._volume_win.raise_()
        self._volume_win.activateWindow()
        
        # Collapse left sidebar only
        self.left_widget.setVisible(False)
        
        # Zoom to Volume
        self.zoom_to_recon_volume()

    def open_calibration_gui_action(self):
        if not self.current_config_path:
            QMessageBox.warning(self, "Save Config First", 
                               "Please save the reconstruction config JSON first, so that the calibration tool knows which dataset to load.")
            return
            
        # Make sure current changes are saved
        if self.editor.document().isModified() or getattr(self, 'trajectory_dirty', False):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Would you like to save them before opening the calibration tool?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_config():
                    return
            elif reply == QMessageBox.StandardButton.Discard:
                pass
            else:
                return

        # Check for ECC_* directories in the parent directory of reconstruction.json
        recon_dir = os.path.dirname(os.path.abspath(self.current_config_path))
        ecc_dirs = []
        if os.path.isdir(recon_dir):
            try:
                for entry in os.listdir(recon_dir):
                    entry_path = os.path.join(recon_dir, entry)
                    if os.path.isdir(entry_path) and entry.upper().startswith("ECC_"):
                        ecc_dirs.append(entry)
            except Exception as e:
                print(f"Error listing parent directory: {e}")

        load_path = self.current_config_path
        if ecc_dirs:
            # Ask the user if they want to continue with an existing ECC folder or start fresh
            dlg = ChooseECCDirectoryDialog(self, sorted(ecc_dirs))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            
            if dlg.result_mode == "continue" and dlg.selected_dir:
                selected_dir_path = os.path.join(recon_dir, dlg.selected_dir)
                geom_json_path = os.path.join(selected_dir_path, "geometry_correction.json")
                if os.path.exists(geom_json_path):
                    load_path = geom_json_path
                else:
                    QMessageBox.warning(
                        self, "Missing Configuration",
                        f"Could not find 'geometry_correction.json' in {dlg.selected_dir}.\n"
                        "Starting a new correction from scratch instead."
                    )
            elif dlg.result_mode == "new":
                # Create a new correction from scratch
                pass
            else:
                return
        
        print(f"\n[INFO] Passing configuration file to GeometryCorrectionGUI: {load_path}")
        if os.path.exists(load_path):
            print_highlighted_config(load_path)
        
        # Instantiate and load the GeometryCorrectionGUI
        from geometry_correction_gui import GeometryCorrectionGUI
        calib_gui = GeometryCorrectionGUI(parent_window=self)
        try:
            calib_gui.load_config_file(load_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load configuration:\n{e}")
            return
            
        if not calib_gui.P_list:
            # Load was cancelled or failed
            return
            
        # Hide Reconstruction GUI and show GeometryCorrectionGUI
        self.hide()
        calib_gui.show()
        calib_gui.raise_()
        calib_gui.activateWindow()
        
        # Keep window reference alive globally
        global _calib_gui_keepalive
        _calib_gui_keepalive = calib_gui

    def open_param_window(self):
        if self._param_win is None:
            self._param_win = ParameterizationWindow(self)
            # Wire paramsChanged to refresh all open projection windows
            self._param_win.paramsChanged.connect(self._refresh_projection_windows)
        self._param_win.show()
        self._param_win.raise_()
        self._param_win.activateWindow()

    def _refresh_projection_windows(self):
        for pw in list(self._projection_windows):
            pw.refresh()

    def open_projection_view(self):
        if not self.P_list:
            self.statusBar().showMessage("No trajectory loaded.")
            return
        idx = self.active_view_slider.value()
        pw = ProjectionViewWindow(self, idx)
        self._projection_windows.append(pw)
        # Wire paramsChanged if param window already exists
        if self._param_win is not None:
            self._param_win.paramsChanged.connect(pw.refresh)
        pw.show()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _setup_menu_bar(self):
        mb = self.menuBar()

        # ---- File ----
        m_file = mb.addMenu("File")
        m_file.addSeparator()
        
        # Grouped: Import Trajectory and Select Images
        a_import = QAction("Import Trajectory\u2026", self)
        a_import.triggered.connect(self.import_trajectory_action)
        m_file.addAction(a_import)
        
        a_select = QAction("Select Images\u2026", self)
        a_select.triggered.connect(self.select_images_action)
        m_file.addAction(a_select)
        
        m_file.addSeparator()
        
        # Export actions
        a_export_ompl = QAction("Export .ompl ...", self)
        a_export_ompl.triggered.connect(self.export_ompl_action)
        m_file.addAction(a_export_ompl)
        
        a_export_svg = QAction("Export view as SVG\u2026", self)
        a_export_svg.triggered.connect(self.export_view_as_svg_action)
        m_file.addAction(a_export_svg)
        
        a_export_sub = QAction("Export Sub-sampled Dataset\u2026", self)
        a_export_sub.triggered.connect(self.export_subsampled_dataset_action)
        m_file.addAction(a_export_sub)

        # ---- Transform ----
        m_transform = mb.addMenu("Transform")

        # Transform > Image
        m_image = m_transform.addMenu("Image")
        for label, fn in [
            ("Flip U Axis",       self.transform_flip_u),
            ("Flip V Axis",       self.transform_flip_v),
            ("Transpose Image",   self.transform_transpose),
        ]:
            a = QAction(label, self)
            a.triggered.connect(fn)
            m_image.addAction(a)
        m_image.addSeparator()
        a = QAction("Custom Matrix H\u2026", self)
        a.triggered.connect(self.transform_custom_image)
        m_image.addAction(a)

        # Transform > World
        m_world = m_transform.addMenu("World")
        for label, fn in [
            ("Flip X", self.transform_flip_x),
            ("Flip Y", self.transform_flip_y),
            ("Flip Z", self.transform_flip_z),
        ]:
            a = QAction(label, self)
            a.triggered.connect(fn)
            m_world.addAction(a)
        m_world.addSeparator()
        a = QAction("Custom Matrix T\u2026", self)
        a.triggered.connect(self.transform_custom_world)
        m_world.addAction(a)

        # Transform > Reconstruction Volume
        m_vol = m_transform.addMenu("Reconstruction Volume")
        a = QAction("Suggest", self)
        a.triggered.connect(self.suggest_volume_action)
        m_vol.addAction(a)
        a = QAction("Center", self)
        a.triggered.connect(self.center_volume_action)
        m_vol.addAction(a)
        m_vol.addSeparator()
        a = QAction("Edit Reconstruction Volume\u2026", self)
        a.triggered.connect(self.open_volume_window)
        m_vol.addAction(a)
        m_vol.addSeparator()

        # Transform > Reconstruction Volume > Swap Axes
        m_swap = m_vol.addMenu("Swap Axes")
        for label, fn in [
            ("X \u2194 Y", self.transform_swap_xy),
            ("X \u2194 Z", self.transform_swap_xz),
            ("Y \u2194 Z", self.transform_swap_yz),
        ]:
            a = QAction(label, self)
            a.triggered.connect(fn)
            m_swap.addAction(a)

        # ---- View ----
        m_view = mb.addMenu("View")
        a = QAction("Projectoin View...", self)
        a.triggered.connect(self.open_projection_view)
        m_view.addAction(a)
        if HAS_ECC:
            m_view.addSeparator()
            a = QAction("Parameterization\u2026", self)
            a.triggered.connect(self.open_param_window)
            m_view.addAction(a)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------
    def _apply_image_transform(self, H):
        """Apply 3x3 homography H to all projection matrices: Pi = H @ Pi."""
        if not self.P_list:
            self.statusBar().showMessage("No trajectory loaded.")
            return
        self.P_list = [
            ProjectionMatrix(H @ p.P, p.image_size, p.pixel_spacing)
            for p in self.P_list
        ]
        self.render_viewport()
        self.statusBar().showMessage("Image transform applied.")

    def _apply_world_transform(self, T):
        """Apply 4x4 world transform T to all projection matrices: Pi = Pi @ T."""
        if not self.P_list:
            self.statusBar().showMessage("No trajectory loaded.")
            return
        self.P_list = [
            ProjectionMatrix(p.P @ T, p.image_size, p.pixel_spacing)
            for p in self.P_list
        ]
        self.render_viewport()
        self.statusBar().showMessage("World transform applied.")

    def transform_flip_u(self):
        if not self.P_list:
            return
        W = self.P_list[0].image_size[0]
        H = np.array([[-1, 0, W - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
        self._apply_image_transform(H)

    def transform_flip_v(self):
        if not self.P_list:
            return
        Hpx = self.P_list[0].image_size[1]
        H = np.array([[1, 0, 0], [0, -1, Hpx - 1], [0, 0, 1]], dtype=float)
        self._apply_image_transform(H)

    def transform_transpose(self):
        if not self.P_list:
            return
        H = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        # Also swap detector W <-> H
        new_list = []
        for p in self.P_list:
            new_P = H @ p.P
            new_size = [p.image_size[1], p.image_size[0]]
            new_list.append(ProjectionMatrix(new_P, new_size, p.pixel_spacing))
        self.P_list = new_list
        self.render_viewport()
        self.statusBar().showMessage("Image transposed.")

    def _flip_volume_axis(self, axis):
        """Flip one world axis (0=X, 1=Y, 2=Z) by negating that column of model_matrix."""
        try:
            config = json.loads(self.editor.toPlainText())
            M = np.array(config.get("model_matrix", np.eye(4).tolist()))
            dims = config.get("voxel_dimensions", [1, 1, 1])
            # Build world T that flips axis through the volume center
            T = np.eye(4)
            T[axis, axis] = -1.0
            center_vox = np.array([dims[0] / 2, dims[1] / 2, dims[2] / 2, 1.0])
            center_world = M @ center_vox
            T[axis, 3] = 2 * center_world[axis]
            self._apply_world_transform(T)
        except Exception as e:
            QMessageBox.critical(self, "Flip Error", str(e))

    def transform_flip_x(self):
        self._flip_volume_axis(0)

    def transform_flip_y(self):
        self._flip_volume_axis(1)

    def transform_flip_z(self):
        self._flip_volume_axis(2)

    def _swap_volume_axes(self, a1, a2):
        """Swap two axes in the model_matrix and voxel_dimensions (changes slicing,
        not physical shape)."""
        try:
            config = json.loads(self.editor.toPlainText())
        except Exception as e:
            QMessageBox.critical(self, "JSON Error", str(e))
            return
        M = np.array(config.get("model_matrix", np.eye(4).tolist()))
        dims = list(config.get("voxel_dimensions", [1, 1, 1]))
        # Swap columns a1 and a2 in the 3x3 part (swaps voxel axes)
        M[:, [a1, a2]] = M[:, [a2, a1]]
        dims[a1], dims[a2] = dims[a2], dims[a1]
        config["model_matrix"] = M.tolist()
        config["voxel_dimensions"] = dims
        self.editor.blockSignals(True)
        self.editor.setPlainText(format_config_json(config))
        self.editor.blockSignals(False)
        self.update_from_editor()
        self.statusBar().showMessage(f"Swapped axes {a1} \u2194 {a2}.")

    def transform_swap_xy(self):
        self._swap_volume_axes(0, 1)

    def transform_swap_xz(self):
        self._swap_volume_axes(0, 2)

    def transform_swap_yz(self):
        self._swap_volume_axes(1, 2)

    def transform_custom_image(self):
        self._show_matrix_dialog(
            "Custom Image Transform (H)",
            np.eye(3),
            lambda H: self._apply_image_transform(H)
        )

    def transform_custom_world(self):
        self._show_matrix_dialog(
            "Custom World Transform (T)",
            np.eye(4),
            lambda T: self._apply_world_transform(T)
        )

    def _show_matrix_dialog(self, title, default_matrix, callback):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QVBoxLayout(dlg)
        rows, cols = default_matrix.shape
        table = QTableWidget(rows, cols)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for r in range(rows):
            for c in range(cols):
                table.setItem(r, c, QTableWidgetItem(f"{default_matrix[r, c]:.6g}"))
        layout.addWidget(table)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                M = np.array([
                    [float(table.item(r, c).text()) for c in range(cols)]
                    for r in range(rows)
                ])
                callback(M)
            except Exception as e:
                QMessageBox.critical(self, "Matrix Error", str(e))

    def scale_image_resolution_action(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Scale Image Resolution")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Scale factor (e.g. 0.5 = 2\u00d72 binning):"))
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setValue(0.5)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        layout.addWidget(spin)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        factor = spin.value()
        if not self.P_list:
            return
        new_list = []
        for p in self.P_list:
            P_new = p.P.copy()
            P_new[:2, :] *= factor          # scale the 2x4 image rows (K changes, geometry stays)
            new_size = [max(1, int(round(p.image_size[0] * factor))),
                        max(1, int(round(p.image_size[1] * factor)))]
            new_spacing = p.pixel_spacing / factor
            new_list.append(ProjectionMatrix(P_new, new_size, new_spacing))
        self.P_list = new_list
        self.render_viewport()
        self.statusBar().showMessage(f"Image resolution scaled by {factor:.3g}.")

    def select_images_action(self):
        filter_str = (
            "TIFF Images (*.tif *.tiff);;"
            "PIL Supported Types (*.png *.jpg *.jpeg);;"
            "NRRD Files (*.nrrd);;"
            "Raw Sequence (*.seq);;"
            "Processed Image (*.IMA);;"
            "All Files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Projection Images", "", filter_str
        )
        if not paths:
            return
            
        # Get location of the imported images
        images_dir = os.path.dirname(os.path.abspath(paths[0]))
        
        # Determine the relative path of data_dir based on config location
        if self.current_config_path:
            config_dir = os.path.dirname(os.path.abspath(self.current_config_path))
            try:
                data_dir_value = os.path.relpath(images_dir, config_dir)
                if data_dir_value.startswith(".." + os.sep + ".."):
                    data_dir_value = images_dir
            except Exception:
                data_dir_value = images_dir
        else:
            data_dir_value = images_dir
            
        # All image paths relative to the images_dir
        rel_paths = []
        for path in paths:
            try:
                rel = os.path.relpath(os.path.abspath(path), images_dir)
                rel_paths.append(rel)
            except Exception:
                rel_paths.append(os.path.basename(path))
                
        self.update_config_json_fields(data_dir=data_dir_value, image_files=rel_paths)

    def export_subsampled_dataset_action(self):
        if not self.current_config_path:
            QMessageBox.warning(self, "Export Error", "Please save or load a reconstruction config JSON first.")
            return
            
        json_text = self.editor.toPlainText().strip()
        if not json_text:
            QMessageBox.critical(self, "Error", "No configuration is currently open.")
            return
            
        try:
            config = json.loads(json_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON in editor: {str(e)}")
            return
            
        image_files = config.get("image_files", [])
        if not image_files:
            QMessageBox.warning(self, "Export Error", "No image files found in the current configuration.")
            return
            
        if not self.P_list:
            QMessageBox.warning(self, "Export Error", "No trajectory loaded.")
            return
            
        if len(self.P_list) != len(image_files):
            QMessageBox.warning(
                self,
                "Export Error",
                f"Number of projection matrices ({len(self.P_list)}) does not match the number of image files ({len(image_files)})."
            )
            return
            
        data_dir = config.get("data_dir", "")
        config_dir = os.path.dirname(os.path.abspath(self.current_config_path)) if self.current_config_path else os.getcwd()
        resolved_data_dir = data_dir
        if resolved_data_dir and not os.path.isabs(resolved_data_dir):
            resolved_data_dir = os.path.normpath(os.path.join(config_dir, resolved_data_dir))
        elif not resolved_data_dir:
            resolved_data_dir = config_dir
            
        resolved_image_paths = []
        for p in image_files:
            if os.path.isabs(p):
                resolved_image_paths.append(p)
            else:
                resolved_image_paths.append(os.path.normpath(os.path.join(resolved_data_dir, p)))

        dlg = ExportSubsampledDialog(self, image_paths=resolved_image_paths)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
            
        export_dir = dlg.edit_dir.text().strip()
        skip = dlg.spin_skip.value()
        factor = dlg.spin_factor.value()
        
        os.makedirs(export_dir, exist_ok=True)
        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        indices = list(range(0, len(image_files), skip))
        
        # We need save_ompl
        try:
            from fileformats import save_ompl
        except ImportError:
            from reconstruct.fileformats import save_ompl
            
        import tifffile
        from PIL import Image
        import nrrd
        
        exported_image_rel_paths = []
        new_P_list = []
        
        progress = QProgressDialog("Exporting downscaled images...", "Cancel", 0, len(indices), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)
        progress.show()
        
        try:
            for idx_step, idx in enumerate(indices):
                if progress.wasCanceled():
                    break
                    
                orig_path = resolved_image_paths[idx]
                if not os.path.exists(orig_path):
                    raise FileNotFoundError(f"Image file not found: {orig_path}")
                    
                # Read image
                if orig_path.lower().endswith('.nrrd'):
                    img, header = nrrd.read(orig_path)
                    img = np.squeeze(img)
                    if img.ndim == 2:
                        img = img.T
                else:
                    pil_img = Image.open(orig_path)
                    img = np.array(pil_img)
                    
                # Downscale
                downscaled = downscale_image(img, factor)
                
                # Filename logic
                base_name = os.path.basename(orig_path)
                root, ext = os.path.splitext(base_name)
                new_name = root + ".tif"
                dest_path = os.path.join(images_dir, new_name)
                
                # Write with tifffile
                tifffile.imwrite(
                    dest_path,
                    downscaled,
                    compression="zstd",
                    compressionargs={"level": 1},
                    tile=(256, 256),
                )
                
                exported_image_rel_paths.append(new_name)
                
                # Adapt projection matrix
                p_orig = self.P_list[idx]
                H = np.diag([1.0 / factor, 1.0 / factor, 1.0])
                P_new = H @ p_orig.P
                new_size = [max(1, int(round(p_orig.image_size[0] / factor))),
                            max(1, int(round(p_orig.image_size[1] / factor)))]
                new_spacing = p_orig.pixel_spacing * factor
                new_P_list.append(ProjectionMatrix(P_new, new_size, new_spacing))
                
                progress.setValue(idx_step + 1)
                QApplication.processEvents()
                
            if progress.wasCanceled():
                QMessageBox.information(self, "Export Cancelled", "The export process was cancelled.")
                return
                
            # Save OMPL file
            trajectory_ompl_path = os.path.join(export_dir, "trajectory.ompl")
            save_ompl(new_P_list, trajectory_ompl_path)
            
            # Save configuration JSON
            new_config = config.copy()
            new_config["data_dir"] = "./images"
            new_config["ompl_file"] = "trajectory.ompl"
            new_config["image_files"] = exported_image_rel_paths
            
            reconstruction_json_path = os.path.join(export_dir, "reconstruction.json")
            with open(reconstruction_json_path, 'w') as f:
                f.write(format_config_json(new_config))
                
            reply = QMessageBox.question(
                self,
                "Load Exported Dataset?",
                "Export completed successfully.\nDo you want to load the new downsampled dataset now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.load_config(reconstruction_json_path)
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")

    def export_ompl_action(self):
        if not self.P_list:
            QMessageBox.warning(self, "Export OMPL", "No trajectory / projection matrices loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Trajectory as OMPL", "", "OMPL Files (*.ompl);;All Files (*)"
        )
        if not path:
            return

        if not path.endswith('.ompl'):
            path += '.ompl'

        try:
            from fileformats import save_ompl

            save_ompl(
                self.P_list, 
                path, 
                spacing=self.P_list[0].pixel_spacing, 
                detector_size_px=list(self.P_list[0].image_size)
            )
            self.statusBar().showMessage(f"Successfully exported trajectory to {path}")
            QMessageBox.information(
                self, "Export Complete", f"Successfully exported trajectory to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export trajectory:\n{str(e)}")

    def export_view_as_svg_action(self):
        if not hasattr(self, 'last_rendered_svg') or not self.last_rendered_svg:
            QMessageBox.warning(self, "Export View as SVG", "No rendered view available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export View as SVG", "", "SVG Files (*.svg);;All Files (*)"
        )
        if not path:
            return

        if not path.lower().endswith('.svg'):
            path += '.svg'

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.last_rendered_svg)
            self.statusBar().showMessage(f"Successfully exported view as SVG to {path}")
            QMessageBox.information(
                self, "Export Complete", f"Successfully exported view as SVG to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export view as SVG:\n{str(e)}")

    def import_trajectory_action(self):
        loaders, _ = discover_formats()
        filters = []
        for name, info in loaders.items():
            ext_patterns = " ".join(f"*{ext}" for ext in info["extensions"])
            filters.append(f"{name} ({ext_patterns})")
        filters.append("All Files (*)")
        filter_str = ";;".join(filters)
        
        path, selected_filter = QFileDialog.getOpenFileName(
            self, "Import Trajectory", "", filter_str
        )
        if not path:
            return
            
        ext = os.path.splitext(path)[1].lower()
        
        loader_info = None
        for name, info in loaders.items():
            ext_patterns = " ".join(f"*{e}" for e in info["extensions"])
            filter_text = f"{name} ({ext_patterns})"
            if filter_text == selected_filter:
                loader_info = (name, info)
                break
                
        if not loader_info:
            matching = []
            for name, info in loaders.items():
                if ext in info["extensions"]:
                    matching.append((name, info))
            matching.sort(key=lambda item: 0 if item[0].lower() == "ompl" else 1)
            if matching:
                loader_info = matching[0]
                
        if not loader_info:
            QMessageBox.critical(self, "Import Error", f"Unsupported file extension: {ext}")
            return
            
        name, info = loader_info
        loader_fn = info["fn"]
        
        try:
            pixel_spacing = 1.0
            detector_size = [600, 400]
            if self.P_list:
                pixel_spacing = self.P_list[0].pixel_spacing
                detector_size = list(self.P_list[0].image_size)
                
            if name.lower() == "ompl":
                Ps = loader_fn(path)
            else:
                Ps = loader_fn(path, pixel_spacing=pixel_spacing, detector_size_px=detector_size)
                
            if not Ps:
                QMessageBox.warning(self, "Import Error", f"No matrices loaded by {name} loader.")
                return
                
            new_P_list = []
            for P in Ps:
                if hasattr(P, 'image_size'):
                    new_P_list.append(ProjectionMatrix(P.P.copy(), P.image_size.copy(), P.pixel_spacing))
                else:
                    new_P_list.append(ProjectionMatrix(P, detector_size, pixel_spacing))
                    
            self.P_list = new_P_list
            
            json_text = self.editor.toPlainText().strip()
            config = {}
            if json_text:
                try:
                    config = json.loads(json_text)
                except Exception:
                    pass
            
            ompl_file = config.get("ompl_file")
            if not ompl_file:
                if self.current_config_path:
                    ompl_file = os.path.splitext(os.path.basename(self.current_config_path))[0] + ".ompl"
                    config["ompl_file"] = ompl_file
                    self.editor.blockSignals(True)
                    self.editor.setPlainText(format_config_json(config))
                    self.editor.blockSignals(False)
                else:
                    QMessageBox.warning(
                        self, "Save Config First",
                        "Please save the reconstruction config JSON first, so that we know where to save the imported trajectory."
                    )
                    return
            
            ompl_path = resolve_relative(self.current_config_path, ompl_file)
            self.last_loaded_ompl_path = ompl_path
            self.trajectory_dirty = True
            
            # Inform the user that the file defined in JSON will be overwritten
            QMessageBox.information(
                self,
                "OMPL File Overwrite Info",
                f"The OMPL file '{ompl_file}' defined in the JSON will be overwritten when saving or starting reconstruction."
            )
            
            self.update_from_editor()
            self.editor.document().setModified(True)
            self.statusBar().showMessage(f"Successfully imported trajectory from {os.path.basename(path)} (changes pending save).")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import trajectory:\n{str(e)}")

    def browse_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Config JSON", self.current_config_path or "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.load_config(file_path)

    def reveal_config_dir(self):
        if not self.current_config_path:
            self.statusBar().showMessage("No configuration file loaded.")
            return
        config_dir = os.path.dirname(os.path.abspath(self.current_config_path))
        if os.path.exists(config_dir):
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(config_dir))
        else:
            self.statusBar().showMessage("Configuration directory does not exist.")

    def estimate_isocenter(self):
        """
        Calculate the isocenter as the point closest to all rays from
        the camera center to the physical detector center.
        """
        if not self.P_list:
            return np.array([0.0, 0.0, 0.0])
            
        from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
        A_mat = np.zeros((3, 3))
        b_vec = np.zeros(3)
        for p in self.P_list:
            # 1. Camera center (dehomogenized)
            C = p.getCenterOfProjection().flatten()
            if abs(C[3]) > 1e-12:
                C = C[:3] / C[3]
            else:
                C = C[:3]
                
            # 2. Ray from source to actual detector center
            sdg = SourceDetectorGeometry(p)
            O_det = sdg.detector_origin.flatten()[:3]
            U_det = sdg.axis_direction_Upx.flatten()[:3] * p.image_size[0]
            V_det = sdg.axis_direction_Vpx.flatten()[:3] * p.image_size[1]
            C_det = O_det + 0.5 * U_det + 0.5 * V_det
            
            # ray direction from source to detector center
            r = C_det - C
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                r /= r_norm
                
            # Projection matrix onto the plane orthogonal to r
            M_proj = np.eye(3) - np.outer(r, r)
            A_mat += M_proj
            b_vec += M_proj @ C
            
        return np.linalg.pinv(A_mat) @ b_vec

    def load_config(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Config JSON", "", "JSON Files (*.json);;All Files (*)"
            )
        if not file_path:
            return
            
        try:
            file_path = os.path.abspath(os.path.expanduser(file_path))
            with open(file_path, 'r') as f:
                json_text = f.read()
            self.editor.blockSignals(True)
            try:
                config = json.loads(json_text)
                formatted_json = format_config_json(config)
                self.editor.setPlainText(formatted_json)
            except:
                self.editor.setPlainText(json_text)
            self.editor.blockSignals(False)
            self.current_config_path = file_path
            self.path_edit.setText(file_path)
            self.last_loaded_ompl_path = None
            self.trajectory_dirty = False
            self.update_from_editor()
            self.editor.document().setModified(False)
            self.statusBar().showMessage(f"Loaded configuration from: {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading configuration file: {str(e)}")

    def on_filter_changed(self, filter_text):
        self.update_config_json_fields(filter_type=filter_text)

    def on_use_memmap_changed(self):
        self.update_config_json_fields(use_memmap=self.chk_use_memmap.isChecked())

    def on_convert_to_line_integral_changed(self):
        self.update_config_json_fields(convert_to_line_integral=self.chk_convert_to_line_integral.isChecked())

    def on_downsample_changed(self, text):
        mapping = {
            "None": [1, 1.0],
            "0.8": [1, 0.8],
            "0.5": [1, 0.5],
            "0.333": [1, 0.333],
            "skip 2 @ 0.8": [2, 0.8],
            "skip 2 @ 0.5": [2, 0.5],
            "skip 2 @ 0.333": [2, 0.333],
            "skip 3 @ 0.333": [3, 0.333],
            "skip 4 @ 0.25": [4, 0.25]
        }
        val = mapping.get(text, [1, 1.0])
        json_text = self.editor.toPlainText().strip()
        try:
            config = json.loads(json_text) if json_text else {}
        except:
            config = {}
            
        if val == [1, 1.0]:
            if "downsample" in config:
                del config["downsample"]
            self.editor.blockSignals(True)
            self.editor.setPlainText(format_config_json(config))
            self.editor.blockSignals(False)
            self.update_from_editor()
        else:
            self.update_config_json_fields(downsample=val)

    def on_mask_cylinder_changed(self):
        self.update_config_json_fields(mask_cylinder=self.spin_mask_cylinder.value())

    def save_config(self):
        file_path = self.current_config_path
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Config JSON", "", "JSON Files (*.json)"
            )
            if not file_path:
                return False
            self.current_config_path = file_path
            if not self.edit_config_only:
                self.path_edit.setText(file_path)
            
        try:
            # If trajectory is dirty, save OMPL trajectory
            if getattr(self, 'trajectory_dirty', False) and self.P_list:
                try:
                    config_json = json.loads(self.editor.toPlainText())
                except:
                    config_json = {}
                ompl_file = config_json.get("ompl_file")
                if ompl_file:
                    ompl_path = resolve_relative(self.current_config_path, ompl_file)
                    from fileformats import save_ompl
                    save_ompl(
                        self.P_list, 
                        ompl_path, 
                        spacing=self.P_list[0].pixel_spacing, 
                        detector_size_px=list(self.P_list[0].image_size)
                    )
                    self.trajectory_dirty = False
            
            with open(file_path, 'w') as f:
                f.write(self.editor.toPlainText())
            self.statusBar().showMessage(f"Saved configuration to: {file_path}")
            self.editor.document().setModified(False)
            return True
        except Exception as e:
            self.statusBar().showMessage(f"Error saving configuration file: {str(e)}")
            return False

    def save_and_close(self):
        if self.save_config():
            self.close()

    def closeEvent(self, event):
        # Close all open projection windows
        if hasattr(self, '_projection_windows') and self._projection_windows:
            for pw in list(self._projection_windows):
                try:
                    pw.close()
                except Exception:
                    pass
        # Close parameterization window if open
        if hasattr(self, '_param_win') and self._param_win:
            try:
                self._param_win.close()
            except Exception:
                pass

        if hasattr(self, 'console_win') and self.console_win:
            try:
                if not self.console_win.close():
                    event.ignore()
                    return
            except Exception:
                pass
        if self.editor.document().isModified() or getattr(self, 'trajectory_dirty', False):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Would you like to save them before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            if reply == QMessageBox.StandardButton.Save:
                if self.save_config():
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def update_config_json_fields(self, **kwargs):
        json_text = self.editor.toPlainText().strip()
        try:
            config = json.loads(json_text) if json_text else {}
        except:
            config = {}
            
        for k, v in kwargs.items():
            config[k] = v
            
        self.editor.blockSignals(True)
        self.editor.setPlainText(format_config_json(config))
        self.editor.blockSignals(False)
        self.update_from_editor()

    def suggest_volume_action(self, volume_type=None):
        if volume_type is None or isinstance(volume_type, bool):
            volume_type = "cylinder_cc"

        if not self.P_list:
            self.statusBar().showMessage("Error: No trajectory loaded.")
            return
            
        try:
            vols = getattr(self, 'cached_volumes', None)
            if vols is None:
                self.statusBar().showMessage("Error: No precomputed volumes available.")
                return
            
            if volume_type == "aabb":
                box = vols.box_aabb
                roi_mode = "Hidden"
            elif volume_type == "aab_inscribed":
                box = vols.box_aab_inscribed
                roi_mode = "Hidden"
            elif volume_type == "obb":
                box = vols.box_obb
                roi_mode = "John's Quadric"
            elif volume_type == "obb_inscribed":
                box = vols.box_obb_inscribed
                roi_mode = "John's Quadric"
            elif volume_type == "cylinder_cc":
                box = vols.cylinder_cc
                roi_mode = "Circumscribed Cylinder"
            else:
                raise ValueError(f"Unknown suggestion volume type: {volume_type}")
                
            if getattr(self, 'combo_roi', None) is not None:
                self.combo_roi.setCurrentText(roi_mode)
                
            model_matrix = box.model_matrix.copy()
            orig_shape = box.number_of_voxels.astype(float)
            
            # Round to the closest multiple of 32 (minimum 32)
            volume_shape = np.maximum(32 * np.round(orig_shape / 32), 32).astype(int)
            
            # Center the volume again (shift origin to keep the center in place)
            for i in range(3):
                model_matrix[:3, 3] -= 0.5 * (volume_shape[i] - orig_shape[i]) * model_matrix[:3, i]
            
            self.update_config_json_fields(
                voxel_dimensions=volume_shape.tolist(),
                model_matrix=model_matrix.tolist()
            )
            self.statusBar().showMessage(f"Volume suggestion ({volume_type}) applied successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Suggestion Error", f"Failed to suggest reconstruction volume:\n{e}")

    def scale_volume_action(self):
        json_text = self.editor.toPlainText().strip()
        if not json_text:
            return
        try:
            config = json.loads(json_text)
        except Exception as e:
            QMessageBox.critical(self, "JSON Error", f"Please fix JSON errors first:\n{e}")
            return
            
        if "voxel_dimensions" not in config or "model_matrix" not in config:
            QMessageBox.warning(self, "Missing Fields", "Config JSON must contain 'voxel_dimensions' and 'model_matrix'.")
            return
            
        factor = self.scale_factor_spin.value()
        if factor <= 0:
            return
            
        try:
            Nx, Ny, Nz = config["voxel_dimensions"]
            new_dims = [
                max(1, int(round(Nx * factor))),
                max(1, int(round(Ny * factor))),
                max(1, int(round(Nz * factor)))
            ]
            
            M = np.array(config["model_matrix"])
            M_new = M.copy()
            M_new[:3, :3] = M[:3, :3] / factor
            
            self.update_config_json_fields(
                voxel_dimensions=new_dims,
                model_matrix=M_new.tolist()
            )
            self.statusBar().showMessage(f"Volume resolution scaled by factor of {factor}.")
            
        except Exception as e:
            QMessageBox.critical(self, "Scaling Error", f"Failed to scale resolution:\n{e}")

    def center_volume_action(self):
        if not self.P_list:
            self.statusBar().showMessage("Error: No trajectory loaded.")
            return
            
        json_text = self.editor.toPlainText().strip()
        if not json_text:
            return
        try:
            config = json.loads(json_text)
        except Exception as e:
            QMessageBox.critical(self, "JSON Error", f"Please fix JSON errors first:\n{e}")
            return
            
        if "voxel_dimensions" not in config or "model_matrix" not in config:
            QMessageBox.warning(self, "Missing Fields", "Config JSON must contain 'voxel_dimensions' and 'model_matrix'.")
            return
            
        try:
            # Calculate isocenter as the point closest to all backprojection rays of the image center
            isocenter = self.estimate_isocenter()
            
            Nx, Ny, Nz = config["voxel_dimensions"]
            M = np.array(config["model_matrix"])
            
            voxel_center = np.array([Nx / 2.0, Ny / 2.0, Nz / 2.0, 1.0])
            volume_center = M @ voxel_center
            
            shift = isocenter - volume_center[:3]
            
            M_new = M.copy()
            M_new[:3, 3] = M[:3, 3] + shift
            
            self.update_config_json_fields(
                model_matrix=M_new.tolist()
            )
            self.statusBar().showMessage("Volume centered to isocenter successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Centering Error", f"Failed to center volume:\n{e}")

    def get_current_s_scale(self):
        if not self.P_list:
            return 1.0
        det_centers = []
        for pm, sdg in zip(self.P_list, self.get_sdg_list(self.P_list)):
            O_det = sdg.detector_origin.flatten()[:3]
            U_det = sdg.axis_direction_Upx.flatten()[:3] * pm.image_size[0]
            V_det = sdg.axis_direction_Vpx.flatten()[:3] * pm.image_size[1]
            C_det = O_det + 0.5 * U_det + 0.5 * V_det
            det_centers.append(C_det)
        
        a = self.rotation_axis.copy()
        norm_a = np.linalg.norm(a)
        if norm_a > 1e-6:
            a = a / norm_a
        else:
            a = np.array([0.0, 0.0, 1.0])
        
        distances = []
        for c_det in det_centers:
            d = c_det - self.isocenter
            d_orth = d - np.dot(d, a) * a
            distances.append(np.linalg.norm(d_orth))
        
        r_mean = np.mean(distances) if distances else 75.0
        D = 2.0 * r_mean
        return 250.0 / D if D > 1e-6 else 1.0

    def zoom_to_sdd(self):
        if getattr(self, 'combo_det_vis', None) is not None:
            self.combo_det_vis.setCurrentText("Physical Location")
        s_scale = self.get_current_s_scale()
        mean_sdd = getattr(self, 'mean_sdd', 0.0)
        if mean_sdd <= 0:
            mean_sdd = 1000.0
        R = mean_sdd
        self.viewport.s = 80.0 / (R * s_scale)
        self.viewport.s = max(0.001, min(100.0, self.viewport.s))
        self.render_viewport()
        self.statusBar().showMessage(f"Zoomed to SDD: {self.viewport.s:.4f}")

    def zoom_to_trajectory(self):
        s_scale = self.get_current_s_scale()
        mean_sid = getattr(self, 'mean_sid', 0.0)
        if mean_sid <= 0:
            mean_sid = 500.0
        R = mean_sid
        self.viewport.s = 80.0 / (R * s_scale)
        self.viewport.s = max(0.001, min(100.0, self.viewport.s))
        self.render_viewport()
        self.statusBar().showMessage(f"Zoomed to Source Trajectory: {self.viewport.s:.4f}")

    def zoom_to_recon_volume(self):
        if getattr(self, 'combo_det_vis', None) is not None:
            self.combo_det_vis.setCurrentText("Virtual (Iso-Center)")
        s_scale = self.get_current_s_scale()
        R = 50.0
        if self.voxel_dimensions is not None:
            shape = np.array([self.voxel_dimensions[0], self.voxel_dimensions[1], self.voxel_dimensions[2]])
            corners_local = np.array([
                [0, 0, 0],
                [shape[0], 0, 0],
                [0, shape[1], 0],
                [shape[0], shape[1], 0],
                [0, 0, shape[2]],
                [shape[0], 0, shape[2]],
                [0, shape[1], shape[2]],
                [shape[0], shape[1], shape[2]]
            ])
            corners_world = [self.model_matrix[:3, 3] + self.model_matrix[:3, :3] @ pt for pt in corners_local]
            dists = [np.linalg.norm(pt - self.isocenter) for pt in corners_world]
            if dists:
                R = max(dists)
        
        self.viewport.s = 80.0 / (R * s_scale)
        self.viewport.s = max(0.001, min(100.0, self.viewport.s))
        self.render_viewport()
        self.statusBar().showMessage(f"Zoomed to Reconstruction Volume: {self.viewport.s:.4f}")

    def on_view_follows_changed(self):
        mode = self.combo_view_follows.currentText()
        if hasattr(self, 'combo_up_axis'):
            self.combo_up_axis.setEnabled(mode == "Gantry")
        if mode in ("Turntable", "Laminography"):
            self.zoom_to_sdd()
        else:
            self.zoom_to_trajectory()

    def run_reconstruction(self):
        if not self.save_config():
            return
            
        config_path = self.current_config_path
        
        # Parse output_file from editor/config
        try:
            json_text = self.editor.toPlainText().strip()
            config = json.loads(json_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse configuration: {e}")
            return
            
        output_file = config.get("output_file", "reconstruction.nrrd")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        output_path = os.path.normpath(os.path.join(config_dir, output_file))
        
        if os.path.exists(output_path):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Output File Exists")
            msg_box.setText(f"The output file already exists:\n{output_path}\n\nWould you like to replace it or rename the output file?")
            
            replace_button = msg_box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
            rename_button = msg_box.addButton("Rename...", QMessageBox.ButtonRole.ActionRole)
            cancel_button = msg_box.addButton(QMessageBox.StandardButton.Cancel)
            msg_box.setDefaultButton(rename_button)
            
            msg_box.exec()
            clicked_button = msg_box.clickedButton()
            
            if clicked_button == rename_button:
                new_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Reconstruction Output As", output_path, "NRRD Files (*.nrrd);;All Files (*)"
                )
                if not new_path:
                    return # User cancelled
                
                # Make relative to config_dir if possible
                try:
                    rel_new_path = os.path.relpath(new_path, config_dir)
                    if rel_new_path.startswith(".." + os.sep + ".."):
                        rel_new_path = new_path
                except ValueError:
                    rel_new_path = new_path
                    
                self.update_config_json_fields(output_file=rel_new_path)
                if not self.save_config():
                    return
            elif clicked_button == replace_button:
                pass
            else:
                return # Cancelled

        # Resolve reconstruct.py path
        reconstruct_file = None
        try:
            import reconstruct
            reconstruct_file = os.path.abspath(reconstruct.__file__)
        except ImportError:
            pass

        if not reconstruct_file or not os.path.exists(reconstruct_file):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.normpath(os.path.join(cur_dir, "reconstruct.py")),
                os.path.normpath(os.path.join(cur_dir, "..", "reconstruct.py")),
                os.path.normpath(os.path.join(cur_dir, "..", "reconstruct", "reconstruct.py")),
                "/run/media/aaichert/Intenso/reconstruct/reconstruct.py"
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    reconstruct_file = p
                    break

        if not reconstruct_file or not os.path.exists(reconstruct_file):
            QMessageBox.critical(self, "Error", "Could not locate reconstruct.py")
            return

        if reconstruct_file.endswith('.pyc'):
            reconstruct_file = reconstruct_file[:-1]

        from gui.process_console import ProcessConsoleWindow

        cmd = f'"{sys.executable}" -u "{reconstruct_file}" "{config_path}"'
        self.console_win = ProcessConsoleWindow(
            command=cmd,
            working_dir=os.path.dirname(config_path),
            title="Reconstruction Progress Console",
            parent=self,
            show_progress=True
        )
        self.console_win.finished_signal.connect(
            lambda exit_code, log: self.on_reconstruction_finished_new(exit_code == 0, log)
        )
        self.console_win.finished.connect(lambda result: self.show())
        self.console_win.show()
        self.hide()

    def on_reconstruction_finished_new(self, success, log):
        if success:
            try:
                config_path = self.current_config_path
                with open(config_path, 'r') as f:
                    config = json.load(f)
                output_file = config.get("output_file", "reconstruction.nrrd")
                config_dir = os.path.dirname(os.path.abspath(config_path))
                output_path = os.path.normpath(os.path.join(config_dir, output_file))
                
                if os.path.exists(output_path):
                    # Find all other .nrrd files with same size in the same directory
                    output_dir = os.path.dirname(os.path.abspath(output_path))
                    target_size = os.path.getsize(output_path)
                    
                    nrrd_files = []
                    try:
                        for f_name in os.listdir(output_dir):
                            if f_name.lower().endswith('.nrrd'):
                                f_path = os.path.join(output_dir, f_name)
                                if os.path.isfile(f_path) and os.path.getsize(f_path) == target_size:
                                    nrrd_files.append(os.path.normpath(f_path))
                    except Exception as e:
                        print(f"Error scanning directory for nrrd files: {e}")
                        nrrd_files = [output_path]
                        
                    # Ensure output_path is the first one in the list, and no duplicates
                    if output_path in nrrd_files:
                        nrrd_files.remove(output_path)
                    nrrd_files.insert(0, output_path)
                    
                    # Run NrrdView3DWindow
                    try:
                        from gui.nrrd_view_3d import NrrdView3DWindow
                    except ImportError:
                        from tools.NrrdView3D.nrrd_view_3d import NrrdView3DWindow
                    self.nrrd_win = NrrdView3DWindow()
                    self.nrrd_win.show()
                    self.nrrd_win.open_file(nrrd_files)
                else:
                    QMessageBox.critical(
                        self, 
                        "Reconstruction Error", 
                        f"Reconstruction finished, but output file was not found at:\n{output_path}"
                    )
            except Exception as ex:
                QMessageBox.critical(
                    self, 
                    "Reconstruction Error", 
                    f"Failed to open reconstruction output:\n{str(ex)}"
                )
        else:
            QMessageBox.critical(
                self, 
                "Reconstruction Error", 
                "Reconstruction process failed. Please check the logs."
            )

    def update_trajectory_coordinates(self):
        """Precompute and cache the 3D source positions and detector centers."""
        disp_list = self.P_list_preview if self.P_list_preview is not None else self.P_list
        if not disp_list:
            self.cached_disp_src = []
            self.cached_disp_det = []
            return
            
        self.cached_disp_src = [pm.getCenterOfProjection().reshape(-1, 1) for pm in disp_list]
        self.cached_disp_det = []
        for pm, sdg in zip(disp_list, self.get_sdg_list(disp_list)):
            O_det = sdg.detector_origin.flatten()[:3]
            U_det = sdg.axis_direction_Upx.flatten()[:3] * pm.image_size[0]
            V_det = sdg.axis_direction_Vpx.flatten()[:3] * pm.image_size[1]
            C_det = O_det + 0.5 * U_det + 0.5 * V_det
            self.cached_disp_det.append(np.array([C_det[0], C_det[1], C_det[2], 1.0]).reshape(-1, 1))

    def get_sdg_list(self, p_list):
        if not p_list:
            return []
        state_key = tuple((id(p), p.pixel_spacing, tuple(p.P.flat)) for p in p_list)
        if not hasattr(self, '_cached_sdg_maps') or self._cached_sdg_maps is None:
            self._cached_sdg_maps = {}
        if state_key not in self._cached_sdg_maps:
            if len(self._cached_sdg_maps) > 5:
                self._cached_sdg_maps.clear()
            self._cached_sdg_maps[state_key] = [SourceDetectorGeometry(p) for p in p_list]
        return self._cached_sdg_maps[state_key]

    def update_from_editor(self):
        json_text = self.editor.toPlainText().strip()
        if not json_text:
            return
        try:
            config = json.loads(json_text)
        except Exception as e:
            self.statusBar().showMessage(f"JSON Parse Error: {str(e)}")
            return

        # Update filter combobox from config
        filter_type = config.get("filter_type", "ram-lak").lower()
        idx = self.filter_combo.findText(filter_type, Qt.MatchFlag.MatchExactly)
        if idx >= 0:
            self.filter_combo.blockSignals(True)
            self.filter_combo.setCurrentIndex(idx)
            self.filter_combo.blockSignals(False)

        # Update use_memmap checkbox from config
        use_memmap = config.get("use_memmap", True)
        self.chk_use_memmap.blockSignals(True)
        self.chk_use_memmap.setChecked(use_memmap)
        self.chk_use_memmap.blockSignals(False)

        # Update convert_to_line_integral checkbox from config
        convert_to_line_integral = config.get("convert_to_line_integral", False)
        self.chk_convert_to_line_integral.blockSignals(True)
        self.chk_convert_to_line_integral.setChecked(convert_to_line_integral)
        self.chk_convert_to_line_integral.blockSignals(False)

        # Update downsample combo from config
        downsample_val = config.get("downsample")
        if downsample_val and isinstance(downsample_val, list) and len(downsample_val) >= 2:
            try:
                loaded_skip = int(downsample_val[0])
                loaded_factor = float(downsample_val[1])
            except (ValueError, TypeError):
                loaded_skip = 1
                loaded_factor = 1.0
        else:
            loaded_skip = 1
            loaded_factor = 1.0

        options_map = {
            "None": [1, 1.0],
            "0.8": [1, 0.8],
            "0.5": [1, 0.5],
            "0.333": [1, 0.333],
            "skip 2 @ 0.8": [2, 0.8],
            "skip 2 @ 0.5": [2, 0.5],
            "skip 2 @ 0.333": [2, 0.333],
            "skip 3 @ 0.333": [3, 0.333],
            "skip 4 @ 0.25": [4, 0.25]
        }

        best_option = "None"
        min_dist = float('inf')
        for opt, val in options_map.items():
            dist = (val[0] - loaded_skip) ** 2 + (val[1] - loaded_factor) ** 2
            if dist < min_dist:
                min_dist = dist
                best_option = opt

        idx = self.combo_downsample.findText(best_option, Qt.MatchFlag.MatchExactly)
        if idx >= 0:
            self.combo_downsample.blockSignals(True)
            self.combo_downsample.setCurrentIndex(idx)
            self.combo_downsample.blockSignals(False)

        # Update mask_cylinder spinbox from config
        mask_cylinder_val = config.get("mask_cylinder", 0.0)
        if isinstance(mask_cylinder_val, bool):
            mask_cylinder = 1.0 if mask_cylinder_val else 0.0
        else:
            try:
                mask_cylinder = float(mask_cylinder_val)
            except (ValueError, TypeError):
                mask_cylinder = 0.0
        self.spin_mask_cylinder.blockSignals(True)
        self.spin_mask_cylinder.setValue(mask_cylinder)
        self.spin_mask_cylinder.blockSignals(False)
            
        if "ompl_file" not in config:
            self.statusBar().showMessage("Error: Config JSON must specify 'ompl_file'.")
            return
            
        ompl_file = config["ompl_file"]
        ompl_path = resolve_relative(self.current_config_path, ompl_file)
        
        # Determine if we actually need to reload the OMPL trajectory from disk
        has_p_list = hasattr(self, 'P_list') and self.P_list is not None and len(self.P_list) > 0
        last_loaded = getattr(self, 'last_loaded_ompl_path', None)
        path_changed = (ompl_path != last_loaded)
        
        needs_reload = path_changed or not has_p_list
        # If we have dirty trajectory changes, do NOT reload from disk unless the path changed
        if needs_reload and getattr(self, 'trajectory_dirty', False) and not path_changed:
            needs_reload = False
            
        trajectory_changed = needs_reload or getattr(self, 'trajectory_dirty', False)
        
        if trajectory_changed:
            try:
                if needs_reload:
                    if not os.path.exists(ompl_path):
                        self.statusBar().showMessage(f"Error: OMPL file not found at: {ompl_path}")
                        return
                    self.P_list = load_ompl(ompl_path)
                    self.last_loaded_ompl_path = ompl_path
                    self.trajectory_dirty = False
                # Clear any stale preview whenever the trajectory changes
                self.P_list_preview = None
                if self._param_win is not None:
                    self._param_win._initialized_obj = None
                    self._param_win._sampled_P_views = None
                self.update_trajectory_coordinates()
            except Exception as e:
                self.statusBar().showMessage(f"Error loading OMPL: {str(e)}")
                return
                
        self.voxel_dimensions = config.get("voxel_dimensions", [100, 100, 100])
        self.model_matrix = np.array(config.get("model_matrix", np.eye(4).tolist()))
        self.unit = config.get("unit", "mm")
        
        if trajectory_changed:
            num_views = len(self.P_list)
            mean_sid = 0.0
            mean_sdd = 0.0
            size_str = "N/A"
            spacing_str = "N/A"
            sid_str = "N/A"
            sdd_str = "N/A"
            plane_str = "N/A"
            
            if num_views > 0:
                first_p = self.P_list[0]
                size_str = f"{first_p.image_size[0]} x {first_p.image_size[1]} px"
                spacing_str = f"{first_p.pixel_spacing:.3f} {self.unit}"
                
                try:
                    # Cache homogeneous source positions to avoid SVD calls during rendering
                    self.source_positions_hom = [p.getCenterOfProjection().reshape(-1, 1) for p in self.P_list]
                    sources = np.array([C.flatten()[:3] for C in self.source_positions_hom])
                    # Calculate isocenter as the point closest to all backprojection rays of the image center
                    self.isocenter = self.estimate_isocenter()
                    
                    # Check if isocenter really is between source and detector for the first projection.
                    # If not, negate the pixel spacing in the OMPL file and reload it.
                    P_first = self.P_list[0]
                    sdg_first = SourceDetectorGeometry(P_first)
                    s_first = sdg_first.source_position.flatten()[:3]
                    O_first = sdg_first.detector_origin.flatten()[:3]
                    U_first = sdg_first.axis_direction_Upx.flatten()[:3] * P_first.image_size[0]
                    V_first = sdg_first.axis_direction_Vpx.flatten()[:3] * P_first.image_size[1]
                    d_first = O_first + 0.5 * U_first + 0.5 * V_first
                    
                    v_sd = d_first - s_first
                    v_si = self.isocenter - s_first
                    
                    dot_sd_sd = np.dot(v_sd, v_sd)
                    is_between = False
                    if dot_sd_sd > 1e-12:
                        t = np.dot(v_si, v_sd) / dot_sd_sd
                        if 0.0 < t < 1.0:
                            is_between = True
                    
                    if not is_between:
                        for p in self.P_list:
                            p.pixel_spacing = -p.pixel_spacing
                        
                        try:
                            from fileformats import save_ompl
                        except ImportError:
                            from reconstruct.fileformats import save_ompl
                        
                        try:
                            save_ompl(self.P_list, ompl_path)
                            self.P_list = load_ompl(ompl_path)
                            self.source_positions_hom = [p.getCenterOfProjection().reshape(-1, 1) for p in self.P_list]
                            sources = np.array([C.flatten()[:3] for C in self.source_positions_hom])
                            self.isocenter = self.estimate_isocenter()
                            
                            P_first = self.P_list[0]
                            sdg_first = SourceDetectorGeometry(P_first)
                            s_first = sdg_first.source_position.flatten()[:3]
                            O_first = sdg_first.detector_origin.flatten()[:3]
                            U_first = sdg_first.axis_direction_Upx.flatten()[:3] * P_first.image_size[0]
                            V_first = sdg_first.axis_direction_Vpx.flatten()[:3] * P_first.image_size[1]
                            d_first = O_first + 0.5 * U_first + 0.5 * V_first
                            
                            self.update_trajectory_coordinates()
                            self.statusBar().showMessage("Negated pixel spacing in OMPL due to incorrect detector side (left-handed fix).")
                        except Exception as e:
                            self.statusBar().showMessage(f"Warning: Failed to save/reload updated OMPL: {str(e)}")
                    
                    mean_sid = np.mean([np.linalg.norm(src - self.isocenter) for src in sources])
                    mean_sdd = np.mean([abs(SourceDetectorGeometry(p).source_detector_distance) for p in self.P_list])
                    
                    sid_str = f"{mean_sid:.2f} {self.unit}"
                    sdd_str = f"{mean_sdd:.2f} {self.unit}"
                    
                    if num_views >= 3:
                        centered = sources - np.mean(sources, axis=0)
                        _, _, Vt = np.linalg.svd(centered)
                        self.rotation_axis = Vt[-1, :]
                        if (self.rotation_axis[2] < 0 or 
                            (abs(self.rotation_axis[2]) < 1e-7 and self.rotation_axis[1] < 0) or
                            (abs(self.rotation_axis[2]) < 1e-7 and abs(self.rotation_axis[1]) < 1e-7 and self.rotation_axis[0] < 0)):
                            self.rotation_axis = -self.rotation_axis
                        d = -np.dot(self.rotation_axis, self.isocenter)
                        plane_str = f"[{self.rotation_axis[0]:.4f}, {self.rotation_axis[1]:.4f}, {self.rotation_axis[2]:.4f}, {d:.4f}]"
                        
                        # Check if the secondary angle of the first projection is > 20 degrees
                        r = first_p.getPrincipalRay().flatten()
                        a = self.rotation_axis
                        norm_r = np.linalg.norm(r)
                        norm_a = np.linalg.norm(a)
                        if norm_r > 1e-12 and norm_a > 1e-12:
                            cos_theta = np.dot(r, a) / (norm_r * norm_a)
                            theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                            secondary_angle = abs(90.0 - theta_deg)
                        else:
                            secondary_angle = 0.0
                            
                        if secondary_angle > 20.0:
                            if hasattr(self, 'combo_view_follows'):
                                self.combo_view_follows.blockSignals(True)
                                self.combo_view_follows.setCurrentText("Laminography")
                                self.combo_view_follows.blockSignals(False)
                                self.on_view_follows_changed()
                    else:
                        self.rotation_axis = np.array([0.0, 0.0, 1.0])
                        plane_str = "N/A (needs >= 3 views)"
                except Exception as ex:
                    self.isocenter = np.array([0.0, 0.0, 0.0])
                    self.rotation_axis = np.array([0.0, 0.0, 1.0])
                    self.source_positions_hom = []
                    sid_str = "Error"
                    sdd_str = "Error"
                    plane_str = f"Error: {str(ex)}"
                    mean_sid = 0.0
                    mean_sdd = 0.0
            else:
                self.isocenter = np.array([0.0, 0.0, 0.0])
                self.rotation_axis = np.array([0.0, 0.0, 1.0])
                self.source_positions_hom = []
                mean_sdd = 0.0
                
            self.mean_sid = mean_sid
            self.mean_sdd = mean_sdd
            # Compute T_align to align rotation axis with world Z-axis and center isocenter
            try:
                a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                v = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, v)) > 0.9:
                    v = np.array([0.0, 1.0, 0.0])
                r1 = np.cross(v, a)
                r1 = r1 / np.linalg.norm(r1)
                r2 = np.cross(a, r1)
                r2 = r2 / np.linalg.norm(r2)
                R = np.vstack([r1, r2, a])
                
                self.T_align = np.eye(4)
                self.T_align[:3, :3] = R
                self.T_align[:3, 3] = -R @ self.isocenter
            except Exception as ex:
                self.T_align = np.eye(4)
                
            self.on_view_follows_changed()
            self.viewport.default_s = self.viewport.s
            
            # Precompute and cache reconstruction volumes using 18 equally spaced views
            self.cached_volumes = None
            self.john_quadric = None
            if num_views > 0:
                try:
                    # Sub-sample trajectory to 18 roughly equally spaced views
                    n_proj = len(self.P_list)
                    if n_proj > 18:
                        indices = np.round(np.linspace(0, n_proj - 1, 18)).astype(int)
                        p_list_sub = [self.P_list[i] for i in indices]
                    else:
                        p_list_sub = self.P_list
                    
                    print(f"Using 18 roughly equally spaced projections for volume estimation.", flush=True)
                    
                    raw_matrices = [p.P for p in p_list_sub]
                    first_p = self.P_list[0]
                    
                    from recon_coverage import ReconstructionVolumeEstimator
                    estimator = ReconstructionVolumeEstimator(
                        raw_matrices,
                        detector_size=first_p.image_size,
                        pixel_spacing=first_p.pixel_spacing,
                        sid=self.mean_sid,
                        sdd=self.mean_sdd
                    )
                    self.cached_volumes = estimator.estimate()
                    
                    # Convert John's ellipsoid to quadric Q (symmetric 4x4 matrix)
                    ellipsoid = self.cached_volumes.ellipsoid
                    B_val = ellipsoid[:3, :3]
                    c_val = ellipsoid[:3, 3]
                    Sigma_inv = np.linalg.inv(B_val @ B_val.T)
                    Q_mat = np.zeros((4, 4))
                    Q_mat[:3, :3] = Sigma_inv
                    Q_mat[:3, 3] = -Sigma_inv @ c_val
                    Q_mat[3, :3] = -c_val.T @ Sigma_inv
                    Q_mat[3, 3] = c_val.T @ Sigma_inv @ c_val - 1.0
                    self.john_quadric = Q_mat
                except Exception as ex:
                    print(f"Warning: Failed to compute John's ellipsoid: {ex}", flush=True)
                    
            info_text = (
                f"Number of views: {num_views}\n"
                f"Detector size: {size_str}\n"
                f"Pixel spacing: {spacing_str}\n"
                f"Mean SID: {sid_str}\n"
                f"Mean SDD: {sdd_str}\n"
                f"Isocenter:\n  [{self.isocenter[0]:.2f}, {self.isocenter[1]:.2f}, {self.isocenter[2]:.2f}]\n"
                f"Plane of rotation:\n  {plane_str}"
            )
            self.txt_file_info.setText(info_text)
            
            # Update slider/spinbox limits
            self.active_view_slider.setRange(0, max(0, num_views - 1))
            self.active_view_spin.setRange(0, max(0, num_views - 1))
            self.active_view_slider.setValue(0)
            self.active_view_spin.setValue(0)
            
        self.render_viewport()
        self.statusBar().showMessage("Geometry configuration loaded and viewport updated.")
        # Keep the floating volume window in sync if it is open
        if self._volume_win is not None and self._volume_win.isVisible():
            self._volume_win.refresh_from_config()

    def get_source_plane_basis(self):
        a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
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
        H[:3, 2] = self.isocenter
        H[3, 2] = 1.0
        return H


    def get_active_geometry(self, active_idx):
        if not self.P_list or active_idx >= len(self.P_list):
            return None
            
        P_active = self.P_list[active_idx]
        sdg = SourceDetectorGeometry(P_active)
        
        C_src = sdg.source_position.flatten()[:3]
        O_det = sdg.detector_origin.flatten()[:3]
        U_det = sdg.axis_direction_Upx.flatten()[:3] * P_active.image_size[0]
        V_det = sdg.axis_direction_Vpx.flatten()[:3] * P_active.image_size[1]
        C_det = O_det + 0.5 * U_det + 0.5 * V_det
        C_mid = 0.5 * (C_src + C_det)
        
        # Primary angle (rotation around axis relative to view 0)
        primary_angle = 0.0
        if len(self.source_positions_hom) > 0:
            C_0 = self.source_positions_hom[0].flatten()[:3]
            a = self.rotation_axis.copy()
            norm_a = np.linalg.norm(a)
            if norm_a > 1e-12:
                a = a / norm_a
            else:
                a = np.array([0.0, 0.0, 1.0])
                
            v_0 = C_0 - self.isocenter
            v_act = C_src - self.isocenter
            v_0_proj = v_0 - np.dot(v_0, a) * a
            v_act_proj = v_act - np.dot(v_act, a) * a
            n_0 = np.linalg.norm(v_0_proj)
            n_act = np.linalg.norm(v_act_proj)
            if n_0 > 1e-6 and n_act > 1e-6:
                u_0 = v_0_proj / n_0
                u_act = v_act_proj / n_act
                cos_phi = np.clip(np.dot(u_0, u_act), -1.0, 1.0)
                sin_phi = np.dot(np.cross(u_0, u_act), a)
                primary_angle = np.arctan2(sin_phi, cos_phi)
                
        return {
            "P_active": P_active,
            "source_position": C_src,
            "detector_center": C_det,
            "midpoint": C_mid,
            "primary_angle": primary_angle,
            "principal_ray": P_active.getPrincipalRay().flatten()
        }

    def render_viewport(self):
        if self.viewport.P_display is None:
            return
        try:
            w_w, w_h = self.viewport.width(), self.viewport.height()
            active_idx = self.active_view_slider.value() if self.P_list else 0

            # 1. Precompute parameters
            geom = self.get_active_geometry(active_idx)

            # 2. Extract elevation (ax) and azimuth (az) from R_view for display
            try:
                ax_val = np.arctan2(-self.viewport.R_view[1, 2], self.viewport.R_view[2, 2])
                az_val = np.arctan2(-self.viewport.R_view[0, 1], self.viewport.R_view[0, 0])
            except:
                ax_val = 0.0
                az_val = 0.0

            # 3. Assemble viewing matrix
            T_rot = np.eye(4)
            T_rot[:3, :3] = self.viewport.R_view
            T_view = scale(self.viewport.s) @ T_rot

            # 4. Determine T_display based on View Mode
            view_mode = "Turntable"
            if hasattr(self, 'combo_view_follows'):
                view_mode = self.combo_view_follows.currentText()

            T_display = np.eye(4)
            if geom is None:
                pass
            else:
                # Precompute detector orbit diameter and scale factor for all modes
                det_centers = []
                for pm, sdg in zip(self.P_list, self.get_sdg_list(self.P_list)):
                    O_det = sdg.detector_origin.flatten()[:3]
                    U_det = sdg.axis_direction_Upx.flatten()[:3] * pm.image_size[0]
                    V_det = sdg.axis_direction_Vpx.flatten()[:3] * pm.image_size[1]
                    C_det = O_det + 0.5 * U_det + 0.5 * V_det
                    det_centers.append(C_det)
                
                a = self.rotation_axis.copy()
                norm_a = np.linalg.norm(a)
                if norm_a > 1e-6:
                    a = a / norm_a
                else:
                    a = np.array([0.0, 0.0, 1.0])
                
                distances = []
                for c_det in det_centers:
                    d = c_det - self.isocenter
                    d_orth = d - np.dot(d, a) * a
                    distances.append(np.linalg.norm(d_orth))
                
                r_mean = np.mean(distances) if distances else 75.0
                D = 2.0 * r_mean
                s_scale = 250.0 / D if D > 1e-6 else 1.0
                
                # A. Rotation and Look-at target selection
                if view_mode == "Turntable":
                    # Align to detector axes
                    r = geom["principal_ray"]
                    u_x = r / np.linalg.norm(r)
                    u_y = np.cross(a, u_x)
                    norm_uy = np.linalg.norm(u_y)
                    if norm_uy < 1e-6:
                        if abs(u_x[0]) > 0.9:
                            u_y = np.array([0.0, 1.0, 0.0])
                        else:
                            u_y = np.array([1.0, 0.0, 0.0])
                        u_y = np.cross(u_x, u_y)
                        u_y = u_y / np.linalg.norm(u_y)
                    else:
                        u_y = u_y / norm_uy
                    u_z = np.cross(u_x, u_y)
                    
                    R_align = np.vstack([u_x, u_y, u_z])
                    T_display[:3, :3] = s_scale * R_align
                    T_display[:3, 3] = -s_scale * (R_align @ self.isocenter)
                    
                elif view_mode == "Laminography":
                    # Z-axis aligns with the rotation axis
                    u_z = a.copy()
                    # X-axis is the cross product of (1, 0, 0) and the rotation axis
                    u_x = np.cross(np.array([1.0, 0.0, 0.0]), a)
                    norm_ux = np.linalg.norm(u_x)
                    if norm_ux > 1e-6:
                        u_x = u_x / norm_ux
                    else:
                        # Fallback if rotation axis is parallel to (1, 0, 0)
                        u_x = np.cross(np.array([0.0, 1.0, 0.0]), a)
                        norm_ux = np.linalg.norm(u_x)
                        if norm_ux > 1e-6:
                            u_x = u_x / norm_ux
                        else:
                            u_x = np.array([1.0, 0.0, 0.0])
                    # Y-axis is given by orthonormality (cross product of z and x)
                    u_y = np.cross(u_z, u_x)
                    norm_uy = np.linalg.norm(u_y)
                    if norm_uy > 1e-6:
                        u_y = u_y / norm_uy
                    
                    R_align = np.vstack([u_x, u_y, u_z])
                    
                    # Rotate by -primary_angle about the rotation axis in world coordinates (Rodrigues' rotation formula)
                    phi = geom["primary_angle"]
                    theta = -phi
                    K = np.array([
                        [0.0, -a[2], a[1]],
                        [a[2], 0.0, -a[0]],
                        [-a[1], a[0], 0.0]
                    ])
                    R_world_rot = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
                    
                    R_lami = R_align @ R_world_rot
                    
                    # Project current detector center to the Plücker line orthogonally and use as look-at target
                    import ProjectiveGeometry23.pluecker as pluecker
                    L_rot = pluecker.join_points(pgu.homogenize(self.isocenter), pgu.infinite(self.rotation_axis))
                    C_det = geom["detector_center"]
                    pt_homogeneous = pluecker.closest_to_point(L_rot, pgu.homogenize(C_det))
                    pt_3d = (pt_homogeneous[:3] / pt_homogeneous[3]).flatten()
                    
                    # Apply rotation and translation with scaling
                    T_display[:3, :3] = s_scale * R_lami
                    T_display[:3, 3] = -s_scale * (R_lami @ pt_3d)
                    
                else: # Gantry
                    up_axis = "Z"
                    if hasattr(self, 'combo_up_axis') and self.combo_up_axis.isEnabled():
                        up_axis = self.combo_up_axis.currentText()
                    
                    if up_axis == "X":
                        R_up = np.array([
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]
                        ])
                    elif up_axis == "Y":
                        R_up = np.array([
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]
                        ])
                    else: # Z
                        R_up = np.eye(3)
                        
                    T_display[:3, :3] = s_scale * R_up
                    T_display[:3, 3] = -s_scale * (R_up @ self.isocenter)

            P_view = self.viewport.H_translation @ self.viewport.P_display.P @ T_view @ T_display

            # 5. Build and Render
            svg_obj = self._build_scene_svg(w_w, w_h, P=P_view)
            self._add_dynamic_elements(
                svg_obj,
                P_view=P_view,
                active_idx=active_idx,
                show_pyramid=True,
                show_current_source=True
            )
            raw_svg = svg_obj.render(P=P_view)
            fixed_svg = self._fix_svg_alpha(raw_svg)

            self.last_rendered_svg = fixed_svg
            self.viewport.load(fixed_svg.encode('utf-8'))

            if self.P_list and geom is not None:
                C_active = geom["source_position"]
                ray_active = geom["principal_ray"]
                P_active = geom["P_active"]
                p_fmt = "\n".join(" ".join(f"{val:7.2f}" for val in row) for row in P_active.P)
                unit = getattr(self, 'unit', 'mm')
                details_text = (
                    f"Source Position:\n  [{C_active[0]:.2f}, {C_active[1]:.2f}, {C_active[2]:.2f}]\n"
                    f"Principal Ray:\n  [{ray_active[0]:.2f}, {ray_active[1]:.2f}, {ray_active[2]:.2f}]\n"
                    f"Projection Matrix [{unit}] -> detector pixels:\n{p_fmt}"
                )
                self.txt_active_details.setText(details_text)

                self.statusBar().showMessage(
                    f"Active Camera Center: [{C_active[0]:.1f}, {C_active[1]:.1f}, "
                    f"{C_active[2]:.1f}] | "
                    f"Elevation (ax): {ax_val:.2f} rad | Azimuth (az): {az_val:.2f} rad | "
                    f"Zoom: {self.viewport.s:.3f}"
                )
            else:
                self.txt_active_details.setText("")

            # Refresh any open projection view windows
            for pw in list(self._projection_windows):
                pw.update_slider_range()
                pw.refresh()

        except Exception as e:
            self.statusBar().showMessage(f"Error during viewport rendering: {str(e)}")

    def get_resolved_image_paths(self):
        json_text = self.editor.toPlainText().strip()
        if not json_text:
            return []
        try:
            config = json.loads(json_text)
        except Exception:
            return []
            
        image_files = config.get("image_files", [])
        if not image_files:
            return []
            
        data_dir = config.get("data_dir", "")
        config_dir = os.path.dirname(os.path.abspath(self.current_config_path)) if self.current_config_path else os.getcwd()
        resolved_data_dir = data_dir
        if resolved_data_dir and not os.path.isabs(resolved_data_dir):
            resolved_data_dir = os.path.normpath(os.path.join(config_dir, resolved_data_dir))
        elif not resolved_data_dir:
            resolved_data_dir = config_dir
            
        resolved_image_paths = []
        for p in image_files:
            if os.path.isabs(p):
                resolved_image_paths.append(p)
            else:
                resolved_image_paths.append(os.path.normpath(os.path.join(resolved_data_dir, p)))
        return resolved_image_paths

    @staticmethod
    def _fix_svg_alpha(raw_svg):
        """Convert specific 8-digit hex colors to 6-digit + opacity attributes using fast string replace."""
        return raw_svg

    def _build_scene_svg(self, w_w, w_h, P,
                         active_idx=0,
                         show_pyramid=True, show_current_source=True,
                         preview_pil_image=None):
        """Build and return a Composer containing the static/shared scene elements in world coordinates."""
        svg = Composer((w_w, w_h))
        svg.add(e2d.rect, x=0, y=0, width=w_w, height=w_h, fill="#ffffff")

        if preview_pil_image is not None:
            svg.add(e2d.image, data=preview_pil_image, x=0, y=0, width=w_w, height=w_h)

        # 1. World coordinate frame
        try:
            mean_sid_val = getattr(self, 'mean_sid', 0.0)
            half_sid = mean_sid_val / 2.0 if mean_sid_val > 0.0 else 80.0
            k = int(np.round(np.log10(half_sid)))
            axis_length = 10.0 ** k
        except:
            axis_length = 10.0
        unit = getattr(self, 'unit', 'mm')
        space = " " if len(unit) > 2 else ""
        len_str = (f"{int(axis_length)}{space}{unit}" if axis_length >= 1.0
                   else f"{axis_length}{space}{unit}")
        
        # Draw labels only if Axis labels checkbox is checked
        show_axis_labels = getattr(self, 'chk_axis_labels', None) is None or self.chk_axis_labels.isChecked()
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[axis_length,0,0,1], stroke='red', stroke_width=2)
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[0,axis_length,0,1], stroke='green', stroke_width=2)
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[0,0,axis_length,1], stroke='blue', stroke_width=2)
        if show_axis_labels:
            svg.add(safe_text, X=[0,0,0,1], content="world", fill='black', font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[axis_length*1.1,0,0,1], content=f"X ({len_str})",
                    fill='red',   font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[0,axis_length*1.1,0,1], content=f"Y ({len_str})",
                    fill='green', font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[0,0,axis_length*1.1,1], content=f"Z ({len_str})",
                    fill='blue',  font_size='11px', font_family='sans-serif')

        # 1b. Voxel coordinate frame
        if self.voxel_dimensions is not None:
            try:
                mean_dim = np.mean(self.voxel_dimensions)
                p = 1
                while p < mean_dim:
                    p *= 10
                p = max(p // 10, 1)
                if p * 5 < mean_dim:
                    p = p * 5
                elif p * 2 < mean_dim:
                    p = p * 2
                
                M = self.model_matrix
                # Voxel origin in world space
                vox_orig = M @ np.array([0, 0, 0, 1])
                # Voxel axes endpoints in world space
                vox_x = M @ np.array([p, 0, 0, 1])
                vox_y = M @ np.array([0, p, 0, 1])
                vox_z = M @ np.array([0, 0, p, 1])
                
                # Text positions (slightly beyond the arrow tip)
                txt_x = M @ np.array([p * 1.1, 0, 0, 1])
                txt_y = M @ np.array([0, p * 1.1, 0, 1])
                txt_z = M @ np.array([0, 0, p * 1.1, 1])
                
                svg.add(safe_arrow, X1=vox_orig, X2=vox_x, stroke='red', stroke_width=2)
                svg.add(safe_arrow, X1=vox_orig, X2=vox_y, stroke='green', stroke_width=2)
                svg.add(safe_arrow, X1=vox_orig, X2=vox_z, stroke='blue', stroke_width=2)
                if show_axis_labels:
                    svg.add(safe_text, X=vox_orig, content="volume", fill='black', font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_x, content=f"X ({p} voxels)",
                            fill='red',   font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_y, content=f"Y ({p} voxels)",
                            fill='green', font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_z, content=f"Z ({p} voxels)",
                            fill='blue',  font_size='11px', font_family='sans-serif')
            except Exception as e:
                print(f"Warning: Failed to render voxel coordinate system: {str(e)}")

        # 2. Rotation axis (dashed yellow Plücker line)
        import ProjectiveGeometry23.pluecker as pluecker
        from ProjectiveGeometry23.svg_utils import svg_pluecker_line
        import ProjectiveGeometry23.utils as pgu
        L_rot = pluecker.join_points(pgu.homogenize(self.isocenter), pgu.infinite(self.rotation_axis))
        svg.add(svg_pluecker_line, L=L_rot, stroke="yellow", stroke_dasharray="4,4", stroke_width=1.5)

        # 3. Detector center orbit trajectory lines (static/shared)
        show_trajectory = getattr(self, 'chk_trajectory', None) is None or self.chk_trajectory.isChecked()
        if self.P_list and show_trajectory:
            disp_det = getattr(self, 'cached_disp_det', [])
            num_views = len(disp_det)
            if num_views > 180:
                all_idx = np.round(np.linspace(0, num_views - 1, 180)).astype(int)
                for i in range(len(all_idx) - 1):
                    a, b = all_idx[i], all_idx[i + 1]
                    if a < num_views and b < num_views:
                        svg.add(safe_line, X1=disp_det[a], X2=disp_det[b], stroke="black", stroke_width=1.2)
            else:
                for i in range(num_views - 1):
                    svg.add(safe_line, X1=disp_det[i], X2=disp_det[i + 1], stroke="black", stroke_width=1.2)

        # Status overlay
        svg.add(e2d.text, x=15, y=25,
                content=f"Views: {len(self.P_list)}",
                fill="#333333", font_size="13px", font_family="monospace")

        return svg

    def _add_dynamic_elements(self, svg_obj, P_view, active_idx, show_pyramid, show_current_source):
        """Add dynamic elements (reconstruction volume, active camera frustum, split source trajectory,
        labels, and arrows) to the composer, evaluating group objects immediately to fix transparency locally."""
        if not self.P_list:
            svg_obj.add(e2d.text, x=15, y=50,
                        content="No Trajectory Loaded",
                        fill="#d00000", font_size="13px", font_family="monospace")
            return

        if active_idx >= len(self.P_list):
            active_idx = 0

        disp_src = getattr(self, 'cached_disp_src', [])
        disp_det = getattr(self, 'cached_disp_det', [])
        num_views = len(disp_src)

        # 1. Reconstruction volume (using patched_volume + transparency fix)
        if self.voxel_dimensions is not None:
            shape = (self.voxel_dimensions[2], self.voxel_dimensions[1], self.voxel_dimensions[0])
            svg_obj.add(
                patched_volume,
                shape=shape,
                model_matrix=self.model_matrix,
                color_axes=False,
                lighting=True,
                fill="#00ff4015",
                stroke="#00ff4080",
                stroke_width=1.0,
            )

        # 1b. Region of Interest (ROI) rendering based on self.combo_roi dropdown
        roi_mode = getattr(self, 'combo_roi', None)
        roi_text = roi_mode.currentText() if roi_mode is not None else "Hidden"
        
        if roi_text == "John's Quadric" and getattr(self, 'john_quadric', None) is not None:
            svg_obj.add(
                draw_ellipsoid,
                Q=self.john_quadric,
                stroke="orange",
                stroke_width=2.0,
                fill="none",
            )
            # Intersection of John's quadric with the source plane (only in 3D views)
            try:
                H_plane = self.get_source_plane_basis()
                svg_obj.add(
                    draw_quadric_plane_intersection,
                    Q=self.john_quadric,
                    H_plane=H_plane,
                    stroke="orange",
                    stroke_width=2.0,
                    fill="none",
                )
            except Exception as e:
                print(f"Warning: Failed to render quadric-plane intersection: {e}", flush=True)
                
        elif roi_text == "Circumscribed Cylinder" and getattr(self, 'cached_volumes', None) is not None:
            try:
                cyl = self.cached_volumes.cylinder_cc
                
                # Fetch scale factor from UI/config
                scale_factor = 1.0
                try:
                    val = self.spin_mask_cylinder.value()
                    # If 0.0 is used (meaning off/disabled), draw it at standard size (scale_factor=1.0)
                    if val > 0.0:
                        scale_factor = val
                except Exception:
                    pass

                p0 = np.array(cyl.point0)
                p1 = np.array(cyl.point1)
                radius = float(cyl.radius)

                # Scale from isocenter
                isocenter = getattr(self, 'isocenter', np.array([0.0, 0.0, 0.0]))
                p0 = isocenter + scale_factor * (p0 - isocenter)
                p1 = isocenter + scale_factor * (p1 - isocenter)
                radius = scale_factor * radius

                svg_obj.add(
                    draw_cylinder,
                    point0=p0,
                    point1=p1,
                    radius=radius,
                    stroke="orange",
                    stroke_width=2.0,
                    fill="none",
                )
            except Exception as e:
                print(f"Warning: Failed to render cylinder ROI: {e}", flush=True)

        # 2. Source orbit points and active source point
        show_traj = getattr(self, 'chk_trajectory', None) is None or self.chk_trajectory.isChecked()
        if num_views > 0 and show_traj:
            svg_obj.add(
                patched_trajectory,
                disp_src=disp_src,
                active_idx=active_idx,
                num_views=num_views
            )

            # Draw active source point (red dot) if requested and not in the null space
            if show_current_source and active_idx < num_views:
                svg_obj.add(safe_point, X=disp_src[active_idx], r=6, fill="#ff2d55")

        # 3. Angle labels and source-to-detector center arrows (for subset of indices)
        if show_traj:
            closest_indices = {}
            try:
                a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                v = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, v)) > 0.9:
                    v = np.array([0.0, 0.0, 1.0])
                r3 = np.cross(a, v)
                r3 /= np.linalg.norm(r3)
                r1 = np.cross(a, r3)
                r1 /= np.linalg.norm(r1)
                angles_deg = []
                for C_hom in self.source_positions_hom:
                    C = C_hom.flatten()[:3] - self.isocenter
                    angles_deg.append(np.arctan2(np.dot(C, r3), np.dot(C, r1)) * 180.0 / np.pi)
                ref_angle = angles_deg[0]
                angles_deg = np.array([(d - ref_angle) % 360.0 for d in angles_deg])
                for target in [45, 90, 135, 180, 225, 270, 315]:
                    idx = int(np.argmin(np.abs(angles_deg - target)))
                    closest_indices[idx] = angles_deg[idx]
            except:
                pass

            for idx, actual_angle in closest_indices.items():
                if idx != active_idx and idx < num_views:
                    C = disp_src[idx]
                    val = round(actual_angle, 1)
                    label_str = f".  {idx} ({int(val) if val == int(val) else val}°)"
                    if num_views > 180:
                        svg_obj.add(safe_point, X=C, r=3, fill="#00adb5")
                    if getattr(self, 'chk_axis_labels', None) is None or self.chk_axis_labels.isChecked():
                        svg_obj.add(safe_text, X=C, content=label_str, fill="#00adb5", font_size="10px", font_family="monospace")
                    svg_obj.add(safe_line, X1=disp_src[idx], X2=disp_det[idx], stroke="#000000", stroke_opacity=0.251)

        # 4. Active view pyramid (source-detector geometry)
        if show_pyramid and active_idx < len(self.P_list):
            # Always get P_active from the full P_list to match active_idx correctly
            P_active = self.P_list[active_idx]
            
            # If Virtual (in Iso-Center) mode is active, modify pixel spacing of P_active
            is_virt = getattr(self, 'combo_det_vis', None) is not None and self.combo_det_vis.currentText() == "Virtual (Iso-Center)"
            if is_virt:
                sdg_temp = SourceDetectorGeometry(P_active)
                C_pos = sdg_temp.source_position.flatten()[:3]
                sid_val = np.linalg.norm(C_pos - self.isocenter)
                sdd_val = abs(sdg_temp.source_detector_distance)
                if sdd_val > 1e-5:
                    f_scale = sid_val / sdd_val
                    from ProjectiveGeometry23.central_projection import ProjectionMatrix
                    P_active = ProjectionMatrix(P_active.P, P_active.image_size, P_active.pixel_spacing * f_scale)

            show_lbls = getattr(self, 'chk_axis_labels', None) is None or self.chk_axis_labels.isChecked()
            if self.voxel_dimensions is not None:
                def draw_voxel_wireframe(P, **kwargs):
                    X, Y, Z = self.voxel_dimensions
                    P_local = P @ self.model_matrix
                    group = Group("Voxel Wireframe and Axes")
                    group.add(e3d.wire_cube, P=P_local, min=[0,0,0], max=[X,Y,Z],
                              stroke="#ff2d55", stroke_width=1.0)
                    
                    mean_dim = np.mean(self.voxel_dimensions)
                    p = 1
                    while p < mean_dim:
                        p *= 10
                    p = max(p // 10, 1)
                    if p * 5 < mean_dim:
                        p = p * 5
                    elif p * 2 < mean_dim:
                        p = p * 2
                    
                    vox_orig = np.array([0, 0, 0, 1])
                    vox_x = np.array([p, 0, 0, 1])
                    vox_y = np.array([0, p, 0, 1])
                    vox_z = np.array([0, 0, p, 1])
                    
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_x, stroke='red', stroke_width=2)
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_y, stroke='green', stroke_width=2)
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_z, stroke='blue', stroke_width=2)
                    
                    # Project and draw John's quadric (if available and checked)
                    roi_mode = getattr(self, 'combo_roi', None)
                    roi_text = roi_mode.currentText() if roi_mode is not None else "Hidden"
                    if getattr(self, 'john_quadric', None) is not None and roi_text == "John's Quadric":
                        group.add(
                            draw_ellipsoid,
                            P=P,
                            Q=self.john_quadric,
                            stroke="orange",
                            stroke_width=2.0,
                            fill="none",
                        )
                    return group
                svg_obj.add(svg_source_detector, projection=P_active,
                            draw_on_detector=draw_voxel_wireframe,
                            label_source=f".  C{active_idx}",
                            is_virtual=is_virt,
                            show_axis_labels=show_lbls)
            else:
                svg_obj.add(svg_source_detector, projection=P_active,
                            draw_on_detector=None, label_source=f".  C{active_idx}",
                            is_virtual=is_virt,
                            show_axis_labels=show_lbls)



class StartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconstruction GUI - Startup")
        self.setFixedSize(420, 200)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(25, 20, 25, 20)
        
        lbl_title = QLabel("Select Startup Option")
        lbl_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)
        
        self.btn_open = QPushButton("Open reconstruction.json ...")
        self.btn_open.setFont(QFont("Segoe UI", 10))
        self.btn_define = QPushButton("Define reconstruction job ...")
        self.btn_define.setFont(QFont("Segoe UI", 10))
        self.btn_example = QPushButton("Load example data")
        self.btn_example.setFont(QFont("Segoe UI", 10))
        
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_define)
        layout.addWidget(self.btn_example)
        
        self.btn_open.clicked.connect(self.on_open)
        self.btn_define.clicked.connect(self.on_define)
        self.btn_example.clicked.connect(self.on_example)
        
        self.result_action = None
        self.selected_path = None
        
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self.selected_path = path
            self.result_action = 'open'
            self.accept()
            
    def on_define(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save New Config JSON", "reconstruction.json", "JSON Files (*.json)"
        )
        if path:
            self.selected_path = path
            self.result_action = 'define'
            self.accept()
            
    def on_example(self):
        self.result_action = 'example'
        self.accept()


class DefineJobDialog(QDialog):
    def __init__(self, save_json_path, parent=None):
        super().__init__(parent)
        self.save_json_path = save_json_path
        self.setWindowTitle("Define Reconstruction Job")
        self.resize(550, 620)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        from PyQt6.QtWidgets import QRadioButton, QGroupBox, QDoubleSpinBox, QComboBox, QGridLayout
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        lbl_title = QLabel("Set Up New Reconstruction Job")
        lbl_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        main_layout.addWidget(lbl_title)
        
        # 1. Images Selection Group
        grp_images = QGroupBox("1. Projection Images")
        images_layout = QVBoxLayout(grp_images)
        
        btn_select_images = QPushButton("Select Images ...")
        btn_select_images.clicked.connect(self.on_select_images)
        images_layout.addWidget(btn_select_images)
        
        self.lbl_images_info = QLabel("No images selected.")
        self.lbl_images_info.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_images_info.setStyleSheet("color: #d00000;")
        images_layout.addWidget(self.lbl_images_info)
        
        main_layout.addWidget(grp_images)
        
        # 2. Trajectory Type Selection
        grp_type = QGroupBox("2. Trajectory Source")
        type_layout = QHBoxLayout(grp_type)
        self.rad_generate = QRadioButton("Generate Circular Cone-Beam")
        self.rad_import = QRadioButton("Import Trajectory File")
        self.rad_generate.setChecked(True)
        type_layout.addWidget(self.rad_generate)
        type_layout.addWidget(self.rad_import)
        
        self.rad_generate.setEnabled(False)
        self.rad_import.setEnabled(False)
        
        self.rad_generate.toggled.connect(self.on_trajectory_type_changed)
        self.rad_import.toggled.connect(self.on_trajectory_type_changed)
        
        main_layout.addWidget(grp_type)
        
        # 3. Define Trajectory Group
        self.group_trajectory = QGroupBox("3. Trajectory Parameters")
        traj_grid = QGridLayout(self.group_trajectory)
        
        self.lbl_load_first = QLabel("Please select images first to define parameters.")
        self.lbl_load_first.setStyleSheet("color: gray; font-style: italic;")
        traj_grid.addWidget(self.lbl_load_first, 0, 0, 1, 4)
        
        # Widgets
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["µm", "mm", "cm", "inch", "vx"])
        self.combo_unit.setCurrentText("mm")
        
        self.combo_rot_axis = QComboBox()
        self.combo_rot_axis.addItems(["Z-axis", "X-axis", "Y-axis"])
        self.combo_rot_axis.setCurrentText("Z-axis")
        
        self.spin_sid = QDoubleSpinBox()
        self.spin_sid.setRange(0.1, 99999.0)
        self.spin_sid.setValue(250.0)
        
        self.spin_sdd = QDoubleSpinBox()
        self.spin_sdd.setRange(0.1, 99999.0)
        self.spin_sdd.setValue(500.0)
        
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.0001, 9999.0)
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setValue(0.2)
        
        self.spin_start_angle = QDoubleSpinBox()
        self.spin_start_angle.setRange(-360.0, 360.0)
        self.spin_start_angle.setValue(0.0)
        
        self.spin_end_angle = QDoubleSpinBox()
        self.spin_end_angle.setRange(-360.0, 360.0)
        self.spin_end_angle.setValue(360.0)
        
        self.spin_shift_lat = QDoubleSpinBox()
        self.spin_shift_lat.setRange(-999.0, 999.0)
        self.spin_shift_lat.setValue(0.0)
        
        self.spin_shift_vert = QDoubleSpinBox()
        self.spin_shift_vert.setRange(-999.0, 999.0)
        self.spin_shift_vert.setValue(0.0)
        
        self.spin_shift_tang = QDoubleSpinBox()
        self.spin_shift_tang.setRange(-999.0, 999.0)
        self.spin_shift_tang.setValue(0.0)
        
        # Grid layout placement
        row = 1
        traj_grid.addWidget(QLabel("World Units:"), row, 0)
        traj_grid.addWidget(self.combo_unit, row, 1)
        traj_grid.addWidget(QLabel("Rotation Axis:"), row, 2)
        traj_grid.addWidget(self.combo_rot_axis, row, 3)
        
        row += 1
        traj_grid.addWidget(QLabel("SID (to Iso):"), row, 0)
        traj_grid.addWidget(self.spin_sid, row, 1)
        traj_grid.addWidget(QLabel("SDD (to Det):"), row, 2)
        traj_grid.addWidget(self.spin_sdd, row, 3)
        
        row += 1
        traj_grid.addWidget(QLabel("Detector Pixel Size:"), row, 0)
        traj_grid.addWidget(self.spin_pixel_size, row, 1)
        self.lbl_detector_size = QLabel("Detector Size: N/A")
        self.lbl_detector_size.setStyleSheet("font-weight: bold; color: blue;")
        traj_grid.addWidget(self.lbl_detector_size, row, 2, 1, 2)
        
        row += 1
        traj_grid.addWidget(QLabel("Rotation:"), row, 0)
        rot_layout = QHBoxLayout()
        rot_layout.addWidget(self.spin_start_angle)
        rot_layout.addWidget(QLabel("° to"))
        rot_layout.addWidget(self.spin_end_angle)
        rot_layout.addWidget(QLabel("°"))
        self.combo_rot_dir = QComboBox()
        self.combo_rot_dir.addItems(["CCW", "Clockwise"])
        self.combo_rot_dir.setCurrentText("CCW")
        rot_layout.addWidget(self.combo_rot_dir)
        traj_grid.addLayout(rot_layout, row, 1, 1, 3)
        
        row += 1
        traj_grid.addWidget(QLabel("Source Shifts (lat, vert, tang):"), row, 0)
        shifts_layout = QHBoxLayout()
        shifts_layout.addWidget(self.spin_shift_lat)
        shifts_layout.addWidget(self.spin_shift_vert)
        shifts_layout.addWidget(self.spin_shift_tang)
        traj_grid.addLayout(shifts_layout, row, 1, 1, 3)
        
        row += 1
        lbl_note = QLabel("Note: Advanced dejusts are supported through 'Parameterizations' in the Trajectory Tab later.")
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color: gray; font-size: 9px; font-style: italic;")
        traj_grid.addWidget(lbl_note, row, 0, 1, 4)
        
        self.trajectory_widgets = [
            self.combo_unit, self.combo_rot_axis, self.spin_sid, self.spin_sdd,
            self.spin_pixel_size, self.spin_start_angle, self.spin_end_angle,
            self.combo_rot_dir,
            self.spin_shift_lat, self.spin_shift_vert, self.spin_shift_tang
        ]
        for widget in self.trajectory_widgets:
            widget.setEnabled(False)
            
        self.group_trajectory.setEnabled(False)
        main_layout.addWidget(self.group_trajectory)
        
        self.spin_pixel_size.valueChanged.connect(self.update_detector_size_label)
        self.combo_unit.currentTextChanged.connect(self.update_detector_size_label)
        
        # 4. Import Trajectory Section
        grp_import = QGroupBox("4. Import Trajectory")
        import_layout = QVBoxLayout(grp_import)
        
        self.btn_import = QPushButton("Import Trajectory...")
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self.on_import_trajectory)
        import_layout.addWidget(self.btn_import)
        
        self.lbl_import_info = QLabel("No trajectory imported.")
        self.lbl_import_info.setStyleSheet("color: gray;")
        import_layout.addWidget(self.lbl_import_info)
        
        main_layout.addWidget(grp_import)
        
        # 5. Buttons area
        btn_box = QHBoxLayout()
        self.lbl_status = QLabel("Select Image please")
        self.lbl_status.setStyleSheet("font-weight: bold; color: darkred;")
        btn_box.addWidget(self.lbl_status)
        btn_box.addStretch()
        
        self.btn_done = QPushButton("Done")
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.on_done)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_box.addWidget(self.btn_done)
        btn_box.addWidget(btn_cancel)
        
        main_layout.addLayout(btn_box)

        self.selected_image_paths = []
        self.imported_P_list = None

    def on_trajectory_type_changed(self):
        is_generate = self.rad_generate.isChecked()
        self.btn_import.setEnabled(not is_generate)
        for widget in self.trajectory_widgets:
            widget.setEnabled(is_generate)
        self.update_done_button_state()

    def update_detector_size_label(self):
        if not hasattr(self, 'image_width') or not hasattr(self, 'image_height'):
            return
        pixel_size = self.spin_pixel_size.value()
        w_mm = pixel_size * self.image_width
        h_mm = pixel_size * self.image_height
        unit = self.combo_unit.currentText()
        self.lbl_detector_size.setText(f"Detector Size: {w_mm:.2f} x {h_mm:.2f} {unit}")

    def on_select_images(self):
        filter_str = (
            "TIFF Images (*.tif *.tiff);;"
            "PIL Supported Types (*.png *.jpg *.jpeg);;"
            "NRRD Files (*.nrrd);;"
            "Raw Sequence (*.seq);;"
            "Processed Image (*.IMA);;"
            "All Files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Projection Images", "", filter_str
        )
        if not paths:
            return
            
        paths = sorted(paths)
        self.selected_image_paths = paths
        
        first_path = paths[0]
        try:
            if first_path.lower().endswith('.nrrd'):
                import nrrd
                img, header = nrrd.read(first_path)
                img = np.squeeze(img)
                if img.ndim == 2:
                    img = img.T
                width, height = img.shape[1], img.shape[0]
            else:
                from PIL import Image
                with Image.open(first_path) as pil_img:
                    width, height = pil_img.size
                    
            self.image_width = width
            self.image_height = height
            self.num_images = len(paths)
            
            self.lbl_images_info.setText(f"Loaded {self.num_images} images ({self.image_width} x {self.image_height} px)")
            self.lbl_images_info.setStyleSheet("color: green;")
            
            # Enable controls
            self.group_trajectory.setEnabled(True)
            self.rad_generate.setEnabled(True)
            self.rad_import.setEnabled(True)
            self.lbl_load_first.hide()
            
            # Update detector size label
            self.update_detector_size_label()
            
            # Trigger state updates
            self.on_trajectory_type_changed()
            
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to read image dimensions:\n{str(e)}")

    def on_import_trajectory(self):
        try:
            from fileformats import discover_formats
        except ImportError:
            try:
                from reconstruct.fileformats import discover_formats
            except ImportError:
                def discover_formats(): return {}, {}
                
        loaders, _ = discover_formats()
        filters = []
        for name, info in loaders.items():
            ext_patterns = " ".join(f"*{ext}" for ext in info["extensions"])
            filters.append(f"{name} ({ext_patterns})")
        filters.append("All Files (*)")
        filter_str = ";;".join(filters)
        
        path, selected_filter = QFileDialog.getOpenFileName(
            self, "Import Trajectory", "", filter_str
        )
        if not path:
            return
            
        ext = os.path.splitext(path)[1].lower()
        
        loader_info = None
        for name, info in loaders.items():
            ext_patterns = " ".join(f"*{e}" for e in info["extensions"])
            filter_text = f"{name} ({ext_patterns})"
            if filter_text == selected_filter:
                loader_info = (name, info)
                break
                
        if not loader_info:
            matching = []
            for name, info in loaders.items():
                if ext in info["extensions"]:
                    matching.append((name, info))
            matching.sort(key=lambda item: 0 if item[0].lower() == "ompl" else 1)
            if matching:
                loader_info = matching[0]
                
        if not loader_info:
            QMessageBox.critical(self, "Import Error", f"Unsupported file extension: {ext}")
            return
            
        name, info = loader_info
        loader_fn = info["fn"]
        
        try:
            pixel_spacing = self.spin_pixel_size.value()
            detector_size = [self.image_width, self.image_height]
            
            if name.lower() == "ompl":
                from fileformats.ompl import load_ompl
                Ps = load_ompl(path)
            else:
                Ps = loader_fn(path, pixel_spacing=pixel_spacing, detector_size_px=detector_size)
                
            if not Ps:
                QMessageBox.warning(self, "Import Error", f"No matrices loaded by {name} loader.")
                return
                
            if len(Ps) != self.num_images:
                QMessageBox.warning(
                    self, 
                    "Trajectory Size Mismatch", 
                    f"The number of imported trajectory matrices ({len(Ps)}) "
                    f"does not match the number of selected images ({self.num_images})."
                )
                return
                
            new_P_list = []
            for P in Ps:
                if hasattr(P, 'image_size'):
                    new_P_list.append(ProjectionMatrix(P.P.copy(), P.image_size.copy(), P.pixel_spacing))
                else:
                    new_P_list.append(ProjectionMatrix(P, detector_size, pixel_spacing))
                    
            self.imported_P_list = new_P_list
            self.lbl_import_info.setText(f"Imported {len(new_P_list)} matrices from {os.path.basename(path)}")
            self.lbl_import_info.setStyleSheet("color: black;")
            self.update_done_button_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import trajectory:\n{str(e)}")

    def update_done_button_state(self):
        if not self.selected_image_paths:
            self.btn_done.setEnabled(False)
            self.lbl_status.setText("Select Image please")
            self.lbl_status.setStyleSheet("font-weight: bold; color: darkred;")
            return
            
        if self.rad_generate.isChecked():
            self.btn_done.setEnabled(True)
            self.lbl_status.setText("Ready to create job.")
            self.lbl_status.setStyleSheet("font-weight: bold; color: green;")
        else: # import selected
            if self.imported_P_list:
                self.btn_done.setEnabled(True)
                self.lbl_status.setText("Ready to create job.")
                self.lbl_status.setStyleSheet("font-weight: bold; color: green;")
            else:
                self.btn_done.setEnabled(False)
                self.lbl_status.setText("Please define or import a trajectory.")
                self.lbl_status.setStyleSheet("font-weight: bold; color: darkred;")

    def on_done(self):
        try:
            config_dir = os.path.dirname(os.path.abspath(self.save_json_path))
            os.makedirs(config_dir, exist_ok=True)
            
            # 1. Save trajectory to trajectory.ompl
            ompl_path = os.path.join(config_dir, "trajectory.ompl")
            
            if self.rad_generate.isChecked():
                n_proj = len(self.selected_image_paths)
                n_u = self.image_width
                n_v = self.image_height
                pixel_size = self.spin_pixel_size.value()
                sid = self.spin_sid.value()
                sdd = self.spin_sdd.value()
                start_angle = self.spin_start_angle.value()
                end_angle = self.spin_end_angle.value()
                rot_dir = self.combo_rot_dir.currentText()
                if rot_dir == "Clockwise":
                    start_angle = -start_angle
                    end_angle = -end_angle
                shift_lat = self.spin_shift_lat.value()
                shift_vert = self.spin_shift_vert.value()
                shift_tang = self.spin_shift_tang.value()
                rot_axis = self.combo_rot_axis.currentText()
                
                def local_camera_look_at(K, eye, center, up=np.array([0.0, 0.0, 1.0])):
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

                f = sdd / pixel_size
                K = np.array([
                    [f, 0.0, n_u * 0.5],
                    [0.0, f, n_v * 0.5],
                    [0.0, 0.0, 1.0]
                ])
                
                Ps = []
                angles = np.linspace(start_angle, end_angle, n_proj, endpoint=False)
                for angle_deg in angles:
                    theta = np.radians(angle_deg)
                    eye_base = np.array([sid * np.cos(theta), sid * np.sin(theta), 0.0])
                    
                    u_radial = np.array([np.cos(theta), np.sin(theta), 0.0])
                    u_tangential = np.array([-np.sin(theta), np.cos(theta), 0.0])
                    u_vertical = np.array([0.0, 0.0, 1.0])
                    
                    eye = eye_base + shift_tang * u_radial + shift_lat * u_tangential + shift_vert * u_vertical
                    
                    P_mat = local_camera_look_at(K, eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
                    P_mat /= np.linalg.norm(P_mat[2, :3])
                    Ps.append(P_mat)
                    
                # Apply global 4x4 homography at the very end and shift principal point to center
                T = np.eye(4)
                if rot_axis == "X-axis":
                    T[0, 0] = 0.0; T[0, 2] = 1.0
                    T[2, 0] = -1.0; T[2, 2] = 0.0
                elif rot_axis == "Y-axis":
                    T[1, 1] = 0.0; T[1, 2] = -1.0
                    T[2, 1] = 1.0; T[2, 2] = 0.0
                    
                final_Ps = [ProjectionMatrix(P @ T, image_size=(n_u, n_v), pixel_spacing=pixel_size) for P in Ps]
            else:
                final_Ps = self.imported_P_list
                pixel_size = final_Ps[0].pixel_spacing
                n_u = final_Ps[0].image_size[0]
                n_v = final_Ps[0].image_size[1]
                
            from fileformats import save_ompl
                
            save_ompl(final_Ps, ompl_path, spacing=pixel_size, detector_size_px=[n_u, n_v])
            
            # 2. Save configuration JSON
            images_dir = os.path.dirname(os.path.abspath(self.selected_image_paths[0]))
            try:
                data_dir_value = os.path.relpath(images_dir, config_dir)
                if data_dir_value.startswith(".." + os.sep + ".."):
                    data_dir_value = images_dir
            except Exception:
                data_dir_value = images_dir
                
            config = {
                "data_dir": data_dir_value,
                "ompl_file": "trajectory.ompl",
                "image_files": [os.path.relpath(p, images_dir) for p in self.selected_image_paths],
                "voxel_dimensions": [128, 128, 128],
                "model_matrix": np.eye(4).tolist(),
                "output_file": "reconstruction.nrrd",
                "convert_to_line_integral": False,
                "filter_type": "ram-lak",
                "unit": self.combo_unit.currentText()
            }
            
            with open(self.save_json_path, 'w') as f:
                f.write(format_config_json(config))
                
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save reconstruction config:\n{str(e)}")


def main():
    import signal
    from PyQt6.QtCore import QTimer
    
    # Restore default behavior for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Start a timer to wake up Python interpreter to process Ctrl+C signals
    sig_timer = QTimer()
    sig_timer.start(500)
    sig_timer.timeout.connect(lambda: None)
    
    edit_config_only = False
    args = sys.argv[1:]
    if "--edit-config-only" in args:
        edit_config_only = True
        args.remove("--edit-config-only")
        
    config_path = args[0] if args else None
    
    suggest_volume_on_start = False
    if not config_path:
        while True:
            startup_dlg = StartupDialog()
            if startup_dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)
                
            if startup_dlg.result_action == 'open':
                config_path = startup_dlg.selected_path
                break
            elif startup_dlg.result_action == 'example':
                default_cfg = None
                try:
                    import ct_recon_fdk_astra as _recon
                    default_cfg = str(_recon.get_data_path(
                        "example_data", "fullscan_180views_600x400.json"
                    ))
                except Exception:
                    pass
                if default_cfg and os.path.exists(default_cfg):
                    config_path = default_cfg
                break
            elif startup_dlg.result_action == 'define':
                save_json_path = startup_dlg.selected_path
                job_dlg = DefineJobDialog(save_json_path)
                if job_dlg.exec() == QDialog.DialogCode.Accepted:
                    config_path = save_json_path
                    suggest_volume_on_start = True
                    break
                else:
                    # User cancelled the define dialog, go back to startup options loop
                    continue
                    
    window = ReconstructionGUIApp(config_path, edit_config_only=edit_config_only)
    if suggest_volume_on_start:
        window.suggest_volume_action()
        window.save_config()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
