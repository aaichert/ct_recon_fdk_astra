#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import nrrd
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QTransform, QFont, QColor, QKeyEvent, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QPushButton, QRadioButton,
    QButtonGroup, QTextEdit, QFileDialog, QSplitter, QStatusBar, QMessageBox,
    QGroupBox, QCheckBox, QFormLayout, QComboBox, QGridLayout
)

class NRRDHeaderEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy and byte values in NRRD headers."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super().default(obj)

class ScientificDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(10)
        self.setRange(-1e300, 1e300)

    def validate(self, text, pos):
        from PyQt6.QtGui import QValidator
        if text in ["", "-", "+", "-.", "+.", "e", "E", "-e", "+e"]:
            return (QValidator.State.Intermediate, text, pos)
        import re
        pattern = r'^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$'
        if re.match(pattern, text):
            try:
                float(text)
                return (QValidator.State.Acceptable, text, pos)
            except ValueError:
                return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Invalid, text, pos)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value):
        # Format using scientific notation for very large/small numbers
        if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
            return f"{value:.4e}"
        else:
            return f"{value:.4f}".rstrip('0').rstrip('.')


class PowerOfTwoSpinBox(QDoubleSpinBox):
    """Spin box whose up/down arrows multiply or divide the value by 2."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(4)
        self.setRange(1.0 / 1024, 1024.0)
        self.setValue(1.0)

    def stepBy(self, steps):
        val = self.value()
        if steps > 0:
            for _ in range(steps):
                val = min(val * 2.0, self.maximum())
        else:
            for _ in range(-steps):
                val = max(val / 2.0, self.minimum())
        self.setValue(val)


class VoxelCanvas(QWidget):
    """Custom widget for rendering and interacting with 2D voxel slices."""
    hoverChanged = pyqtSignal(int, int, float)
    scaleChanged = pyqtSignal(float)
    sliceScrollRequested = pyqtSignal(int)
    rightDragBoxSelected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slice_data = None
        self.slice_image = None
        self.scale_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.last_mouse_pos = None
        self.drag_start_pos = None
        self.drag_current_pos = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def get_image_coords_clipped(self, mouse_pos):
        if self.slice_image is None:
            return None
        img_w, img_h = self.slice_image.width(), self.slice_image.height()
        canvas_w, canvas_h = self.width(), self.height()
        tx = mouse_pos.x() - (canvas_w / 2 + self.pan_offset.x())
        ty = mouse_pos.y() - (canvas_h / 2 + self.pan_offset.y())
        x_img = tx / self.scale_factor
        y_img = ty / self.scale_factor
        
        # Clamp to image boundaries
        x_img = max(-img_w / 2.0, min(img_w / 2.0, x_img))
        y_img = max(-img_h / 2.0, min(img_h / 2.0, y_img))
        return QPointF(x_img, y_img)

    def to_voxel_index(self, img_pos):
        if self.slice_image is None or img_pos is None:
            return None
        img_w, img_h = self.slice_image.width(), self.slice_image.height()
        x_vox = int(img_pos.x() + img_w / 2)
        y_vox = int(img_pos.y() + img_h / 2)
        x_vox = max(0, min(img_w - 1, x_vox))
        y_vox = max(0, min(img_h - 1, y_vox))
        return x_vox, y_vox

    def set_slice(self, raw_slice, q_image):
        self.slice_data = raw_slice
        self.slice_image = q_image
        self.update()

    def fit_view(self):
        if self.slice_image is None:
            return
        img_w, img_h = self.slice_image.width(), self.slice_image.height()
        canvas_w, canvas_h = self.width(), self.height()
        if img_w > 0 and img_h > 0 and canvas_w > 0 and canvas_h > 0:
            scale_w = canvas_w / img_w
            scale_h = canvas_h / img_h
            self.scale_factor = min(scale_w, scale_h) * 0.95
            self.pan_offset = QPointF(0, 0)
            self.scaleChanged.emit(self.scale_factor)
            self.update()

    def reset_view(self):
        self.scale_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.scaleChanged.emit(self.scale_factor)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        if self.slice_image is None:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No NRRD volume loaded.")
            return

        painter.save()
        img_w, img_h = self.slice_image.width(), self.slice_image.height()
        canvas_w, canvas_h = self.width(), self.height()

        painter.translate(canvas_w / 2 + self.pan_offset.x(), canvas_h / 2 + self.pan_offset.y())
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawImage(QRectF(-img_w / 2, -img_h / 2, img_w, img_h), self.slice_image)

        # Draw drag box if active
        if self.drag_start_pos is not None and self.drag_current_pos is not None:
            rect = QRectF(self.drag_start_pos, self.drag_current_pos).normalized()
            pen_color = QColor(0, 255, 255, 220)  # Cyan
            brush_color = QColor(0, 255, 255, 50)  # Semi-transparent cyan fill
            
            pen = QPen(pen_color)
            pen.setWidth(2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            
            painter.setPen(pen)
            painter.setBrush(brush_color)
            painter.drawRect(rect)

        painter.restore()

    def get_voxel_coords(self, mouse_pos):
        if self.slice_image is None or self.slice_data is None:
            return None
        img_w, img_h = self.slice_image.width(), self.slice_image.height()
        canvas_w, canvas_h = self.width(), self.height()

        tx = mouse_pos.x() - (canvas_w / 2 + self.pan_offset.x())
        ty = mouse_pos.y() - (canvas_h / 2 + self.pan_offset.y())
        
        x_img = tx / self.scale_factor
        y_img = ty / self.scale_factor
        
        # Translate from image center back to voxel index
        x_vox = int(x_img + img_w / 2)
        y_vox = int(y_img + img_h / 2)

        if 0 <= x_vox < img_w and 0 <= y_vox < img_h:
            try:
                # Row-major index matches y_vox (rows) and x_vox (cols)
                val = self.slice_data[y_vox, x_vox]
                return x_vox, y_vox, float(val)
            except IndexError:
                pass
        return None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            import math
            # Next power of 2 strictly above current scale
            exp = math.floor(math.log2(max(self.scale_factor, 1e-9))) + 1
            new_scale = min(2.0 ** exp, 100.0)
            if new_scale <= 100.0:
                pos = event.position()
                # Keep the cursor point fixed: shift pan by the zoom delta
                zoom_factor = new_scale / self.scale_factor
                d_x = pos.x() - (self.width()  / 2 + self.pan_offset.x())
                d_y = pos.y() - (self.height() / 2 + self.pan_offset.y())
                self.pan_offset = QPointF(
                    self.pan_offset.x() - d_x * (zoom_factor - 1),
                    self.pan_offset.y() - d_y * (zoom_factor - 1),
                )
                self.scale_factor = new_scale
                self.scaleChanged.emit(self.scale_factor)
                self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            if self.slice_image is not None:
                self.drag_start_pos = self.get_image_coords_clipped(event.position())
                self.drag_current_pos = self.drag_start_pos
                self.update()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.position()
            self.update()

        if self.drag_start_pos is not None:
            self.drag_current_pos = self.get_image_coords_clipped(event.position())
            self.update()

        coords = self.get_voxel_coords(event.position())
        if coords is not None:
            self.hoverChanged.emit(coords[0], coords[1], coords[2])

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
        elif event.button() == Qt.MouseButton.RightButton:
            if self.drag_start_pos is not None and self.drag_current_pos is not None:
                x1, y1 = self.to_voxel_index(self.drag_start_pos)
                x2, y2 = self.to_voxel_index(self.drag_current_pos)
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                if x_max >= x_min and y_max >= y_min:
                    self.rightDragBoxSelected.emit(x_min, y_min, x_max, y_max)
                self.drag_start_pos = None
                self.drag_current_pos = None
                self.update()

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            # Ctrl/Shift + wheel → zoom centred on cursor
            angle = event.angleDelta().y()
            zoom_factor = 1.15 if angle > 0 else 0.85
            new_scale = self.scale_factor * zoom_factor
            if 0.01 <= new_scale <= 100.0:
                pos = event.position()
                d_x = pos.x() - (self.width()  / 2 + self.pan_offset.x())
                d_y = pos.y() - (self.height() / 2 + self.pan_offset.y())
                self.pan_offset = QPointF(
                    self.pan_offset.x() - d_x * (zoom_factor - 1),
                    self.pan_offset.y() - d_y * (zoom_factor - 1),
                )
                self.scale_factor = new_scale
                self.scaleChanged.emit(self.scale_factor)
                self.update()
            return

        # Default: scroll through slices
        steps = 1 if event.angleDelta().y() > 0 else -1
        self.sliceScrollRequested.emit(steps)

MAX_VOLUMES = 9

class NrrdView3DWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NrrdView3D - NRRD Volume Viewer")
        self.resize(1024, 768)

        # Multi-volume storage: list of (file_path, data, header)
        self._volumes: list[tuple[str, object, dict]] = []
        self._active_volume_idx: int = 0  # 0-based index into _volumes

        # Convenience properties pointing to the active volume
        self.volume_data = None
        self.volume_header = {}
        self.active_axis = 2  # default Z-axis (slices along axis 2)
        self.current_slice_index = 0
        import shutil
        self.vol_render_path = shutil.which("VolumeRenderingGUIPy")

        self.setup_ui()

    def setup_ui(self):
        # Main split container
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Control Panel (Left)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(12)

        # File IO buttons
        file_io_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open NRRD")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_save = QPushButton("Save NRRD")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setEnabled(False)
        file_io_layout.addWidget(self.btn_open)
        file_io_layout.addWidget(self.btn_save)
        control_layout.addLayout(file_io_layout)

        # Volume selector dropdown (visible only when multiple volumes are loaded)
        self.volume_combo = QComboBox()
        self.volume_combo.setVisible(False)
        self.volume_combo.currentIndexChanged.connect(self._on_combo_changed)
        control_layout.addWidget(self.volume_combo)

        # Tab Widget
        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs)

        # Tab 1: Slicing & Contrast
        tab_view = QWidget()
        tab_view_layout = QVBoxLayout(tab_view)
        
        # Slicing Plane Selection
        axis_group = QGroupBox("Slicing Axis")
        axis_layout = QHBoxLayout(axis_group)
        self.radio_z = QRadioButton("Z-axis (X-Y)")
        self.radio_y = QRadioButton("Y-axis (X-Z)")
        self.radio_x = QRadioButton("X-axis (Y-Z)")
        self.radio_z.setChecked(True)
        axis_layout.addWidget(self.radio_z)
        axis_layout.addWidget(self.radio_y)
        axis_layout.addWidget(self.radio_x)
        
        self.axis_btn_group = QButtonGroup()
        self.axis_btn_group.addButton(self.radio_x, 0)
        self.axis_btn_group.addButton(self.radio_y, 1)
        self.axis_btn_group.addButton(self.radio_z, 2)
        self.axis_btn_group.idClicked.connect(self.change_axis)
        tab_view_layout.addWidget(axis_group)

        # Slice index selector
        slice_group = QGroupBox("Slice Navigation")
        slice_layout = QFormLayout(slice_group)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self.change_slice_slider)
        self.slice_spin = QSpinBox()
        self.slice_spin.valueChanged.connect(self.change_slice_spin)

        slice_row_widget = QWidget()
        slice_row_layout = QHBoxLayout(slice_row_widget)
        slice_row_layout.setContentsMargins(0, 0, 0, 0)
        slice_row_layout.addWidget(self.slice_slider)
        slice_row_layout.addWidget(self.slice_spin)
        slice_layout.addRow("Slice:", slice_row_widget)
        tab_view_layout.addWidget(slice_group)

        # Contrast mapping
        contrast_group = QGroupBox("Intensity Contrast Mapping")
        contrast_layout = QFormLayout(contrast_group)
        self.contrast_min_spin = ScientificDoubleSpinBox()
        self.contrast_min_spin.setValue(0.0)
        self.contrast_min_spin.valueChanged.connect(self.update_slice)
        
        self.contrast_max_spin = ScientificDoubleSpinBox()
        self.contrast_max_spin.setValue(1.0)
        self.contrast_max_spin.valueChanged.connect(self.update_slice)

        self.btn_auto_contrast = QPushButton("Auto-Contrast (Min/Max)")
        self.btn_auto_contrast.clicked.connect(self.apply_auto_contrast)
        self.btn_reset_intensity = QPushButton("Reset to Min/Max")
        self.btn_reset_intensity.clicked.connect(self.reset_intensity_min_max)
        self.chk_auto_contrast = QCheckBox("Auto-Contrast per Slice")

        auto_contrast_widget = QWidget()
        auto_contrast_layout = QVBoxLayout(auto_contrast_widget)
        auto_contrast_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_auto_contrast)
        btn_layout.addWidget(self.btn_reset_intensity)
        
        auto_contrast_layout.addLayout(btn_layout)
        auto_contrast_layout.addWidget(self.chk_auto_contrast)

        contrast_layout.addRow("Min Intensity (Black):", self.contrast_min_spin)
        contrast_layout.addRow("Max Intensity (White):", self.contrast_max_spin)
        contrast_layout.addRow(auto_contrast_widget)
        tab_view_layout.addWidget(contrast_group)

        # Zoom & Fit Controls
        zoom_group = QGroupBox("Scale & Fit")
        zoom_layout = QVBoxLayout(zoom_group)
        
        scale_input_layout = QHBoxLayout()
        scale_input_layout.addWidget(QLabel("Scale Factor:"))
        self.zoom_spin = PowerOfTwoSpinBox()
        self.zoom_spin.valueChanged.connect(self.change_zoom_spin)
        scale_input_layout.addWidget(self.zoom_spin)
        
        self.btn_fit = QPushButton("Fit to Window")
        self.btn_fit.clicked.connect(self.fit_window)
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.reset_view)
        
        zoom_buttons_layout = QHBoxLayout()
        zoom_buttons_layout.addWidget(self.btn_fit)
        zoom_buttons_layout.addWidget(self.btn_reset_view)
        
        zoom_layout.addLayout(scale_input_layout)
        zoom_layout.addLayout(zoom_buttons_layout)
        tab_view_layout.addWidget(zoom_group)
        
        if hasattr(self, 'vol_render_path') and self.vol_render_path:
            render_group = QGroupBox("3D Visualization")
            render_layout = QVBoxLayout(render_group)
            self.btn_vol_render = QPushButton("Volume Rendering")
            self.btn_vol_render.clicked.connect(self.run_volume_rendering)
            self.btn_vol_render.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
            render_layout.addWidget(self.btn_vol_render)
            tab_view_layout.addWidget(render_group)

        tab_view_layout.addStretch()

        self.tabs.addTab(tab_view, "View Settings")

        # Tab 2: JSON Metadata
        tab_json = QWidget()
        tab_json_layout = QVBoxLayout(tab_json)
        self.txt_json = QTextEdit()
        self.txt_json.setFont(QFont("Courier New", 10))
        self.btn_update_json = QPushButton("Update Header in Memory")
        self.btn_update_json.clicked.connect(self.update_header_from_json)
        
        tab_json_layout.addWidget(QLabel("Raw Metadata (JSON Dictionary):"))
        tab_json_layout.addWidget(self.txt_json)
        tab_json_layout.addWidget(self.btn_update_json)
        self.tabs.addTab(tab_json, "Metadata JSON")

        # Tab 3: Operations
        tab_ops = QWidget()
        tab_ops_layout = QVBoxLayout(tab_ops)

        # Type conversion
        conv_group = QGroupBox("Type Conversion")
        conv_layout = QHBoxLayout(conv_group)
        self.btn_conv_float = QPushButton("To Float32")
        self.btn_conv_float.clicked.connect(self.convert_to_float)
        self.btn_conv_uint8 = QPushButton("To Uint8")
        self.btn_conv_uint8.clicked.connect(self.convert_to_uint8)
        conv_layout.addWidget(self.btn_conv_float)
        conv_layout.addWidget(self.btn_conv_uint8)
        tab_ops_layout.addWidget(conv_group)

        # Downsampling/Binning
        bin_group = QGroupBox("2x2 Binning")
        bin_layout = QVBoxLayout(bin_group)
        self.chk_bin_3d = QCheckBox("Isotropic 3D Binning (2x2x2)")
        self.chk_bin_3d.setChecked(True)
        self.btn_apply_bin = QPushButton("Apply Binning")
        self.btn_apply_bin.clicked.connect(self.apply_binning)
        bin_layout.addWidget(self.chk_bin_3d)
        bin_layout.addWidget(self.btn_apply_bin)
        tab_ops_layout.addWidget(bin_group)

        # Cropping
        crop_group = QGroupBox("Volume Cropping")
        crop_layout = QFormLayout(crop_group)
        self.crop_x_start = QSpinBox()
        self.crop_x_end = QSpinBox()
        self.crop_y_start = QSpinBox()
        self.crop_y_end = QSpinBox()
        self.crop_z_start = QSpinBox()
        self.crop_z_end = QSpinBox()

        crop_layout.addRow("X range:", self.make_range_layout(self.crop_x_start, self.crop_x_end))
        crop_layout.addRow("Y range:", self.make_range_layout(self.crop_y_start, self.crop_y_end))
        crop_layout.addRow("Z range:", self.make_range_layout(self.crop_z_start, self.crop_z_end))
        
        self.btn_apply_crop = QPushButton("Apply Crop")
        self.btn_apply_crop.clicked.connect(self.apply_cropping)
        crop_layout.addWidget(self.btn_apply_crop)
        tab_ops_layout.addWidget(crop_group)
        # Data Layout Group Box
        layout_group = QGroupBox("Data Layout")
        layout_grid = QGridLayout(layout_group)
        layout_grid.setSpacing(6)
        
        lbl_flip = QLabel("Flip:")
        self.btn_flip_x = QPushButton("X")
        self.btn_flip_y = QPushButton("Y")
        self.btn_flip_z = QPushButton("Z")
        
        lbl_transpose = QLabel("Transpose:")
        self.btn_trans_xy = QPushButton("X->Y")
        self.btn_trans_xz = QPushButton("X->Z")
        self.btn_trans_yz = QPushButton("Y->Z")
        
        # Connect buttons
        self.btn_flip_x.clicked.connect(lambda: self.apply_flip(0))
        self.btn_flip_y.clicked.connect(lambda: self.apply_flip(1))
        self.btn_flip_z.clicked.connect(lambda: self.apply_flip(2))
        
        self.btn_trans_xy.clicked.connect(lambda: self.apply_transpose(0, 1))
        self.btn_trans_xz.clicked.connect(lambda: self.apply_transpose(0, 2))
        self.btn_trans_yz.clicked.connect(lambda: self.apply_transpose(1, 2))
        
        # Align buttons in grid
        layout_grid.addWidget(lbl_flip, 0, 0)
        layout_grid.addWidget(self.btn_flip_x, 0, 1)
        layout_grid.addWidget(self.btn_flip_y, 0, 2)
        layout_grid.addWidget(self.btn_flip_z, 0, 3)
        
        layout_grid.addWidget(lbl_transpose, 1, 0)
        layout_grid.addWidget(self.btn_trans_xy, 1, 1)
        layout_grid.addWidget(self.btn_trans_xz, 1, 2)
        layout_grid.addWidget(self.btn_trans_yz, 1, 3)
        
        tab_ops_layout.addWidget(layout_group)
        tab_ops_layout.addStretch()

        self.tabs.addTab(tab_ops, "Operations")

        # Disable fields initially
        self.set_volume_dependent_widgets_enabled(False)

        # Canvas (Right)
        self.canvas = VoxelCanvas()
        self.canvas.hoverChanged.connect(self.update_hover_info)
        self.canvas.scaleChanged.connect(self.update_zoom_spin_value)
        self.canvas.sliceScrollRequested.connect(self.scroll_slice)
        self.canvas.rightDragBoxSelected.connect(self.apply_box_contrast_stretch)

        # Add to splitter
        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(self.canvas)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 7)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load an NRRD reconstruction to begin.")

        # Accept key events on the main window
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def make_range_layout(self, spin_start, spin_end):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Min:"))
        lay.addWidget(spin_start)
        lay.addWidget(QLabel("Max:"))
        lay.addWidget(spin_end)
        return w

    def set_volume_dependent_widgets_enabled(self, enabled):
        self.btn_save.setEnabled(enabled)
        self.tabs.setEnabled(enabled)

    def run_volume_rendering(self):
        if not self.file_path:
            QMessageBox.warning(self, "No Volume", "No NRRD volume file path is loaded.")
            return
        import subprocess
        try:
            subprocess.Popen([self.vol_render_path, self.file_path])
            self.status_bar.showMessage("Started VolumeRenderingGUIPy in a separate process.")
        except Exception as e:
            QMessageBox.critical(self, "Error Running VolumeRenderingGUIPy", f"Could not launch VolumeRenderingGUIPy:\n{e}")

    def change_slice_slider(self, val):
        if self.current_slice_index != val:
            self.current_slice_index = val
            self.slice_spin.setValue(val)
            self.update_slice()

    def change_slice_spin(self, val):
        if self.current_slice_index != val:
            self.current_slice_index = val
            self.slice_slider.setValue(val)
            self.update_slice()

    def change_axis(self, idx):
        if self.volume_data is None:
            return
        self.active_axis = idx
        sizes = self.volume_data.shape
        max_idx = sizes[self.active_axis] - 1
        
        self.slice_slider.setRange(0, max_idx)
        self.slice_spin.setRange(0, max_idx)
        
        self.current_slice_index = max_idx // 2
        self.slice_slider.setValue(self.current_slice_index)
        self.slice_spin.setValue(self.current_slice_index)
        
        self.update_slice()

    # ------------------------------------------------------------------
    # Multi-volume helpers
    # ------------------------------------------------------------------
    def _combo_label(self, idx_1based: int, file_path: str) -> str:
        """Format the dropdown label for a volume entry (1-based index)."""
        parent = os.path.basename(os.path.dirname(file_path))
        fname = os.path.basename(file_path)
        return f"{idx_1based}: {parent}/{fname}"

    def _rebuild_combo(self):
        """Repopulate the volume selector dropdown from _volumes."""
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for i, (fp, _data, _hdr) in enumerate(self._volumes):
            self.volume_combo.addItem(self._combo_label(i + 1, fp))
        self.volume_combo.setCurrentIndex(self._active_volume_idx)
        self.volume_combo.setVisible(len(self._volumes) > 1)
        self.volume_combo.blockSignals(False)
        # Enable auto-contrast per slice automatically once multiple volumes are loaded
        if len(self._volumes) > 1 and not self.chk_auto_contrast.isChecked():
            self.chk_auto_contrast.setChecked(True)

    def _activate_volume(self, idx: int):
        """Switch the active volume to the given 0-based index.

        Only swaps the data pointer and redraws the current slice.
        Slice position, axis, contrast, zoom and pan are preserved.
        """
        if not (0 <= idx < len(self._volumes)):
            return
        self._active_volume_idx = idx
        fp, data, header = self._volumes[idx]
        self.volume_data = data
        self.volume_header = header
        self.file_path = fp
        # Sync dropdown without re-triggering
        self.volume_combo.blockSignals(True)
        self.volume_combo.setCurrentIndex(idx)
        self.volume_combo.blockSignals(False)
        # Redraw the slice only — nothing else changes
        self.update_slice()
        self.status_bar.showMessage(
            f"[{idx + 1}/{len(self._volumes)}] {os.path.basename(fp)} "
            f"(Shape: {data.shape}, Type: {data.dtype})"
        )

    def _on_combo_changed(self, combo_idx: int):
        """Called when the user picks a different entry in the dropdown."""
        if combo_idx != self._active_volume_idx:
            self._activate_volume(combo_idx)

    # ------------------------------------------------------------------
    # Key handling – number keys 1-9 switch volumes
    # ------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            requested = key - Qt.Key.Key_1  # convert to 0-based
            if requested < len(self._volumes):
                self._activate_volume(requested)
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def open_file(self, file_path=None):
        """Open one or more NRRD files.

        Accepts:
          - no argument   → show multi-select file dialog
          - a str path    → load that single file
          - a list[str]   → load all paths in the list (no dialog)
        In all cases exactly one error dialog can appear.
        """
        if file_path is None:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Open NRRD File(s)", "", "NRRD Volumes (*.nrrd);;All Files (*)"
            )
        elif isinstance(file_path, list):
            paths = file_path
        else:
            paths = [file_path]

        if not paths:
            return

        # Read all files silently - no dialogs yet
        self.status_bar.showMessage(f"Reading {len(paths)} file(s)...")
        rows        = []  # (basename, shape_or_dash, problem_note)  – for the report
        candidates  = []  # (path, data, header) – valid 3-D volumes

        for p in paths:
            try:
                data, header = nrrd.read(p)
                if len(data.shape) != 3:
                    rows.append((os.path.basename(p), str(data.shape), "not a 3-D volume"))
                else:
                    candidates.append((p, data, header))
            except Exception as exc:
                rows.append((os.path.basename(p), "—", str(exc)))

        # Silently cap at MAX_VOLUMES; note extras in the report
        slots_free = MAX_VOLUMES - len(self._volumes)
        for p, d, _ in candidates[slots_free:]:
            rows.append((os.path.basename(p), str(d.shape), "exceeds {}-volume limit".format(MAX_VOLUMES)))
        candidates = candidates[:slots_free]

        # Determine reference shape
        ref_path = ref_shape = None
        if self._volumes:
            ref_path, ref_data, _ = self._volumes[0]
            ref_shape = ref_data.shape
        elif candidates:
            ref_path, ref_data, _ = candidates[0]
            ref_shape = ref_data.shape

        # Collect size mismatches
        size_ok = True
        if ref_shape is not None:
            for p, d, _ in candidates:
                if d.shape != ref_shape:
                    size_ok = False

        # Build and show AT MOST ONE dialog if anything is wrong
        if rows or not size_ok:
            lines = []
            if ref_shape is not None:
                lines.append("Reference shape: {}  ({})".format(ref_shape, os.path.basename(ref_path)))
                lines.append("")
                lines.append("All files:")
                col = max((len(os.path.basename(p)) for p, _, _ in candidates), default=4)
                for p, d, _ in candidates:
                    mark = "OK " if d.shape == ref_shape else "ERR"
                    lines.append("  [{}]  {:<{}}  {}".format(mark, os.path.basename(p), col, d.shape))
            if rows:
                if lines:
                    lines.append("")
                lines.append("Could not load:")
                col = max((len(r[0]) for r in rows), default=4)
                for name, shape, note in rows:
                    lines.append("  {:<{}}  {}  {}".format(name, col, shape, note))

            if not size_ok or not candidates:
                QMessageBox.critical(self, "Cannot Load Files", "\n".join(lines))
                self.status_bar.showMessage("Load aborted.")
                return
            else:
                QMessageBox.warning(self, "Some Files Skipped", "\n".join(lines))

        if not candidates:
            self.status_bar.showMessage("No files loaded.")
            return

        # Commit all validated candidates
        start_idx = len(self._volumes)
        for fp, data, header in candidates:
            self._volumes.append((fp, data, header))

        self._active_volume_idx = start_idx
        fp, data, header = self._volumes[self._active_volume_idx]
        self.volume_data   = data
        self.volume_header = header
        self.file_path     = fp

        self._rebuild_combo()
        self.set_volume_dependent_widgets_enabled(True)
        self.change_axis(self.active_axis)
        self.reset_crop_spinboxes()
        self.apply_auto_contrast()
        self.update_metadata_json()
        self.fit_window()

        n = len(candidates)
        self.status_bar.showMessage(
            "[{}/{}] Loaded {} file(s). Active: {} (Shape: {}, Type: {})".format(
                len(self._volumes), MAX_VOLUMES, n,
                os.path.basename(fp), data.shape, data.dtype)
        )


    def save_file(self):
        if self.volume_data is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save NRRD File", self.file_path or "", "NRRD Volumes (*.nrrd);;All Files (*)"
        )
        if file_path:
            try:
                self.status_bar.showMessage(f"Saving to {os.path.basename(file_path)}...")
                # Write to NRRD
                nrrd.write(file_path, self.volume_data, header=self.volume_header)
                self.file_path = file_path
                self.status_bar.showMessage(f"Successfully saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save NRRD file:\n{str(e)}")
                self.status_bar.showMessage("Save failed.")

    def update_slice(self):
        if self.volume_data is None:
            return

        # Slicing
        idx = self.current_slice_index
        if self.active_axis == 0:
            slice_data = self.volume_data[idx, :, :].T
        elif self.active_axis == 1:
            slice_data = self.volume_data[:, idx, :].T
        else:
            slice_data = self.volume_data[:, :, idx].T

        # Auto-contrast per slice: compute and update spinboxes without re-triggering update_slice
        if self.chk_auto_contrast.isChecked():
            p1 = float(np.percentile(slice_data, 1))
            p99 = float(np.percentile(slice_data, 99))
            if p1 == p99:
                p1 = float(np.min(slice_data))
                p99 = float(np.max(slice_data))
            self.contrast_min_spin.blockSignals(True)
            self.contrast_max_spin.blockSignals(True)
            self.contrast_min_spin.setValue(p1)
            self.contrast_max_spin.setValue(p99)
            self.contrast_min_spin.blockSignals(False)
            self.contrast_max_spin.blockSignals(False)

        min_val = self.contrast_min_spin.value()
        max_val = self.contrast_max_spin.value()
        diff = max_val - min_val
        if diff <= 0:
            diff = 1e-5

        # Perform mapped linear normalization to [0, 255]
        norm_data = np.clip((slice_data - min_val) / diff * 255.0, 0, 255).astype(np.uint8)

        height, width = norm_data.shape
        # Recreate QImage from numpy buffer
        q_image = QImage(norm_data.data, width, height, width, QImage.Format.Format_Grayscale8)

        # Critical copy to decouple QImage from the temporary numpy memory array buffer
        q_image_copy = q_image.copy()

        self.canvas.set_slice(slice_data, q_image_copy)

    def apply_auto_contrast(self):
        if self.volume_data is None:
            return
        # Calculate optimal contrast range based on the active 2D slice
        idx = self.current_slice_index
        if self.active_axis == 0:
            slice_data = self.volume_data[idx, :, :]
        elif self.active_axis == 1:
            slice_data = self.volume_data[:, idx, :]
        else:
            slice_data = self.volume_data[:, :, idx]

        # Use 1st and 99th percentiles for robust mapping bounds
        p1 = float(np.percentile(slice_data, 1))
        p99 = float(np.percentile(slice_data, 99))

        if p1 == p99:
            # Fall back to global min/max
            p1 = float(np.min(slice_data))
            p99 = float(np.max(slice_data))

        self.contrast_min_spin.setValue(p1)
        self.contrast_max_spin.setValue(p99)
        self.update_slice()

    def fit_window(self):
        if self.volume_data is None:
            return
        self.canvas.fit_view()

    def reset_view(self):
        if self.volume_data is None:
            return
        self.canvas.reset_view()

    def reset_intensity_min_max(self):
        if self.volume_data is None:
            return
        g_min = float(np.min(self.volume_data))
        g_max = float(np.max(self.volume_data))
        
        # Uncheck Auto-Contrast per Slice to preserve this manual stretch
        if self.chk_auto_contrast.isChecked():
            self.chk_auto_contrast.setChecked(False)
            
        self.contrast_min_spin.setValue(g_min)
        self.contrast_max_spin.setValue(g_max)
        self.update_slice()

    def apply_box_contrast_stretch(self, x_min, y_min, x_max, y_max):
        if self.volume_data is None or self.canvas.slice_data is None:
            return
        
        # Extract pixels within the box
        box_pixels = self.canvas.slice_data[y_min : y_max + 1, x_min : x_max + 1]
        if box_pixels.size == 0:
            return
            
        p1 = float(np.percentile(box_pixels, 1))
        p99 = float(np.percentile(box_pixels, 99))
        
        if p1 == p99:
            p1 = float(np.min(box_pixels))
            p99 = float(np.max(box_pixels))
            
        # Uncheck Auto-Contrast per Slice to preserve this manual stretch
        if self.chk_auto_contrast.isChecked():
            self.chk_auto_contrast.setChecked(False)
            
        # Update the UI controls
        self.contrast_min_spin.setValue(p1)
        self.contrast_max_spin.setValue(p99)
        self.update_slice()
        self.status_bar.showMessage(
            f"Contrast stretched to box statistics: Min={p1:.4f}, Max={p99:.4f}"
        )

    def change_zoom_spin(self, val):
        if self.volume_data is not None:
            if abs(self.canvas.scale_factor - val) > 1e-4:
                self.canvas.scale_factor = val
                self.canvas.update()

    def update_zoom_spin_value(self, val):
        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(val)
        self.zoom_spin.blockSignals(False)

    def scroll_slice(self, steps):
        if self.volume_data is None:
            return
        new_idx = self.current_slice_index + steps
        # Clamp
        max_idx = self.slice_slider.maximum()
        new_idx = max(0, min(max_idx, new_idx))
        self.slice_slider.setValue(new_idx)

    def update_metadata_json(self):
        # Clean header dict so it can be dumped to valid JSON
        clean_header = {}
        for k, v in self.volume_header.items():
            if isinstance(v, np.ndarray):
                clean_header[k] = v.tolist()
            elif isinstance(v, bytes):
                clean_header[k] = v.decode('utf-8')
            else:
                clean_header[k] = v

        json_str = json.dumps(clean_header, indent=4, cls=NRRDHeaderEncoder)
        self.txt_json.setText(json_str)

    def update_header_from_json(self):
        if self.volume_data is None:
            return
        try:
            raw_text = self.txt_json.toPlainText()
            updated_dict = json.loads(raw_text)
            
            # Recast back fields that are required as specific types
            parsed_header = {}
            for k, v in updated_dict.items():
                if k == 'sizes':
                    parsed_header[k] = [int(x) for x in v]
                elif k == 'space directions':
                    parsed_header[k] = []
                    for direction in v:
                        if direction is None:
                            parsed_header[k].append(None)
                        else:
                            parsed_header[k].append([float(x) for x in direction])
                elif k == 'space origin':
                    parsed_header[k] = [float(x) for x in v]
                elif k == 'dimension':
                    parsed_header[k] = int(v)
                else:
                    parsed_header[k] = v

            self.volume_header = parsed_header
            QMessageBox.information(self, "Metadata Updated", "Header has been successfully loaded into memory.")
            self.update_slice()
        except Exception as e:
            QMessageBox.critical(self, "JSON Parse Error", f"Failed to parse and update header metadata:\n{str(e)}")

    def update_hover_info(self, x, y, val):
        if self.volume_data is None:
            return
        # Display coordinate positions
        # Standard index calculation based on slice coordinates
        idx = self.current_slice_index
        
        # Check active slicing axis to output true coordinate
        if self.active_axis == 0:
            # Sliced X, displayed Y-Z transposed.
            # Transposed means: row is Z, col is Y.
            # So y is Z, x is Y.
            voxel_coords = (idx, x, y)
        elif self.active_axis == 1:
            # Sliced Y, displayed X-Z transposed.
            # Transposed means: row is Z, col is X.
            # So y is Z, x is X.
            voxel_coords = (x, idx, y)
        else:
            # Sliced Z, displayed X-Y transposed.
            # Transposed means: row is Y, col is X.
            # So y is Y, x is X.
            voxel_coords = (x, y, idx)

        # Get space physical coords if space directions and origins are present
        origin = self.volume_header.get('space origin', None)
        dirs = self.volume_header.get('space directions', None)
        
        phys_str = ""
        if origin is not None and dirs is not None:
            try:
                phys_pos = np.array(origin, dtype=float)
                for i, vox in enumerate(voxel_coords):
                    d_vec = dirs[i]
                    if d_vec is not None:
                        phys_pos += vox * np.array(d_vec, dtype=float)
                phys_str = f" | Physical: ({phys_pos[0]:.2f}, {phys_pos[1]:.2f}, {phys_pos[2]:.2f}) mm"
            except Exception:
                pass

        self.status_bar.showMessage(
            f"Voxel: ({voxel_coords[0]}, {voxel_coords[1]}, {voxel_coords[2]}) | Raw Intensity: {val:.4f}{phys_str}"
        )

    # Operations
    def convert_to_float(self):
        if self.volume_data is None:
            return
        self.volume_data = self.volume_data.astype(np.float32)
        self.volume_header['type'] = 'float'
        self.update_metadata_json()
        self.update_slice()
        self.status_bar.showMessage("Volume converted to Float32.")
        QMessageBox.information(self, "Type Conversion", "Volume data successfully cast to float32.")

    def convert_to_uint8(self):
        if self.volume_data is None:
            return
        min_val = self.contrast_min_spin.value()
        max_val = self.contrast_max_spin.value()
        diff = max_val - min_val
        if diff <= 0:
            diff = 1e-5

        # Normalize and cast entire volume
        self.volume_data = np.clip((self.volume_data - min_val) / diff * 255.0, 0, 255).astype(np.uint8)
        self.volume_header['type'] = 'uchar'
        
        # Reset contrast controls to uint8 range
        self.contrast_min_spin.setValue(0.0)
        self.contrast_max_spin.setValue(255.0)
        
        self.update_metadata_json()
        self.update_slice()
        self.status_bar.showMessage("Volume converted to Uint8.")
        QMessageBox.information(self, "Type Conversion", "Volume data successfully scaled and cast to uint8.")

    def apply_binning(self):
        if self.volume_data is None:
            return
        
        is_3d = self.chk_bin_3d.isChecked()
        shape = self.volume_data.shape
        
        try:
            if is_3d:
                # Isotropic 2x2x2 binning
                sz = [s - (s % 2) for s in shape]
                if any(s < 2 for s in sz):
                    raise ValueError("Volume dimensions are too small to apply 2x2 binning.")
                
                truncated = self.volume_data[:sz[0], :sz[1], :sz[2]]
                # Fast mean downsampling via reshaping
                binned = truncated.reshape(sz[0]//2, 2, sz[1]//2, 2, sz[2]//2, 2).mean(axis=(1, 3, 5))
                self.volume_data = binned
                
                # Update spacing metadata by factor of 2
                if 'space directions' in self.volume_header:
                    new_dirs = []
                    for d in self.volume_header['space directions']:
                        if d is not None:
                            new_dirs.append([float(x) * 2.0 for x in d])
                        else:
                            new_dirs.append(None)
                    self.volume_header['space directions'] = new_dirs
            else:
                # In-plane 2D binning (2x2x1) on axes 0 and 1
                sz_x = shape[0] - (shape[0] % 2)
                sz_y = shape[1] - (shape[1] % 2)
                if sz_x < 2 or sz_y < 2:
                    raise ValueError("X/Y Dimensions are too small to apply 2x2 binning.")
                
                truncated = self.volume_data[:sz_x, :sz_y, :]
                binned = truncated.reshape(sz_x//2, 2, sz_y//2, 2, shape[2]).mean(axis=(1, 3))
                self.volume_data = binned

                if 'space directions' in self.volume_header:
                    new_dirs = []
                    for i, d in enumerate(self.volume_header['space directions']):
                        if d is not None and i < 2:
                            new_dirs.append([float(x) * 2.0 for x in d])
                        else:
                            new_dirs.append(d)
                    self.volume_header['space directions'] = new_dirs

            self.volume_header['sizes'] = list(self.volume_data.shape)
            
            # Reset slice sliders
            self.change_axis(self.active_axis)
            self.reset_crop_spinboxes()
            self.update_metadata_json()
            self.update_slice()
            
            self.status_bar.showMessage(f"2x2 Binning complete. New Shape: {self.volume_data.shape}")
            QMessageBox.information(self, "Binning Complete", f"Applied 2x2 binning.\nNew Shape: {self.volume_data.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Binning Error", f"Failed to apply binning:\n{str(e)}")

    def reset_crop_spinboxes(self):
        if self.volume_data is None:
            return
        nx, ny, nz = self.volume_data.shape
        
        self.crop_x_start.setRange(0, nx - 1)
        self.crop_x_start.setValue(0)
        self.crop_x_end.setRange(1, nx)
        self.crop_x_end.setValue(nx)

        self.crop_y_start.setRange(0, ny - 1)
        self.crop_y_start.setValue(0)
        self.crop_y_end.setRange(1, ny)
        self.crop_y_end.setValue(ny)

        self.crop_z_start.setRange(0, nz - 1)
        self.crop_z_start.setValue(0)
        self.crop_z_end.setRange(1, nz)
        self.crop_z_end.setValue(nz)

    def apply_cropping(self):
        if self.volume_data is None:
            return

        x_start = self.crop_x_start.value()
        x_end = self.crop_x_end.value()
        y_start = self.crop_y_start.value()
        y_end = self.crop_y_end.value()
        z_start = self.crop_z_start.value()
        z_end = self.crop_z_end.value()

        if x_start >= x_end or y_start >= y_end or z_start >= z_end:
            QMessageBox.warning(self, "Crop Bounds Error", "Start index must be strictly less than End index.")
            return

        try:
            # Perform numpy crop slicing
            self.volume_data = self.volume_data[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Recalculate spatial physical origin shift
            origin = self.volume_header.get('space origin', None)
            dirs = self.volume_header.get('space directions', None)
            if origin is not None and dirs is not None:
                new_origin = np.array(origin, dtype=float)
                offsets = [x_start, y_start, z_start]
                for i, idx in enumerate(offsets):
                    d_vec = dirs[i]
                    if d_vec is not None:
                        new_origin += idx * np.array(d_vec, dtype=float)
                self.volume_header['space origin'] = new_origin.tolist()

            self.volume_header['sizes'] = list(self.volume_data.shape)

            self.change_axis(self.active_axis)
            self.reset_crop_spinboxes()
            self.update_metadata_json()
            self.update_slice()

            self.status_bar.showMessage(f"Crop complete. New Shape: {self.volume_data.shape}")
            QMessageBox.information(self, "Crop Complete", f"Applied cropping.\nNew Shape: {self.volume_data.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Crop Error", f"Failed to crop volume:\n{str(e)}")

    def apply_flip(self, axis):
        if self.volume_data is None:
            return
        try:
            self.volume_data = np.flip(self.volume_data, axis=axis)
            self._volumes[self._active_volume_idx] = (self.file_path, self.volume_data, self.volume_header)
            
            self.change_axis(self.active_axis)
            self.reset_crop_spinboxes()
            self.update_metadata_json()
            self.update_slice()
            
            axes_names = ["X", "Y", "Z"]
            self.status_bar.showMessage(f"Flipped volume along {axes_names[axis]}-axis.")
        except Exception as e:
            QMessageBox.critical(self, "Flip Error", f"Failed to flip volume:\n{str(e)}")

    def apply_transpose(self, ax1, ax2):
        if self.volume_data is None:
            return
        try:
            self.volume_data = np.swapaxes(self.volume_data, ax1, ax2)
            
            # Update header sizes
            self.volume_header['sizes'] = list(self.volume_data.shape)
            
            # Swap space directions if present
            if 'space directions' in self.volume_header:
                sd = list(self.volume_header['space directions'])
                if ax1 < len(sd) and ax2 < len(sd):
                    sd[ax1], sd[ax2] = sd[ax2], sd[ax1]
                    self.volume_header['space directions'] = sd
            
            self._volumes[self._active_volume_idx] = (self.file_path, self.volume_data, self.volume_header)
            
            self.change_axis(self.active_axis)
            self.reset_crop_spinboxes()
            self.update_metadata_json()
            self.update_slice()
            
            axes_names = ["X", "Y", "Z"]
            self.status_bar.showMessage(f"Swapped {axes_names[ax1]} and {axes_names[ax2]} axes.")
        except Exception as e:
            QMessageBox.critical(self, "Transpose Error", f"Failed to transpose volume:\n{str(e)}")



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
    
    window = NrrdView3DWindow()
    window.show()

    # Collect CLI paths and pass them ALL at once so only one error dialog can appear
    cli_paths = [p for p in sys.argv[1:] if os.path.exists(p)][:MAX_VOLUMES]

    if cli_paths:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: window.open_file(cli_paths))

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
