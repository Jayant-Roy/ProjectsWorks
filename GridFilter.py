
# GeoFFT – CSV + Raster FFT Filters (Oasis-style, CORRECT)


import sys
import numpy as np
import pandas as pd
import cv2
import rasterio

from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QFormLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# CSV → GRID HELPERS

def create_grid(xmin, xmax, ymin, ymax, cell):
    
    MIN_CELL = 0.01      # minimum allowed cell size
    MAX_CELLS = 5000     # maximum grid size per direction

    # Enforce minimum cell size (hard safety)
    cell = max(cell, MIN_CELL)

    # Compute number of grid cells
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1

    # Safety check: grid too large
    if nx > MAX_CELLS or ny > MAX_CELLS:
        raise RuntimeError(
            f"Grid too large ({nx} x {ny}). "
            f"Increase cell size (current cell = {cell})."
        )

    # Create grid coordinates
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymax, ymin, ny)  # top-to-bottom for raster

    return np.meshgrid(gx, gy)


# def create_grid(xmin, xmax, ymin, ymax, cell):
#     max_cells = 5000  # safe upper limit per direction
#     min_cells = 0.01  # hard safety limit (meters or degrees)

#     nx = int(np.ceil((xmax() - xmin()) / cell)) + 1
#     ny = int(np.ceil((ymax() - ymin()) / cell)) + 1

#     if min_cells > nx > max_cells or min_cells > ny > max_cells:
#         raise RuntimeError(
#             f"Grid too large/ ({nx} x {ny}). Increase cell size."
#         )

#     # nx = int(np.ceil((xmax - xmin) / cell)) + 1
#     # ny = int(np.ceil((ymax - ymin) / cell)) + 1
#     # gx = np.linspace(xmin, xmax, nx)
#     # gy = np.linspace(ymax, ymin, ny)
#     # return np.meshgrid(gx, gy)


def idw(x, y, z, gx, gy, power=2, k=8):
    tree = cKDTree(np.column_stack((x, y)))
    d, idx = tree.query(np.column_stack((gx.ravel(), gy.ravel())), k=k)
    d[d == 0] = 1e-12
    w = 1.0 / (d ** power)
    vals = np.sum(w * z[idx], axis=1) / np.sum(w, axis=1)
    return vals.reshape(gx.shape)


# MAIN GUI

class FFTGeoscience(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoFFT - CSV & Raster FFT Filters")
        self.setGeometry(100, 100, 1600, 780)

        self.df = None
        self.image = None
        self.base_image = None
        self.working_image = None
        self.last_output = None
        self.src_profile = None

        self.last_vmin = None
        self.last_vmax = None

        self.init_ui()

    def update_param_visibility(self):

        f = self.filter_box.currentText()

        # Hide all first
        for i in range(self.param_layout.rowCount()):
            self.param_layout.itemAt(i, QFormLayout.LabelRole).widget().hide()
            self.param_layout.itemAt(i, QFormLayout.FieldRole).widget().hide()

        def show(row):
            self.param_layout.itemAt(row, QFormLayout.LabelRole).widget().show()
            self.param_layout.itemAt(row, QFormLayout.FieldRole).widget().show()

        if f in ["Low Pass Filter", "High Pass Filter"]:
            show(0)  # cutoff

        elif f == "Band Pass Filter":
            show(1)  # low
            show(2)  # high

        elif f == "Gaussian High Pass":
            show(3)  # wavelength

        elif f in ["Horizontal Derivative", "Vertical Derivative"]:
            show(4)  # derivative order

        elif f == "Reduction To Pole":
            show(6)  # inclination
            show(7)  # declination

        elif f == "Tilt Derivative":
            show(4)  # derivative order
            show(5)  # z-derivative


    # --------------------------------------------------------
    def init_ui(self):

        main = QWidget()
        layout = QHBoxLayout(main)

        #  CONTROLS 
        control_box = QGroupBox("Controls")
        ctrl = QVBoxLayout(control_box)

        # -------- CSV → GRID --------
        csv_box = QGroupBox("CSV → Grid")
        csv_form = QFormLayout(csv_box)

        self.csv_btn = QPushButton("Load CSV / TXT")
        self.csv_btn.clicked.connect(self.load_table)

        self.x_cb = QComboBox()
        self.y_cb = QComboBox()
        self.z_cb = QComboBox()

        self.grid_method = QComboBox()
        self.grid_method.addItems(["IDW", "linear", "cubic", "nearest"])

        self.cell_size = QDoubleSpinBox()
        self.cell_size.setDecimals(2)
        self.cell_size.setRange(0.01, 1e6)
        self.cell_size.setSingleStep(0.01)
        self.cell_size.setValue(0.1)

        #self.cell_size.setRange(1e-6, 1e6)
        #self.cell_size.setValue(0.01)

        self.grid_btn = QPushButton("Create Grid from CSV")
        self.grid_btn.clicked.connect(self.create_grid_from_csv)

        csv_form.addRow(self.csv_btn)
        csv_form.addRow("X / Lon", self.x_cb)
        csv_form.addRow("Y / Lat", self.y_cb)
        csv_form.addRow("Value", self.z_cb)
        csv_form.addRow("Method", self.grid_method)
        csv_form.addRow("Cell size", self.cell_size)
        csv_form.addRow(self.grid_btn)

        # RASTER LOAD 
        self.load_btn = QPushButton("Load Raster / GeoTIFF")
        self.load_btn.clicked.connect(self.load_image)

        # FILTERS
        self.filter_box = QComboBox()
        self.filter_box.addItems([
            "Low Pass Filter",
            "High Pass Filter",
            "Band Pass Filter",
            "Gaussian High Pass",
            "Horizontal Derivative",
            "Vertical Derivative",
            "Reduction To Pole",
            "Tilt Derivative"
        ])
        self.filter_box.currentTextChanged.connect(self.update_param_visibility)

        param_box = QGroupBox("Filter Parameters")
        self.param_layout = QFormLayout(param_box)

        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.001, 0.5)
        self.cutoff.setValue(0.15)

        self.low_cut = QDoubleSpinBox()
        self.low_cut.setRange(0.001, 0.5)
        self.low_cut.setValue(0.05)

        self.high_cut = QDoubleSpinBox()
        self.high_cut.setRange(0.001, 0.5)
        self.high_cut.setValue(0.25)

        self.wavelength = QDoubleSpinBox()
        self.wavelength.setRange(1, 10000)
        self.wavelength.setValue(5000)

        self.deriv_order = QDoubleSpinBox()
        self.deriv_order.setRange(1, 2)
        self.deriv_order.setValue(1)

        self.z_order = QDoubleSpinBox()
        self.z_order.setRange(1, 2)
        self.z_order.setValue(1)

        self.inc = QDoubleSpinBox()
        self.inc.setRange(-90, 90)
        self.inc.setValue(90)

        self.dec = QDoubleSpinBox()
        self.dec.setRange(-90, 90)
        self.dec.setValue(0)

        self.param_layout.addRow("Cutoff Frequency", self.cutoff)
        self.param_layout.addRow("Low Cutoff", self.low_cut)
        self.param_layout.addRow("High Cutoff", self.high_cut)
        self.param_layout.addRow("Gaussian Cutoff Wavelength", self.wavelength)
        self.param_layout.addRow("Derivative Order", self.deriv_order)
        self.param_layout.addRow("Z-Derivative Order", self.z_order)
        self.param_layout.addRow("Inclination (°)", self.inc)
        self.param_layout.addRow("Declination (°)", self.dec)

        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self.apply_filter)
        self.apply_btn.setEnabled(False)

        self.reset_btn = QPushButton("Reset To Original Grid")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)

        self.save_btn = QPushButton("Save Output")
        self.save_btn.clicked.connect(self.save_output)
        self.save_btn.setEnabled(False)

        ctrl.addWidget(csv_box)
        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.filter_box)
        ctrl.addWidget(param_box)
        ctrl.addWidget(self.apply_btn)
        ctrl.addWidget(self.reset_btn)
        ctrl.addWidget(self.save_btn)
        ctrl.addStretch()

        # DISPLAY 
        self.input_label = QLabel("Input")
        self.output_label = QLabel("Output")

        for lbl in (self.input_label, self.output_label):
            lbl.setFixedSize(520, 420)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid gray")

        self.cbar_label = QLabel()
        self.cbar_label.setFixedSize(30, 420)

        self.cbar_text = QLabel()
        self.cbar_text.setStyleSheet("font-size:10px")

        cbar_layout = QHBoxLayout()
        cbar_layout.addWidget(self.cbar_label)
        cbar_layout.addWidget(self.cbar_text)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.input_label)
        img_layout.addWidget(self.output_label)
        img_layout.addLayout(cbar_layout)

        layout.addWidget(control_box, 1)
        layout.addLayout(img_layout, 3)

        self.setCentralWidget(main)
        self.update_param_visibility()

    # CSV → GRID
   
    def load_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV / TXT",
            "",
            "Data files (*.csv *.txt);;CSV (*.csv);;Text (*.txt)"
        )
        if not path:
            return

        self.df = pd.read_csv(path, sep=None, engine="python")

        if self.df.shape[1] <= 1:
            self.df = pd.read_csv(path, delim_whitespace=True)

        cols = list(self.df.columns)

        self.x_cb.clear()
        self.y_cb.clear()
        self.z_cb.clear()

        self.x_cb.addItems(cols)
        self.y_cb.addItems(cols)
        self.z_cb.addItems(cols)

    def create_grid_from_csv(self):
        if self.df is None:
            QMessageBox.warning(self, "No table loaded", "Please load a CSV or TXT file first.")
            return

        MIN_CELL = 0.01  # hard safety limit (meters or degrees)

        x = self.df[self.x_cb.currentText()].values
        y = self.df[self.y_cb.currentText()].values
        z = self.df[self.z_cb.currentText()].values

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[mask], y[mask], z[mask]

        extent_x = x.max() - x.min()
        extent_y = y.max() - y.min()

        suggested = max(extent_x, extent_y) / 1000

        cell = max(self.cell_size.value(), suggested, MIN_CELL)

        gx, gy = create_grid(x.min(), x.max(), y.min(), y.max(), cell)

        if self.grid_method.currentText() == "IDW":
            grid = idw(x, y, z, gx, gy)
        else:
            grid = griddata((x, y), z, (gx, gy),
                            method=self.grid_method.currentText())

        grid = np.nan_to_num(grid).astype(np.float32)
        grid -= np.mean(grid)

        transform = from_origin(
            x.min(), y.max(),
            cell,
            cell
        )

        self.image = grid
        self.base_image = grid.copy()
        self.working_image = grid.copy()
        self.src_profile = {
            "driver": "GTiff",
            "height": grid.shape[0],
            "width": grid.shape[1],
            "count": 1,
            "dtype": "float32",
            "transform": transform,
            "crs": None
        }

        self.display(self.prepare_display(grid), self.input_label)
        self.update_colorbar()
        self.apply_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    # RASTER LOAD
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Raster", "", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            self.src_profile = src.profile

        data = np.nan_to_num(data)
        data -= np.mean(data)

        self.image = data
        self.base_image = data.copy()
        self.working_image = data.copy()
        self.display(self.prepare_display(data), self.input_label)
        self.update_colorbar()
        self.apply_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    
    # FFT FILTER ENGINE
    
    def apply_filter(self):

        if self.working_image is None:
            QMessageBox.warning(self, "No grid loaded", "Load or create a grid before applying filters.")
            return

        img = self.working_image.copy()
        rows, cols = img.shape

        img *= np.outer(np.hanning(rows), np.hanning(cols))
        F = np.fft.fftshift(np.fft.fft2(img))

        transform = self.src_profile["transform"]
        dx = transform[0]
        dy = -transform[4]

        kx = 2 * np.pi * np.fft.fftfreq(cols, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(rows, d=dy)

        KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
        K = np.sqrt(KX**2 + KY**2)
        kmax = np.max(K)

        f = self.filter_box.currentText()

        if f == "Low Pass Filter":
            kc = self.cutoff.value() * kmax
            H = (K <= kc).astype(float)

        elif f == "High Pass Filter":
            kc = self.cutoff.value() * kmax
            H = (K >= kc).astype(float)

        elif f == "Band Pass Filter":
            k1 = self.low_cut.value() * kmax
            k2 = self.high_cut.value() * kmax
            H = ((K >= k1) & (K <= k2)).astype(float)

        elif f == "Gaussian High Pass":
            kc = 2 * np.pi / self.wavelength.value()
            H = 1.0 - np.exp(-(K**2) / (2 * kc**2))

        elif f == "Horizontal Derivative":
            n = int(self.deriv_order.value())
            H = (1j * KX) ** n

        elif f == "Vertical Derivative":
            n = int(self.deriv_order.value())
            H = (K ** n)

        elif f == "Reduction To Pole":
            I = np.deg2rad(self.inc.value())
            D = np.deg2rad(self.dec.value())

            eps = 1e-10
            denom = (
                1j * (KX * np.cos(I) * np.cos(D) +
                      KY * np.cos(I) * np.sin(D)) +
                K * np.sin(I)
            )
            denom[np.abs(denom) < eps] = eps
            H = K / denom

        elif f == "Tilt Derivative":
            n = int(self.z_order.value())

            Fx = 1j * KX * F
            Fy = 1j * KY * F
            Fz = (K ** n) * F

            dx_f = np.real(np.fft.ifft2(np.fft.ifftshift(Fx)))
            dy_f = np.real(np.fft.ifft2(np.fft.ifftshift(Fy)))
            dz_f = np.real(np.fft.ifft2(np.fft.ifftshift(Fz)))

            out = np.arctan(dz_f / (np.sqrt(dx_f**2 + dy_f**2) + 1e-10))
            self.finalize(out)
            return

        out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
        self.finalize(out)
        


    # --------------------------------------------------------
    def finalize(self, out):
        self.last_output = out
        self.working_image = out.copy()
        self.image = self.working_image
        self.display(self.prepare_display(self.working_image), self.input_label)
        self.display(self.prepare_display(out), self.output_label)
        self.update_colorbar()
        self.save_btn.setEnabled(True)

    def reset_to_original(self):
        if self.base_image is None:
            return
        self.working_image = self.base_image.copy()
        self.image = self.base_image.copy()
        self.last_output = None
        self.display(self.prepare_display(self.working_image), self.input_label)
        self.output_label.setText("Output")

    # --------------------------------------------------------
    def prepare_display(self, img):
        img = np.nan_to_num(img)
        vmin, vmax = np.percentile(img, [2, 98])
        self.last_vmin, self.last_vmax = vmin, vmax
        img = np.clip(img, vmin, vmax)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)

    def update_colorbar(self):
        grad = np.linspace(255, 0, 256).astype(np.uint8).reshape(256, 1)
        cbar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        cbar = cv2.resize(cbar, (20, 420))
        qimg = QImage(cbar.data, cbar.shape[1], cbar.shape[0],
                      cbar.strides[0], QImage.Format_BGR888)
        self.cbar_label.setPixmap(QPixmap.fromImage(qimg))
        self.cbar_text.setText(
            f"{self.last_vmax:.2f}\n\n\n0\n\n\n{self.last_vmin:.2f}"
        )

    def save_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "filtered.tif", "GeoTIFF (*.tif)")
        if not path:
            return

        profile = self.src_profile.copy()
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.last_output.astype(np.float32), 1)

    def display(self, img, label):
        qimg = QImage(img.data, img.shape[1], img.shape[0],
                      img.strides[0], QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio))


# --------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FFTGeoscience()
    win.show()
    sys.exit(app.exec_())
