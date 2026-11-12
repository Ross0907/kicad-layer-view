"""Read-only desktop application for exploring physical PCB layers."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
import time

from qtpy.QtCore import Qt, QThread, Signal, QTimer, QVariantAnimation, QEasingCurve
from qtpy.QtGui import QAction, QColor, QIcon
from qtpy.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QFrame, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from .viewer import BoardViewport

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

STYLE = """
QMainWindow, QWidget { background: #111c28; color: #dce6ef; font-size: 12px; font-family: 'Segoe UI'; }
QToolBar { background: #192a3a; border: 0; padding: 8px; spacing: 6px; }
QPushButton, QToolButton, QComboBox, QDoubleSpinBox, QLineEdit {
  background: #203345; border: 1px solid #31495d;
  border-radius: 6px; padding: 5px 8px; min-height: 22px; }
QPushButton:hover, QToolButton:hover { background: #2d485d; border-color: #64cabb; }
QPushButton:disabled { color: #718493; }
QTableWidget { background: #152534; alternate-background-color: #192c3d;
  border: 1px solid #283e51; border-radius: 6px; selection-background-color: #28534f; }
QTableWidget::item { padding: 4px; }
QHeaderView::section { background: #203344; color: #99afc1; border: 0; padding: 8px; }
QStatusBar { background: #10202c; color: #a8becd; }
QLabel#title { font-size: 23px; font-weight: 600; color: #f1f7fa; }
QLabel#subtitle { color: #9db4c5; }
QLabel#section { color: #81a1b7; font-size: 10px; font-weight: 600; letter-spacing: 2px; }
QProgressBar { border: 0; background: #203345; border-radius: 3px; max-height: 5px; }
QProgressBar::chunk { background: #6ed6c4; border-radius: 3px; }
QScrollBar:vertical { width: 8px; background: #142432; margin: 0; }
QScrollBar::handle:vertical { background: #405b6d; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSlider::groove:horizontal { height: 5px; background: #3c5263; border-radius: 2px; }
QSlider::handle:horizontal { background: #6ed6c4; width: 13px; margin: -4px 0; border-radius: 6px; }
QCheckBox { spacing: 7px; }
QSplitter::handle:horizontal { background: #293f50; margin: 0 2px; }
QSplitter::handle:horizontal:hover { background: #64cabb; }
QToolTip { background: #e2edf4; color: #172432; border: 0; }
"""


class LayerColorControl(QWidget):
    changed = Signal(object)

    def __init__(self, part, use_editor=True, parent=None):
        super().__init__(parent)
        self.part = part
        self.custom_rgb = tuple(part.color)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(4)
        self.combo = QComboBox()
        self.combo.addItems(["Material", "PCB editor", "Custom"])
        if part.kind == "dielectric":
            self.combo.model().item(1).setEnabled(False)
        self.combo.setCurrentIndex(1 if use_editor and part.kind in ("copper", "mask", "silk", "paste") else 0)
        self.swatch = QToolButton()
        self.swatch.setFixedWidth(26)
        self.swatch.setToolTip("Choose a custom layer color")
        self.swatch.clicked.connect(self.choose)
        self.combo.currentIndexChanged.connect(self.update_color)
        self.combo.activated.connect(lambda index: self.choose() if index == 2 else None)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.swatch)
        self.update_color()

    def rgb(self):
        return (self.part.color, self.part.editor_color, self.custom_rgb)[self.combo.currentIndex()]

    def update_color(self, *_):
        color = QColor.fromRgbF(*self.rgb()).name()
        self.swatch.setStyleSheet(f"QToolButton {{ background: {color}; border: 1px solid #8ea7b8; padding: 0; }}")
        self.changed.emit(self.rgb())

    def choose(self):
        color = QColorDialog.getColor(QColor.fromRgbF(*self.rgb()), self, f"{self.part.name} color")
        if color.isValid():
            self.custom_rgb = (color.redF(), color.greenF(), color.blueF())
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(2)
            self.combo.blockSignals(False)
            self.update_color()

    def use_editor(self, enabled):
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(1 if enabled and self.part.kind in ("copper", "mask", "silk", "paste") else 0)
        self.combo.blockSignals(False)
        self.update_color()


def read_saved_board(path, progress):
    from .source_cache import read_saved_board as read_cached_board
    return read_cached_board(path, progress)


class LoadWorker(QThread):
    ready = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, source, path=None, parent=None, *, via_details=False, components=True):
        super().__init__(parent)
        self.source, self.path = source, path
        self.via_details = via_details
        self.components = components

    def info(self, message):
        log.info(message)
        self.progress.emit(message)

    def warn(self, message):
        log.warning(message)
        self.progress.emit(message)

    def debug(self, message):
        log.debug(message)

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("Loading cancelled")

    def run(self):
        try:
            started = time.perf_counter()
            from .cache import save_cached_scene
            from .loading import load_complete_scene
            from .model import demo_snapshot, load_snapshot
            if self.source == "demo":
                snapshot = demo_snapshot()
            elif self.source == "snapshot":
                snapshot = load_snapshot(self.path)
            elif self.source == "board":
                snapshot = read_saved_board(self.path, self)
            else:
                from .extract import extract_from_kicad
                snapshot = extract_from_kicad(progress=self)
            self.check_cancel()
            self.info("Checking cached geometry…")
            # One complete scene: all layers and 3D models appear together.
            scene, cached = load_complete_scene(snapshot, via_details=self.via_details,
                                                components=self.components, progress=self)
            self.check_cancel()
            self.elapsed = time.perf_counter() - started
            self.cache_hit = cached
            self.reused_count = len(scene.reused_keys)
            self.ready.emit(snapshot, scene)
            self.elapsed = time.perf_counter() - started
            self.cache_hit = cached
            self.reused_count = len(getattr(scene, "reused_keys", ()))
            log.info("Board ready in %.2fs (cache=%s)", self.elapsed, cached)
            if not cached:
                # Rendering only changes actor transforms, never mesh arrays.
                # Persist the immutable scene after making it usable in the UI.
                save_cached_scene(snapshot, scene, should_cancel=self.isInterruptionRequested)
        except Exception as exc:
            log.exception("Board load failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    scene_ready = Signal()
    load_failed = Signal(str)

    def __init__(self, gap=0.0):
        super().__init__()
        self.setWindowTitle("Layer View · Exploded PCB")
        self.setWindowIcon(QIcon(str(ROOT / "resources" / "layers_48.png")))
        self.resize(1460, 920)
        self.snapshot = None
        self.scene = None
        self._board_preview = False
        self.worker = None
        self._pending_load = None
        self._closing = False
        self._component_worker = None
        self._component_signature = None
        self._component_parts = []
        self._component_warnings = []
        self._component_loaded_signature = None
        self._detail_worker = None
        self.source = ("ipc", None)
        self.rows = {}
        self._solo_key = None
        self._row_keys = []
        self.mask_opacity = .83
        self._mask_opacity_override = False
        self._opacity_overrides = {}
        self.screenshot_scale = 1
        self.render_dialog = None
        self.setStyleSheet(STYLE)
        self._toolbar()
        self.splitter = splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        self._sidebar_width = 510
        self.setCentralWidget(splitter)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 20, 18, 14)
        eyebrow = QLabel("LAYER VIEW  /  3D INSPECTOR")
        eyebrow.setObjectName("section")
        panel_header = QHBoxLayout()
        panel_header.addWidget(eyebrow, 1)
        collapse = QToolButton()
        collapse.setText("‹")
        collapse.setAccessibleName("Collapse layer sidebar")
        collapse.setToolTip("Collapse sidebar · Ctrl+Shift+L to reopen")
        collapse.clicked.connect(lambda: self.set_sidebar_visible(False))
        panel_header.addWidget(collapse)
        layout.addLayout(panel_header)
        title = QLabel("Explore your board")
        title.setObjectName("title")
        layout.addWidget(title)
        self.summary = QLabel("Open a PCB or connect to KiCad")
        self.summary.setObjectName("subtitle")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.loading = QProgressBar()
        self.loading.setRange(0, 0)
        self.loading.setTextVisible(False)
        self.loading.hide()
        layout.addWidget(self.loading)
        layout.addSpacing(15)
        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("Layer spacing"))
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 30)
        self.gap.setDecimals(2)
        self.gap.setSingleStep(0.25)
        self.gap.setSuffix(" mm")
        self.gap.setValue(gap)
        gap_row.addWidget(self.gap)
        layout.addLayout(gap_row)
        self.gap_slider = QSlider(Qt.Horizontal)
        self.gap_slider.setRange(0, 3000)
        self.gap_slider.setValue(round(gap * 100))
        layout.addWidget(self.gap_slider)
        buttons = QHBoxLayout()
        for name, target in (("Assemble", 0.0), ("Explode", 3.0)):
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, target=target: self.animate_gap(target))
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(450)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.animation.valueChanged.connect(lambda value: self.gap.setValue(float(value)))
        preset_row = QHBoxLayout()
        for label, mode in (("Copper", "copper"), ("Stackup", "stackup"), ("X-ray", "xray"), ("Vias", "vias")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, mode=mode: self.apply_preset(mode))
            preset_row.addWidget(button)
        layout.addLayout(preset_row)
        layout.addSpacing(8)
        self.kind_boxes = {}
        category_row = QHBoxLayout()
        for kind, title, checked in (("dielectric", "Dielectric", True),
                                      ("mask", "Mask", True),
                                      ("silk", "Silk", True),
                                      ("barrel", "Barrels", False),
                                      ("fill", "Fill / plugs", False)):
            box = QCheckBox(title)
            box.setChecked(checked)
            box.toggled.connect(self.apply_visibility)
            self.kind_boxes[kind] = box
            category_row.addWidget(box)
        layout.addLayout(category_row)
        detail_row = QHBoxLayout()
        for kind, title in (("component", "3D components"), ("via_patch", "POFV finish contrast"),
                            ("paste", "Solder paste")):
            box = QCheckBox(title)
            box.setChecked(kind != "paste")
            box.toggled.connect(self.apply_visibility)
            self.kind_boxes[kind] = box
            detail_row.addWidget(box)
        self.kind_boxes["via_patch"].setToolTip(
            "Show a contrasting finish over capped via bores. This is a visual inspection aid; "
            "KiCad does not record a measured surface finish.")
        layout.addLayout(detail_row)
        self.kind_boxes["paste"].setToolTip("Front/back stencil deposits from the PCB paste artwork; "
                                           "illustrative 0.08 mm thickness, not reflowed joints.")
        self.extend_vias = QCheckBox("Extend via connections across layer gaps")
        self.extend_vias.setChecked(False)
        self.extend_vias.setToolTip(
            "Extend plated barrels and filling between their real copper endpoints. "
            "Includes plated through-hole pads. This is a visual connection guide.")
        self.extend_vias.toggled.connect(
            lambda enabled: self.viewport.set_extended_connections(enabled))
        layout.addWidget(self.extend_vias)
        layout.addSpacing(10)
        self.all_colors = QCheckBox("Use PCB editor colors for copper, mask, silk and paste")
        self.all_colors.setChecked(False)
        self.all_colors.toggled.connect(self.set_all_colors)
        layout.addWidget(self.all_colors)
        layer_tools = QHBoxLayout()
        self.layer_search = QLineEdit()
        self.layer_search.setPlaceholderText("Find a layer…")
        self.layer_search.setClearButtonEnabled(True)
        self.layer_search.textChanged.connect(self.filter_layers)
        layer_tools.addWidget(self.layer_search, 1)
        self.solo_button = QPushButton("Solo")
        self.solo_button.setToolTip("Isolate the selected layer; click again to restore")
        self.solo_button.clicked.connect(self.solo_selected)
        layer_tools.addWidget(self.solo_button)
        all_layers = QPushButton("All")
        all_layers.clicked.connect(self.show_all_layers)
        layer_tools.addWidget(all_layers)
        layout.addLayout(layer_tools)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["On", "Layer", "Color", "Opacity"])
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setShowGrid(False)
        self.table.cellDoubleClicked.connect(lambda *_: self.solo_selected())
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 160)
        self.table.setColumnWidth(3, 65)
        self.table.verticalHeader().setDefaultSectionSize(42)
        layout.addWidget(self.table, 1)
        clip_form = QFormLayout()
        self.clip_axis = QComboBox()
        self.clip_axis.addItems(["Off", "X", "Y", "Z"])
        self.clip_axis.setCurrentIndex(3)
        self.clip_slider = QSlider(Qt.Horizontal)
        self.clip_slider.setRange(0, 1000)
        self.clip_slider.setValue(0)
        self.clip_flip = QCheckBox("Reverse cut")
        clip_form.addRow("Cross-section", self.clip_axis)
        clip_form.addRow("Position", self.clip_slider)
        clip_form.addRow("", self.clip_flip)
        layout.addLayout(clip_form)
        self.notices = QLabel()
        self.notices.setObjectName("subtitle")
        self.notices.setWordWrap(True)
        layout.addWidget(self.notices)
        # Keep all controls reachable at narrow widths and on short displays;
        # the content's minimum size must not lock the splitter at 475 pixels.
        self.sidebar = QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setFrameShape(QFrame.NoFrame)
        self.sidebar.setMinimumWidth(240)
        self.sidebar.setWidget(panel)
        splitter.addWidget(self.sidebar)
        self.viewport = BoardViewport()
        splitter.addWidget(self.viewport)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([510, 950])
        splitter.splitterMoved.connect(self._sidebar_moved)
        self.viewport.picked.connect(self.statusBar().showMessage)
        self.viewport.live_ray.status.connect(self.statusBar().showMessage)
        self.viewport.live_ray.enabled_changed.connect(self.live_ray_action.setChecked)
        self.connection_status = QLabel("Offline")
        self.connection_status.setContentsMargins(8, 0, 8, 0)
        self.statusBar().addPermanentWidget(self.connection_status)
        self.gap.valueChanged.connect(self._gap_changed)
        self.gap_slider.valueChanged.connect(lambda value: self.gap.setValue(value / 100))
        self.clip_axis.currentIndexChanged.connect(self.update_clip)
        self.clip_slider.valueChanged.connect(self.update_clip)
        self.clip_flip.toggled.connect(self.update_clip)
        self.statusBar().showMessage("Drag: orbit · Middle/right drag: pan · Wheel: zoom · Click XYZ: snap view")

    def _sidebar_moved(self, *_):
        width = self.splitter.sizes()[0]
        if width > 0:
            self._sidebar_width = width
        self.sidebar_action.blockSignals(True)
        self.sidebar_action.setChecked(width > 0)
        self.sidebar_action.blockSignals(False)

    def set_sidebar_visible(self, visible):
        sizes = self.splitter.sizes()
        if visible:
            self.sidebar.show()
            total = max(480, self.splitter.width() - self.splitter.handleWidth())
            width = min(max(240, self._sidebar_width), total - 240)
            self.splitter.setSizes([width, total - width])
        else:
            if sizes[0] > 0:
                self._sidebar_width = sizes[0]
            self.sidebar.hide()
        self.sidebar_action.blockSignals(True)
        self.sidebar_action.setChecked(bool(visible))
        self.sidebar_action.blockSignals(False)

    def _toolbar(self):
        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)
        self.sidebar_action = QAction("Layers", self)
        self.sidebar_action.setCheckable(True)
        self.sidebar_action.setChecked(True)
        self.sidebar_action.setShortcut("Ctrl+Shift+L")
        self.sidebar_action.setToolTip("Show or collapse the layer sidebar (Ctrl+Shift+L)")
        self.sidebar_action.toggled.connect(self.set_sidebar_visible)
        toolbar.addAction(self.sidebar_action)
        toolbar.addSeparator()
        for label, callback in (("Connect to KiCad", lambda: self.load("ipc")),
                                ("Open PCB…", self.open_board),
                                ("Open snapshot…", self.open_snapshot),
                                ("Save snapshot…", self.save_snapshot)):
            action = QAction(label, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)
        self.follow_action = QAction("Live sync", self)
        self.follow_action.setCheckable(True)
        self.follow_action.setToolTip("Follow the active PCB, including unsaved edits. Uncheck to pause.")
        self.follow_action.toggled.connect(self.set_follow)
        toolbar.addAction(self.follow_action)
        toolbar.addSeparator()
        for name in ("Iso", "Top", "Bottom", "Edge"):
            action = QAction(name, self)
            action.triggered.connect(lambda _=False, n=name: self.viewport.orient(n.lower()))
            toolbar.addAction(action)
        toolbar.addAction("Fit", lambda: self.viewport.fit())
        toolbar.addSeparator()
        ortho = QAction("Orthographic", self)
        ortho.setCheckable(True)
        ortho.setChecked(True)
        ortho.toggled.connect(lambda checked: self.viewport.set_projection(checked))
        toolbar.addAction(ortho)
        self.live_ray_action = QAction("Live ray tracing", self)
        self.live_ray_action.setCheckable(True)
        self.live_ray_action.setToolTip("Progressive GPU ray tracing; orbit in OpenGL, then refine at full resolution")
        self.live_ray_action.toggled.connect(lambda enabled: self.viewport.live_ray.set_enabled(enabled))
        toolbar.addAction(self.live_ray_action)
        toolbar.addAction("Screenshot…", self.save_screenshot)
        toolbar.addAction("Render settings…", self.show_render_settings)
        measure = QAction("Measure", self)
        measure.setCheckable(True)
        measure.setToolTip("Place two points to measure 3D distance in mm; exploded spacing is included")
        measure.toggled.connect(lambda checked: self.viewport.set_measurement(checked))
        toolbar.addAction(measure)

    def load(self, source="ipc", path=None):
        if self._closing:
            return
        for task in (self._component_worker, self._detail_worker):
            if task is not None:
                task.requestInterruption()
        # The newest request wins even if a stopped worker's queued ``finished``
        # signal has not reached the UI yet.
        self._pending_load = (source, path)
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.statusBar().showMessage("Switching board source…")
            return
        if self.worker is not None:
            self.worker.deleteLater()
        self._pending_load = None
        self.source = (source, path)
        self.follow_action.blockSignals(True)
        self.follow_action.setChecked(source == "ipc")
        self.follow_action.blockSignals(False)
        if source == "ipc":
            from .live_worker import LiveWorker
            self.worker = LiveWorker(self, via_details=self._via_details_enabled())
            self.worker.components = self.kind_boxes["component"].isChecked()
            self.worker.state.connect(self._worker_state)
            self.worker.failed.connect(self.live_failed)
        else:
            self.worker = LoadWorker(source, path, self, via_details=self._via_details_enabled(),
                                     components=self.kind_boxes["component"].isChecked())
            self.worker.failed.connect(self.failed)
            self.connection_status.setText("Saved PCB · offline" if source == "board" else "Snapshot · offline")
        self.worker.progress.connect(self._worker_progress)
        self.worker.ready.connect(self.loaded)
        self.worker.finished.connect(self._worker_finished)
        self.statusBar().showMessage("Loading board…")
        self.loading.show()
        if source != "demo":
            # Start mesh workers now so they are ready when the board has been read.
            from . import mesh_pool
            mesh_pool.warm_up()
        self.worker.start(QThread.LowPriority)

    def _worker_finished(self):
        worker = self.sender()
        if worker is not self.worker:
            # A newer request can already have replaced this completed worker.
            return
        self.worker = None
        worker.deleteLater()
        if self._pending_load is not None and not self._closing:
            source, path = self._pending_load
            self._pending_load = None
            self.load(source, path)
        elif not self._closing:
            self._ensure_via_details()

    def _current_worker_signal(self):
        sender = self.sender()
        return (not self._closing and self._pending_load is None
                and (sender is None or sender is self.worker))

    def _worker_progress(self, message):
        if self._current_worker_signal():
            self.statusBar().showMessage(message)

    def _worker_state(self, message):
        if self._current_worker_signal():
            self.connection_status.setText(message)

    def set_follow(self, enabled):
        from .live_worker import LiveWorker
        if isinstance(self.worker, LiveWorker) and self.worker.isRunning():
            self.worker.set_paused(not enabled)
        elif enabled:
            self.load("ipc")

    def live_failed(self, message):
        if not self._current_worker_signal():
            return
        self.loading.hide()
        self.connection_status.setToolTip(message)
        self.statusBar().showMessage("Open a PCB in KiCad with its API enabled; Layer View will reconnect.")
        # Connection interruptions are normal when the editor closes/switches
        # boards. Do not interrupt the user with a repeating modal dialog.
        if getattr(self, "automation", False):
            self.load_failed.emit(message)

    def loaded(self, snapshot, scene):
        if not self._current_worker_signal():
            return
        from .live_worker import board_identity
        preserve = (self.snapshot is not None and board_identity(snapshot) == board_identity(self.snapshot)
                    and (self.source[0] == "ipc" or snapshot is self.snapshot))
        previous_controls = self.layer_controls() if preserve else {}
        if not preserve:
            self._opacity_overrides.clear()
        self._board_preview = getattr(scene, "_preview", False)
        from .components import component_signature
        signature = component_signature(snapshot)
        if signature != self._component_signature:
            self._component_signature = signature
            self._component_loaded_signature = None
            self._component_parts = []
            self._component_warnings = []
            if self._component_worker is not None:
                self._component_worker.requestInterruption()
        delivered = getattr(scene, "_components", None)
        if delivered is not None and delivered[2] == signature:
            # The load worker built the models alongside the board.
            self._component_parts, self._component_warnings = list(delivered[0]), list(delivered[1])
            self._component_loaded_signature = signature
        if self._component_parts or self._component_warnings:
            scene = replace(scene, parts=list(scene.parts) + self._component_parts,
                            warnings=list(dict.fromkeys(scene.warnings + self._component_warnings)))
        self.snapshot, self.scene = snapshot, scene
        if not preserve or self._solo_key not in {part.key for part in scene.parts}:
            self._solo_key = None
            self.solo_button.setText("Solo")
        self.loading.hide()
        self.viewport.gap_mm = self.gap.value()
        self.viewport.set_scene(scene, snapshot,
                                {kind: box.isChecked() for kind, box in self.kind_boxes.items()},
                                preserve_view=preserve)
        self.populate_layers(previous_controls)
        self.apply_visibility()
        self.update_clip()
        if not preserve:
            self.viewport.fit()
        count = len(snapshot.stackup.copper)
        self.summary.setText(f"{snapshot.board_name}\n{count} copper layers · "
                             f"{snapshot.stackup.board_thickness_mm:.3f} mm · "
                             f"{len(snapshot.vias):,} vias\n"
                             f"{sum(v.filled for v in snapshot.vias):,} filled · "
                             f"{sum(v.capped for v in snapshot.vias):,} capped · "
                             f"{sum(v.front_plugged or v.back_plugged for v in snapshot.vias):,} plugged")
        self.setWindowTitle(f"{snapshot.board_name} · Layer View")
        warnings = getattr(scene, "warnings", []) or getattr(snapshot, "warnings", [])
        if snapshot.silkscreen is None:
            warnings = ["This older snapshot has no silkscreen. Reopen the PCB or reload from KiCad.", *warnings]
        self.notices.setText("\n".join(str(w) for w in warnings[:3]))
        self.notices.setToolTip("\n".join(str(w) for w in warnings))
        elapsed = getattr(self.worker, "elapsed", 0)
        source = "cached" if getattr(self.worker, "cache_hit", False) else "built"
        if preserve or (not getattr(self.worker, "cache_hit", False)
                        and getattr(self.worker, "reused_count", 0)):
            source = f"{getattr(self.worker, 'reused_count', 0)}/{len(scene.parts)} meshes reused"
        self.statusBar().showMessage(f"Ready in {elapsed:.1f}s · {source} geometry · "
                                     "Drag: orbit · Middle/right: pan · Wheel: zoom · XYZ: snap")
        if self._board_preview:
            self.statusBar().showMessage(f"Board surface ready in {elapsed:.1f}s · preparing inner layers…")
        self._ensure_components()
        self.scene_ready.emit()

    def layer_controls(self):
        return {key: (visible.isChecked(), color.combo.currentIndex(), color.custom_rgb, opacity.value())
                for key, (visible, color, opacity) in self.rows.items()}

    def _via_details_enabled(self):
        return any(self.kind_boxes[key].isChecked() for key in ("barrel", "fill"))

    def _ensure_via_details(self):
        if self._closing or self.snapshot is None or self._pending_load is not None:
            return
        enabled = self._via_details_enabled()
        if self.worker is not None:
            if hasattr(self.worker, "set_via_details"):
                self.worker.set_via_details(enabled)
            elif hasattr(self.worker, "via_details"):
                self.worker.via_details = enabled
            if self.worker.isRunning():
                return
        if not enabled:
            if self._detail_worker is not None:
                self._detail_worker.requestInterruption()
            return
        from .geometry import scene_keys
        if scene_keys(self.snapshot) <= self.scene._geometry_signatures.keys():
            return
        if self._detail_worker is not None:
            if self._detail_worker.snapshot is not self.snapshot:
                self._detail_worker.requestInterruption()
            return
        from .scene_tasks import ViaDetailTask
        task = self._detail_worker = ViaDetailTask(self.snapshot, self.scene, self)
        task.ready.connect(self._details_ready)
        task.failed.connect(lambda message: self.statusBar().showMessage(f"Via geometry: {message}"))
        task.finished.connect(self._details_finished)
        task.start(QThread.LowPriority)

    def _details_ready(self, snapshot, scene):
        if (self._closing or self._pending_load is not None or snapshot is not self.snapshot
                or self.sender() is not self._detail_worker
                or self._detail_worker.isInterruptionRequested()):
            return
        self._refresh_extra_scene(replace(scene, parts=list(scene.parts) + self._component_parts,
                                  warnings=list(dict.fromkeys(scene.warnings + self._component_warnings))))

    def _details_finished(self):
        task = self.sender()
        if task is self._detail_worker:
            self._detail_worker = None
            task.deleteLater()
            # Retry only if the board changed while the old request was running.
            if task.snapshot is not self.snapshot or task.isInterruptionRequested():
                self._ensure_via_details()

    def _ensure_components(self):
        if (self._closing or self.snapshot is None or self._pending_load is not None
                or not self.kind_boxes["component"].isChecked()):
            return
        if self._component_signature == self._component_loaded_signature:
            return
        if not any(footprint.models for footprint in self.snapshot.footprints):
            self._component_loaded_signature = self._component_signature
            return
        if self._component_worker is not None:
            return
        from .scene_tasks import ComponentTask
        task = self._component_worker = ComponentTask(self.snapshot, self._component_signature,
                                                    self.scene.slot_count, self)
        task.ready.connect(self._components_ready)
        task.failed.connect(lambda message: self.statusBar().showMessage(f"3D components: {message}"))
        task.finished.connect(self._components_finished)
        task.start(QThread.LowPriority)

    def _components_ready(self, snapshot, result):
        task = self.sender()
        if (self._closing or self._pending_load is not None or task is not self._component_worker
                or task.signature != self._component_signature or task.isInterruptionRequested()):
            return
        parts, warnings = result
        self._component_parts = parts
        self._component_warnings = warnings
        self._component_loaded_signature = task.signature
        scene = replace(self.scene,
                        parts=[p for p in self.scene.parts if p.kind != "component"] + parts,
                        warnings=list(dict.fromkeys(self.scene.warnings + warnings)))
        self._refresh_extra_scene(scene)
        self.notices.setText("\n".join(scene.warnings[:3]))
        self.notices.setToolTip("\n".join(scene.warnings))
        self.statusBar().showMessage("3D components ready")

    def _components_finished(self):
        task = self.sender()
        if task is self._component_worker:
            self._component_worker = None
            task.deleteLater()
            if task.signature != self._component_signature or task.isInterruptionRequested():
                self._ensure_components()

    def _refresh_extra_scene(self, scene):
        previous = self.layer_controls()
        self.scene = scene
        self.viewport.set_scene(scene, self.snapshot,
                                {kind: box.isChecked() for kind, box in self.kind_boxes.items()},
                                preserve_view=True)
        self.populate_layers(previous)
        self.apply_visibility()
        self.update_clip()

    def populate_layers(self, previous=None):
        previous = previous or {}
        current_row = self.table.currentRow()
        selected = (self._row_keys[current_row]
                    if previous and 0 <= current_row < len(self._row_keys) else None)
        scroll_position = self.table.verticalScrollBar().value()
        self.rows.clear()
        parts = [p for p in self.scene.parts if p.kind in ("copper", "dielectric", "mask", "silk", "paste")]
        parts.sort(key=lambda p: (p.stack_index, p.name))
        self._row_keys = [part.key for part in parts]
        self.table.setRowCount(len(parts))
        for row, part in enumerate(parts):
            visible = QCheckBox()
            saved = previous.get(part.key)
            visible.setChecked(saved[0] if saved else True)
            visible.toggled.connect(self.apply_visibility)
            self.table.setCellWidget(row, 0, visible)
            label = QTableWidgetItem(part.name)
            label.setForeground(QColor.fromRgbF(*part.editor_color)
                                if part.kind in ("copper", "mask", "silk", "paste") else QColor("#b9cbd8"))
            label.setToolTip(f"{part.kind} · {part.area_mm2:,.2f} mm²")
            self.table.setItem(row, 1, label)
            color = LayerColorControl(part, use_editor=self.all_colors.isChecked())
            if saved:
                color.custom_rgb = saved[2]
                color.combo.setCurrentIndex(saved[1])
                color.update_color()
            color.changed.connect(lambda rgb, key=part.key: self.viewport.set_part(key, color=rgb))
            self.table.setCellWidget(row, 2, color)
            opacity = QDoubleSpinBox()
            opacity.setRange(0.0, 1.0)
            opacity.setSingleStep(0.1)
            opacity.setDecimals(2)
            opacity.setValue(self._layer_opacity(part))
            opacity.setButtonSymbols(QAbstractSpinBox.NoButtons)
            opacity.valueChanged.connect(
                lambda value, key=part.key: self._set_layer_opacity(key, value))
            self.table.setCellWidget(row, 3, opacity)
            self.rows[part.key] = (visible, color, opacity)
            self.viewport.set_part(part.key, color=color.rgb(), opacity=opacity.value(), render=False)
        self.filter_layers(self.layer_search.text())
        if selected in self._row_keys:
            self.table.selectRow(self._row_keys.index(selected))
        if previous:
            self.table.verticalScrollBar().setValue(scroll_position)

    def set_all_colors(self, enabled):
        for key, (_, color, _) in self.rows.items():
            color.blockSignals(True)
            color.use_editor(enabled)
            color.blockSignals(False)
            self.viewport.set_part(key, color=color.rgb(), render=False)
        self.viewport.refresh()

    def apply_visibility(self, *_):
        if self.scene is None:
            return
        self.viewport.show_via_patches = self.kind_boxes["via_patch"].isChecked()
        for part in self.scene.parts:
            category = "fill" if part.kind == "plug" else part.kind
            enabled = self.kind_boxes[category].isChecked() if category in self.kind_boxes else True
            if part.key in self.rows:
                enabled = enabled and self.rows[part.key][0].isChecked()
            if part.kind == "via_patch":
                parent = self.rows.get(f"copper:{part.layer_id}")
                # Contrast is optional, but a capped bore remains physically
                # closed when its finish is shown in ordinary copper color.
                enabled = parent is None or parent[0].isChecked()
                self.viewport.set_part(f"copper:{part.layer_id}", render=False)
            if self._solo_key:
                enabled = part.key == self._solo_key or (part.kind == "via_patch"
                          and self._solo_key == f"copper:{part.layer_id}" and enabled)
            self.viewport.set_part(part.key, visible=enabled, render=False)
        self.viewport.refresh()
        self._ensure_components()
        self._ensure_via_details()

    def filter_layers(self, text):
        for index in range(self.table.rowCount()):
            item = self.table.item(index, 1)
            self.table.setRowHidden(index, text.casefold() not in item.text().casefold())

    def solo_selected(self):
        if self._solo_key:
            self._solo_key = None
        elif 0 <= self.table.currentRow() < len(self._row_keys):
            self._solo_key = self._row_keys[self.table.currentRow()]
        self.solo_button.setText("Restore" if self._solo_key else "Solo")
        self.apply_visibility()

    def show_all_layers(self):
        self._solo_key = None
        self.solo_button.setText("Solo")
        for visible, _, _ in self.rows.values():
            visible.blockSignals(True)
            visible.setChecked(True)
            visible.blockSignals(False)
        self.apply_visibility()

    def apply_preset(self, mode):
        self._solo_key = None
        self.solo_button.setText("Solo")
        use_editor = self.all_colors.isChecked()
        for kind, box in self.kind_boxes.items():
            box.blockSignals(True)
            if kind != "paste":
                box.setChecked(mode not in ("copper", "vias") or kind in ("barrel", "fill"))
            box.blockSignals(False)
        for key, (visible, color, opacity) in self.rows.items():
            part = self.viewport.parts[key]
            visible.blockSignals(True)
            visible.setChecked(mode != "vias")
            visible.blockSignals(False)
            color.blockSignals(True)
            color.use_editor(use_editor)
            color.blockSignals(False)
            opacity.blockSignals(True)
            self._opacity_overrides.pop(key, None)
            value = (.14 if mode == "xray" and part.kind in ("dielectric", "mask")
                     else self._layer_opacity(part))
            if mode == "xray" and part.kind in ("dielectric", "mask"):
                self._opacity_overrides[key] = value
            opacity.setValue(value)
            opacity.blockSignals(False)
            self.viewport.set_part(key, color=color.rgb(), opacity=value, render=False)
        self.apply_visibility()

    def _layer_opacity(self, part):
        """Material alpha follows live edits unless the user explicitly overrides it."""
        if part.key in self._opacity_overrides:
            return self._opacity_overrides[part.key]
        if part.kind == "mask" and self._mask_opacity_override:
            return self.mask_opacity
        return part.opacity

    def _set_layer_opacity(self, key, value):
        self._opacity_overrides[key] = value
        self.viewport.set_part(key, opacity=value)

    def set_mask_opacity(self, value):
        self.mask_opacity = value
        self._mask_opacity_override = True
        for key, (_, _, opacity) in self.rows.items():
            if self.viewport.parts[key].kind == "mask":
                self._opacity_overrides[key] = value
                opacity.blockSignals(True)
                opacity.setValue(value)
                opacity.blockSignals(False)
                self.viewport.set_part(key, opacity=value, render=False)
        self.viewport.refresh()

    def show_render_settings(self):
        if self.render_dialog is None:
            from .render_dialog import RenderDialog
            self.render_dialog = RenderDialog(self)
        self.render_dialog.show()
        self.render_dialog.raise_()

    def _gap_changed(self, value):
        self.gap_slider.blockSignals(True)
        self.gap_slider.setValue(round(value * 100))
        self.gap_slider.blockSignals(False)
        self.viewport.set_gap(value)

    def animate_gap(self, target):
        self.animation.stop()
        self.animation.setStartValue(self.gap.value())
        self.animation.setEndValue(target)
        self.animation.start()
        QTimer.singleShot(500, lambda: None if self._closing else self.viewport.fit())

    def update_clip(self, *_):
        index = self.clip_axis.currentIndex()
        self.viewport.set_clip(index - 1 if index else None,
                               self.clip_slider.value() / 1000, self.clip_flip.isChecked())

    def failed(self, message):
        if not self._current_worker_signal():
            return
        self.loading.hide()
        self.statusBar().showMessage("Could not load board")
        self.load_failed.emit(message)
        if not getattr(self, "automation", False):
            QMessageBox.warning(self, "Layer View", message)

    def open_board(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PCB (read only)", "", "KiCad PCB (*.kicad_pcb)")
        if path:
            self.load("board", path)

    def open_snapshot(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open board snapshot", "", "Snapshots (*.json *.json.gz)")
        if path:
            self.load("snapshot", path)

    def save_snapshot(self):
        if self.snapshot is None:
            return
        from .model import save_snapshot
        path, _ = QFileDialog.getSaveFileName(self, "Save viewer snapshot", "board.layers.json.gz",
                                             "Compressed snapshot (*.json.gz)")
        if path:
            try:
                save_snapshot(self.snapshot, path)
            except Exception as exc:
                QMessageBox.warning(self, "Snapshot failed", str(exc))

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D view", "layers.png", "PNG image (*.png)")
        if path:
            try:
                self.viewport.screenshot(path, scale=self.screenshot_scale)
            except Exception as exc:
                QMessageBox.warning(self, "Screenshot failed", str(exc))

    def closeEvent(self, event):
        self._closing = True
        self._pending_load = None
        self.animation.stop()
        self.viewport.camera_animation.stop()
        self.viewport.idle_aa.shutdown()
        self.viewport.render_timer.stop()
        waiting = not self.viewport.live_ray.shutdown()
        export = self.render_dialog.worker if self.render_dialog is not None else None
        # Cancel every task on the first close request. Waiting for one worker
        # first otherwise lets the others keep polling, meshing or exporting.
        for task in (self.worker, self._component_worker, self._detail_worker, export):
            if task is not None and task.isRunning():
                task.requestInterruption()
                waiting = True
        if waiting:
            self.statusBar().showMessage("Stopping background work before closing…")
            QTimer.singleShot(150, self.close)
            event.ignore()
            return
        server = getattr(self, "instance_server", None)
        if server is not None:
            server.close()
            server.instance_lock.unlock()
        if self.render_dialog is not None:
            self.render_dialog.close()
        from . import mesh_pool
        mesh_pool.shutdown()
        self.viewport.plotter.close()
        super().closeEvent(event)
