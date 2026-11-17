import json
import importlib
import os
import shutil
import sys
import uuid
from pathlib import Path

from PyQt5.QtCore import QPointF, QThread, Qt
from PyQt5.QtGui import QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QInputDialog,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QHeaderView,
    QSizePolicy,
    QProgressBar,
    QSplitterHandle,
    QScrollArea,
    QFrame,
)

from .config import DATA_ROOT, DB_PATH, INFERENCE_MODES, PROJECT_ROOT, TASKS_ROOT, ensure_app_dirs
from .database import TaskStore
from .inference import export_csv, export_geojson
from .preprocessing import PreprocessOptions, apply_preprocessing, read_image, write_image
from .widgets import MapCanvas
from .worker import InferenceWorker


APP_STYLESHEET = """
QMainWindow, QWidget { background: #f4f7fb; color: #172033; font-family: "Microsoft YaHei"; font-size: 13px; }
QWidget#sidePanel, QWidget#resultPanel { background: #ffffff; border: 1px solid #e4e9f2; border-radius: 14px; }
QLabel#appTitle { color: #172033; font-size: 24px; font-weight: 700; }
QLabel#appSubtitle { color: #718096; font-size: 12px; }
QLabel#productBadge { color: #2563eb; background: #edf4ff; border: 1px solid #cfe0ff; border-radius: 9px; padding: 4px 9px; }
QLabel#zoomLabel { color: #667085; min-width: 48px; font-weight: 600; }
QLabel#canvasHint { color: #98a2b3; padding-left: 6px; }
QPushButton#zoomButton { min-width: 30px; padding: 4px 8px; font-size: 15px; font-weight: 700; }
QGroupBox { background: #fbfcfe; border: 1px solid #e4e9f2; border-radius: 11px; margin-top: 13px; padding: 14px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #344054; background: #fbfcfe; }
QPushButton { background: #ffffff; border: 1px solid #d8e0ec; border-radius: 8px; padding: 8px 11px; color: #344054; }
QPushButton:hover { border-color: #7aa7f7; background: #f4f8ff; }
QPushButton:pressed { background: #e8f0ff; }
QPushButton:disabled { color: #a0aec0; background: #f5f7fa; border-color: #e7ebf2; }
QPushButton#primaryButton { background: #2563eb; border: none; color: white; font-weight: 700; padding: 10px; }
QPushButton#primaryButton:hover { background: #1d4ed8; }
QPushButton#stopButton { color: #b42318; }
QComboBox, QLineEdit { background: #ffffff; border: 1px solid #d8e0ec; border-radius: 7px; padding: 7px 9px; min-height: 18px; }
QComboBox:focus, QLineEdit:focus { border: 1px solid #70a2f8; }
QComboBox QAbstractItemView { background: white; selection-background-color: #e8f0ff; selection-color: #172033; }
QCheckBox { spacing: 8px; padding: 3px 0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QTableWidget { background: #ffffff; border: 1px solid #e4e9f2; border-radius: 9px; gridline-color: #edf0f5; alternate-background-color: #fafcff; }
QHeaderView::section { background: #f4f7fb; color: #667085; border: none; border-bottom: 1px solid #e4e9f2; padding: 8px; font-weight: 600; }
QListWidget { background: #172033; color: #dbe7ff; border: none; border-radius: 9px; padding: 7px; }
QProgressBar { background: #e9eef6; border: none; border-radius: 6px; height: 12px; text-align: center; color: #344054; }
QProgressBar::chunk { background: #2563eb; border-radius: 6px; }
QStatusBar { background: #eef3f9; color: #667085; border-top: 1px solid #dde5f0; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #eef2f7; width: 10px; margin: 2px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #b8c4d6; min-height: 35px; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #8ea3c0; }
QSplitter::handle:horizontal { background: #d8e0ec; width: 12px; }
QSplitter::handle:horizontal:hover { background: #9bbcf7; }
"""


class ResizeHandle(QSplitterHandle):
    """Visible splitter handle with chevrons so users can resize the panels."""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window().color())
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.palette().mid().color(), 1.5))
        center_x = self.width() // 2
        center_y = self.height() // 2
        for offset in (-7, 7):
            y = center_y + offset
            painter.drawLine(center_x - 3, y - 3, center_x - 6, y)
            painter.drawLine(center_x - 6, y, center_x - 3, y + 3)
            painter.drawLine(center_x + 3, y - 3, center_x + 6, y)
            painter.drawLine(center_x + 6, y, center_x + 3, y + 3)


class ResizableSplitter(QSplitter):
    def createHandle(self):
        return ResizeHandle(self.orientation(), self)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.setWindowTitle("历史地图地名提取平台")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 700)
        self.store = TaskStore(DB_PATH)
        self.backend = None
        self.source_path: Path | None = None
        self.current_image = None
        self.current_processed_path: Path | None = None
        self.current_result_path: Path | None = None
        self.current_task_id: str | None = None
        self.detections: list[dict] = []
        self.cuda_backend = None
        self.cpu_backend = None
        self.openvino_backend = None
        self._updating_table = False
        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None
        self._build_ui()

    def _build_ui(self):
        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(True)
        root = QWidget()
        root_layout = QHBoxLayout(root)

        left = QWidget()
        left.setObjectName("sidePanel")
        left.setMinimumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 18, 16)
        left_layout.setSpacing(10)
        title = QLabel("历史地图识别平台")
        title.setObjectName("appTitle")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        subtitle = QLabel("历史地图 · 地名检测与交互校正")
        subtitle.setObjectName("appSubtitle")
        badge = QLabel("本地运行 · GPU 优先")
        badge.setObjectName("productBadge")
        badge.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)
        left_layout.addWidget(badge)

        self.import_button = QPushButton("导入历史地图")
        self.import_button.clicked.connect(self.import_image)
        left_layout.addWidget(self.import_button)
        self.file_label = QLabel("尚未选择图像")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color:#475467;")
        left_layout.addWidget(self.file_label)

        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QVBoxLayout(preprocess_group)
        self.gray_check = QCheckBox("灰度化")
        self.clahe_check = QCheckBox("局部对比度增强（CLAHE）")
        self.denoise_check = QCheckBox("去噪")
        self.sharpen_check = QCheckBox("锐化")
        self.threshold_check = QCheckBox("OTSU 二值化")
        for check in (self.gray_check, self.clahe_check, self.denoise_check, self.sharpen_check, self.threshold_check):
            preprocess_layout.addWidget(check)
            check.stateChanged.connect(self.preview_preprocessing)
        self.preview_button = QPushButton("预览预处理")
        self.preview_button.clicked.connect(self.preview_preprocessing)
        preprocess_layout.addWidget(self.preview_button)
        preprocess_group.setMinimumHeight(205)
        preprocess_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_layout.addWidget(preprocess_group)

        inference_group = QGroupBox("识别设置")
        inference_layout = QFormLayout(inference_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(INFERENCE_MODES.keys())
        inference_layout.addRow("滑窗/窗口尺寸", self.mode_combo)
        self.backend_combo = QComboBox()
        backend_options = [
            "自动选择（GPU优先）",
            "Windows GPU（CUDA）",
            "Windows CPU（OpenVINO）",
            "Windows CPU（TorchScript）",
        ]
        if os.environ.get("HMP_SHOW_RESEARCH_BACKEND") == "1":
            backend_options.append("研究后端（仅调试）")
        self.backend_combo.addItems(backend_options)
        inference_layout.addRow("推理后端", self.backend_combo)
        self.display_combo = QComboBox()
        self.display_combo.addItems(["框外文字", "干净模式（只显示框）"])
        self.display_combo.currentIndexChanged.connect(self.change_display_mode)
        inference_layout.addRow("显示模式", self.display_combo)
        inference_group.setMinimumHeight(165)
        inference_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_layout.addWidget(inference_group)

        self.run_button = QPushButton("开始地名提取")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.start_inference)
        left_layout.addWidget(self.run_button)
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_inference)
        left_layout.addWidget(self.stop_button)

        progress_group = QGroupBox("任务进度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_label = QLabel("准备就绪")
        self.progress_label.setStyleSheet("color:#667085; font-weight:400;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setMinimumHeight(88)
        progress_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_layout.addWidget(progress_group)

        export_group = QGroupBox("结果导出")
        export_layout = QVBoxLayout(export_group)
        self.export_json_button = QPushButton("导出 JSON")
        self.export_geojson_button = QPushButton("导出 GeoJSON（像素坐标）")
        self.export_csv_button = QPushButton("导出 CSV")
        self.merge_button = QPushButton("合并选中地名")
        self.add_button = QPushButton("手动添加地名")
        self.history_button = QPushButton("任务历史")
        self.export_json_button.clicked.connect(self.export_json)
        self.export_geojson_button.clicked.connect(self.export_geojson)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.merge_button.clicked.connect(self.merge_selected)
        self.add_button.clicked.connect(self.start_add_mode)
        self.history_button.clicked.connect(self.show_history)
        for button in (
            self.export_json_button,
            self.export_geojson_button,
            self.export_csv_button,
            self.merge_button,
            self.add_button,
        ):
            button.setEnabled(False)
            button.setMinimumHeight(34)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            export_layout.addWidget(button)
        self.history_button.setMinimumHeight(34)
        self.history_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        export_layout.addWidget(self.history_button)
        export_group.setMinimumHeight(285)
        export_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_layout.addWidget(export_group)
        left_layout.addStretch(1)

        self.canvas = MapCanvas()
        self.canvas.detection_clicked.connect(self.select_detection)
        self.canvas.rectangle_created.connect(self.add_detection_from_rect)

        canvas_panel = QWidget()
        canvas_panel.setObjectName("canvasPanel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.setSpacing(6)
        canvas_toolbar = QHBoxLayout()
        canvas_hint = QLabel("滚轮缩放 · 双击放大 · 拖拽平移")
        canvas_hint.setObjectName("canvasHint")
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoomButton")
        self.zoom_out_button.setToolTip("缩小地图")
        self.zoom_out_button.clicked.connect(self.canvas.zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.setObjectName("zoomButton")
        self.fit_button.setToolTip("让地图重新适应当前窗口")
        self.fit_button.clicked.connect(self.canvas.fit_image)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoomButton")
        self.zoom_in_button.setToolTip("放大地图")
        self.zoom_in_button.clicked.connect(self.canvas.zoom_in)
        self.canvas.zoom_changed.connect(self.update_zoom_label)
        canvas_toolbar.addWidget(canvas_hint)
        canvas_toolbar.addStretch(1)
        canvas_toolbar.addWidget(self.zoom_out_button)
        canvas_toolbar.addWidget(self.zoom_label)
        canvas_toolbar.addWidget(self.fit_button)
        canvas_toolbar.addWidget(self.zoom_in_button)
        canvas_layout.addLayout(canvas_toolbar)
        canvas_layout.addWidget(self.canvas, 1)

        right = QWidget()
        right.setObjectName("resultPanel")
        right.setMinimumWidth(390)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(9)
        result_title = QLabel("识别结果")
        result_title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        right_layout.addWidget(result_title)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索地名…")
        self.search_edit.textChanged.connect(self.filter_results)
        right_layout.addWidget(self.search_edit)
        self.summary_label = QLabel("等待识别")
        self.summary_label.setStyleSheet("color:#667085; padding:3px 0;")
        right_layout.addWidget(self.summary_label)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["序号", "地名", "检测", "识别"])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.result_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.result_table.setWordWrap(False)
        self.result_table.setTextElideMode(Qt.ElideRight)
        self.result_table.verticalHeader().setDefaultSectionSize(30)
        self.result_table.itemSelectionChanged.connect(self.table_selection_changed)
        self.result_table.itemChanged.connect(self.table_item_changed)
        result_header = self.result_table.horizontalHeader()
        result_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        result_header.setSectionResizeMode(1, QHeaderView.Stretch)
        result_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        result_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.result_table.setMinimumHeight(260)
        right_layout.addWidget(self.result_table, 1)
        log_title = QLabel("运行日志")
        log_title.setStyleSheet("font-weight:bold;")
        right_layout.addWidget(log_title)
        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(170)
        right_layout.addWidget(self.log_list)

        splitter = ResizableSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setHandleWidth(14)
        splitter.setChildrenCollapsible(False)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("leftScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 820, 410])
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def log(self, message: str):
        self.log_list.addItem(message)
        self.log_list.scrollToBottom()
        self.statusBar().showMessage(message[:180])

    def update_zoom_label(self, factor: float):
        self.zoom_label.setText(f"{round(float(factor) * 100)}%")

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择历史地图",
            str(PROJECT_ROOT),
            "地图图像 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)",
        )
        if not path:
            return
        self.source_path = Path(path)
        self.file_label.setText(str(self.source_path))
        self.current_image = read_image(self.source_path)
        self.canvas.set_image_array(self.current_image)
        self.detections = []
        self.result_table.setRowCount(0)
        self.summary_label.setText("已导入，等待识别")
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备就绪")
        self._set_export_enabled(False)
        self.log(f"已导入：{self.source_path.name}")

    def options(self) -> PreprocessOptions:
        return PreprocessOptions(
            grayscale=self.gray_check.isChecked(),
            clahe=self.clahe_check.isChecked(),
            denoise=self.denoise_check.isChecked(),
            sharpen=self.sharpen_check.isChecked(),
            threshold=self.threshold_check.isChecked(),
        )

    def preview_preprocessing(self):
        if self.source_path is None:
            return
        try:
            self.current_image = read_image(self.source_path)
            processed = apply_preprocessing(self.current_image, self.options())
            self.canvas.set_image_array(processed)
            self.log("已更新预处理预览")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "预处理失败", str(exc))

    def change_display_mode(self):
        self.canvas.set_display_mode("clean" if self.display_combo.currentIndex() == 1 else "outside")

    def start_inference(self):
        if self.source_path is None:
            QMessageBox.information(self, "提示", "请先导入一张历史地图。")
            return
        if self.thread is not None:
            return
        try:
            task_id = uuid.uuid4().hex
            task_dir = TASKS_ROOT / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            processed_path = task_dir / "preprocessed.png"
            processed = apply_preprocessing(read_image(self.source_path), self.options())
            write_image(processed_path, processed)
            result_path = task_dir / "predictions.json"
            mode_name = self.mode_combo.currentText()
            mode = INFERENCE_MODES[mode_name]
            backend_name = self.backend_combo.currentText()
            backend_index = self.backend_combo.currentIndex()
            if backend_index == 0:
                try:
                    import torch
                    use_cuda = torch.cuda.is_available()
                except ModuleNotFoundError:
                    use_cuda = False
                if use_cuda:
                    backend_name = "Windows GPU（CUDA，自动选择）"
                    if self.cuda_backend is None:
                        from .cuda_inference import TorchScriptCudaBackend
                        self.cuda_backend = TorchScriptCudaBackend(PROJECT_ROOT)
                    inference_backend = self.cuda_backend
                else:
                    backend_name = "Windows CPU（OpenVINO，自动回退）"
                    if self.openvino_backend is None:
                        from .openvino_inference import OpenVINOCpuBackend
                        self.openvino_backend = OpenVINOCpuBackend(PROJECT_ROOT)
                    inference_backend = self.openvino_backend
            elif backend_index == 1:
                if self.cuda_backend is None:
                    try:
                        from .cuda_inference import TorchScriptCudaBackend
                    except ModuleNotFoundError as exc:
                        if exc.name == "torch":
                            raise RuntimeError(
                                "未安装 PyTorch。请先运行 .\\scripts\\setup_windows_gpu.ps1"
                            ) from exc
                        raise
                    self.cuda_backend = TorchScriptCudaBackend(PROJECT_ROOT)
                inference_backend = self.cuda_backend
            elif backend_index == 2:
                if self.openvino_backend is None:
                    try:
                        from .openvino_inference import OpenVINOCpuBackend
                    except ModuleNotFoundError as exc:
                        if exc.name in {"openvino", "numpy"}:
                            raise RuntimeError(
                                "未安装 OpenVINO，请先运行："
                                ".\\scripts\\setup_windows_cpu.ps1"
                            ) from exc
                        raise
                    self.openvino_backend = OpenVINOCpuBackend(PROJECT_ROOT)
                inference_backend = self.openvino_backend
            elif backend_index == 3:
                if self.cpu_backend is None:
                    try:
                        from .cpu_inference import TorchScriptCpuBackend
                    except ModuleNotFoundError as exc:
                        if exc.name == "torch":
                            raise RuntimeError(
                                "未安装 Windows CPU 版 PyTorch，请先运行："
                                ".\\scripts\\setup_windows_cpu.ps1"
                            ) from exc
                        raise

                    self.cpu_backend = TorchScriptCpuBackend(PROJECT_ROOT)
                inference_backend = self.cpu_backend
            else:
                if os.environ.get("HMP_SHOW_RESEARCH_BACKEND") != "1":
                    raise RuntimeError("研究后端仅在调试环境中可用。")
                if self.backend is None:
                    research_module = importlib.import_module(".inference", package=__package__)
                    self.backend = research_module.DeepSoloBackend(PROJECT_ROOT)
                inference_backend = self.backend
            self.store.create_task(task_id, self.source_path, f"{backend_name} · {mode_name}")
            self.current_task_id = task_id
            self.current_processed_path = processed_path
            self.current_result_path = result_path
            self.canvas.set_image_array(processed)
            self.log(f"任务 {task_id[:8]}：{backend_name} · {mode_name}")
            self.progress_bar.setValue(0)
            self.progress_label.setText("正在准备模型与切片…")
            self.summary_label.setText("正在识别，请稍候…")
            self.thread = QThread()
            self.worker = InferenceWorker(
                inference_backend,
                processed_path,
                result_path,
                mode["tile_size"],
                mode["overlap"],
                mode,
            )
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.log.connect(self.log)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.inference_finished)
            self.worker.failed.connect(self.inference_failed)
            self.worker.finished.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.finished.connect(self.thread_finished)
            self.thread.start()
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "任务启动失败", str(exc))

    def stop_inference(self):
        if self.worker:
            self.worker.stop()
            self.progress_label.setText("正在停止当前任务…")
            self.log("正在停止任务…")

    def update_progress(self, value: int):
        self.progress_bar.setValue(max(0, min(100, int(value))))
        if value < 100:
            self.progress_label.setText(f"正在识别 · {int(value)}%")

    def thread_finished(self):
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def inference_finished(self, detections):
        self.detections = detections
        self.canvas.set_detections(detections)
        self.populate_table(detections)
        self.summary_label.setText(f"共检测到 {len(detections)} 个文字实例")
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"识别完成 · 共 {len(detections)} 个地名")
        if self.current_task_id and self.current_processed_path and self.current_result_path:
            self.store.finish_task(self.current_task_id, self.current_processed_path, self.current_result_path)
            self.store.save_detections(self.current_task_id, detections)
        self._set_export_enabled(True)
        self.log(f"识别完成：{len(detections)} 个文字实例")

    def inference_failed(self, message: str):
        if self.current_task_id and self.current_processed_path and self.current_result_path:
            self.store.finish_task(self.current_task_id, self.current_processed_path, self.current_result_path, "failed")
        self.log("识别失败：" + message)
        self.progress_label.setText("识别失败，请检查配置或切换后端")
        QMessageBox.critical(self, "识别失败", message)

    def populate_table(self, detections):
        self._updating_table = True
        try:
            self.result_table.setRowCount(0)
            for index, item in enumerate(detections, start=1):
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                values = [
                    str(index),
                    str(item.get("text", "")) or "[空文本]",
                    f"{float(item.get('score', 0.0)):.2f}",
                    f"{float(item.get('rec_confidence', 0.0)):.2f}",
                ]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column != 1:
                        cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                    self.result_table.setItem(row, column, cell)
            self.result_table.resizeColumnsToContents()
        finally:
            self._updating_table = False

    def filter_results(self, text: str):
        needle = text.strip().lower()
        for row in range(self.result_table.rowCount()):
            value = self.result_table.item(row, 1).text().lower()
            self.result_table.setRowHidden(row, bool(needle and needle not in value))

    def table_selection_changed(self):
        rows = self.result_table.selectionModel().selectedRows()
        if rows:
            self.select_detection(rows[0].row())

    def table_item_changed(self, item):
        if self._updating_table or item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self.detections):
            new_text = item.text().strip()
            if new_text:
                self.detections[row]["text"] = new_text
                self.detections[row]["text_correction"] = {"method": "manual_edit"}
                self.canvas.set_detections(self.detections)
                self.persist_current_result()
                self.log(f"已人工修正第 {row + 1} 项：{new_text}")

    def select_detection(self, index: int):
        if 0 <= index < len(self.detections):
            self.canvas.highlight(index)
            self.result_table.selectRow(index)
            point = self.detections[index]["polygon"][0]
            self.canvas.centerOn(QPointF(float(point[0]), float(point[1])))

    def _set_export_enabled(self, enabled: bool):
        for button in (
            self.export_json_button,
            self.export_geojson_button,
            self.export_csv_button,
            self.merge_button,
            self.add_button,
        ):
            button.setEnabled(enabled)

    def merge_selected(self):
        rows = sorted({index.row() for index in self.result_table.selectionModel().selectedRows()})
        if len(rows) < 2:
            QMessageBox.information(self, "合并地名", "请在右侧结果表中选择至少两个文字实例。")
            return
        default_text = "".join(str(self.detections[row].get("text", "")) for row in rows)
        text, ok = QInputDialog.getText(self, "合并地名", "合并后的地名：", text=default_text)
        if not ok or not text.strip():
            return
        points = [point for row in rows for point in self.detections[row].get("polygon", [])]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        merged = dict(self.detections[rows[0]])
        merged["text"] = text.strip()
        merged["polygon"] = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        merged["text_correction"] = {"method": "manual_merge", "parts": [self.detections[row].get("text", "") for row in rows]}
        self.detections = [
            merged if index == rows[0] else item
            for index, item in enumerate(self.detections)
            if index == rows[0] or index not in rows[1:]
        ]
        self.populate_table(self.detections)
        self.canvas.set_detections(self.detections)
        self.persist_current_result()
        self.log(f"已合并 {len(rows)} 个文字实例：{text.strip()}")

    def start_add_mode(self):
        if self.canvas.pixmap_item is None:
            return
        self.canvas.enable_add_mode(True)
        self.add_button.setText("请在地图上拖拽框选")
        self.log("请在地图上拖拽框选缺失地名区域")

    def add_detection_from_rect(self, x, y, width, height):
        text, ok = QInputDialog.getText(self, "手动添加地名", "地名：")
        self.add_button.setText("手动添加地名")
        if not ok or not text.strip():
            return
        self.detections.append(
            {
                "polygon": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                "text": text.strip(),
                "score": 1.0,
                "rec_confidence": 1.0,
                "text_correction": {"method": "manual_add"},
            }
        )
        self.populate_table(self.detections)
        self.canvas.set_detections(self.detections)
        self.persist_current_result()
        self.log(f"已手动添加：{text.strip()}")

    def persist_current_result(self):
        if not self.current_result_path:
            return
        payload = [{
            "image_id": self.source_path.stem if self.source_path else "current",
            "file_name": str(self.source_path) if self.source_path else "",
            "instances": self.detections,
        }]
        self.current_result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def show_history(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("任务历史")
        dialog.resize(900, 420)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["时间", "状态", "模式", "输入图像", "任务 ID"])
        table.setSelectionBehavior(QTableWidget.SelectRows)
        tasks = self.store.recent_tasks()
        for task in tasks:
            row = table.rowCount()
            table.insertRow(row)
            values = [task["created_at"], task["status"], task["mode"], Path(task["source_path"]).name, task["id"][:12]]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.UserRole, task["id"])
                table.setItem(row, column, cell)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        load_button = QPushButton("加载选中任务")
        layout.addWidget(load_button)

        def load_selected():
            rows = table.selectionModel().selectedRows()
            if not rows:
                return
            task_id = table.item(rows[0].row(), 0).data(Qt.UserRole)
            task = next((item for item in tasks if item["id"] == task_id), None)
            if task is None or not task["result_path"] or not Path(task["result_path"]).is_file():
                QMessageBox.information(dialog, "任务历史", "该任务没有可加载的结果文件。")
                return
            payload = json.loads(Path(task["result_path"]).read_text(encoding="utf-8"))
            self.source_path = Path(task["source_path"])
            image_path = Path(task["processed_path"] or task["source_path"])
            if image_path.is_file():
                self.current_image = read_image(image_path)
                self.canvas.set_image_array(self.current_image)
            self.current_task_id = task_id
            self.current_result_path = Path(task["result_path"])
            self.detections = payload[0].get("instances", []) if payload else []
            self.canvas.set_detections(self.detections)
            self.populate_table(self.detections)
            self.summary_label.setText(f"已加载历史任务：{len(self.detections)} 个文字实例")
            self._set_export_enabled(True)
            self.log(f"已加载历史任务：{task_id[:8]}")
            dialog.accept()

        load_button.clicked.connect(load_selected)
        dialog.exec_()

    def _choose_export(self, suffix: str):
        if not self.current_task_id:
            return None
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", str(PROJECT_ROOT / f"place_names.{suffix}"), f"*.{suffix}")
        return Path(path) if path else None

    def export_json(self):
        path = self._choose_export("json")
        if path:
            path.write_text(json.dumps(self.detections, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"已导出：{path}")

    def export_geojson(self):
        path = self._choose_export("geojson")
        if path:
            export_geojson(self.detections, path, self.current_task_id or "")
            self.log(f"已导出：{path}")

    def export_csv(self):
        path = self._choose_export("csv")
        if path:
            export_csv(self.detections, path)
            self.log(f"已导出：{path}")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            if self.thread:
                self.thread.quit()
                if not self.thread.wait(5000):
                    QMessageBox.warning(
                        self,
                        "任务仍在运行",
                        "当前推理线程尚未结束，请等待当前切片完成后再关闭平台。",
                    )
                    event.ignore()
                    return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Historical Map Place Name Platform")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    main()
