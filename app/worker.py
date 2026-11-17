import re
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .inference import DeepSoloBackend


class InferenceWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        backend: DeepSoloBackend,
        image_path: Path,
        output_path: Path,
        tile_size: int,
        overlap: float,
        options: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.image_path = image_path
        self.output_path = output_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.options = options or {}
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    @pyqtSlot()
    def run(self):
        try:
            self.progress.emit(1)

            def handle_log(message: str):
                self.log.emit(message)
                match = re.search(r"(\d+)\s*/\s*(\d+)\s*个切片", message)
                if match:
                    current = int(match.group(1))
                    total = max(1, int(match.group(2)))
                    self.progress.emit(min(99, max(1, int(current * 99 / total))))

            detections = self.backend.run(
                self.image_path,
                self.output_path,
                self.tile_size,
                self.overlap,
                self.options,
                log=handle_log,
                should_stop=lambda: self._stop_requested,
            )
            self.progress.emit(100)
            self.finished.emit(detections)
        except Exception as exc:  # noqa: BLE001 - forwarded to the GUI
            self.failed.emit(str(exc))
