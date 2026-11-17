import html
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)


class MapCanvas(QGraphicsView):
    detection_clicked = pyqtSignal(int)
    rectangle_created = pyqtSignal(float, float, float, float)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#1e2329"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None
        self.overlay_items = []
        self.detections = []
        self.display_mode = "outside"
        self.add_mode = False
        self._selection_start = None
        self._rubber_band = None
        self._zoom_factor = 1.0

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.18 if delta > 0 else 1 / 1.18
        self._apply_zoom(factor)
        event.accept()

    def _apply_zoom(self, factor: float):
        next_zoom = max(0.20, min(8.0, self._zoom_factor * factor))
        actual_factor = next_zoom / self._zoom_factor
        if abs(actual_factor - 1.0) < 1e-6:
            return
        self.scale(actual_factor, actual_factor)
        self._zoom_factor = next_zoom
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(1 / 1.25)

    def fit_image(self):
        if self.pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_factor)

    def set_display_mode(self, mode: str):
        self.display_mode = mode
        if self.detections:
            self.set_detections(self.detections)

    def set_image_path(self, path: Path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"无法显示图像：{path}")
        self._set_pixmap(pixmap)

    def set_image_array(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self._set_pixmap(QPixmap.fromImage(qimage))

    def _set_pixmap(self, pixmap: QPixmap):
        self.scene.clear()
        self.overlay_items.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_factor)
        self.detections = []

    def clear_detections(self):
        if self._rubber_band is not None:
            self.scene.removeItem(self._rubber_band)
            self._rubber_band = None
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items.clear()
        self.detections = []

    def _label_position(self, bounds: QRectF, label_bounds: QRectF, image_rect: QRectF, occupied):
        """Find a nearby label position without covering another label or its box.

        A single four-position choice is not enough for dense historical maps:
        several boxes can share the same side, and the old fallback could put
        the next label directly on top of an existing one.  We search several
        staggered positions around the box first, then a small local grid, and
        only use an overlap-minimizing fallback when the image is genuinely
        saturated.
        """
        gap = 8.0
        width = label_bounds.width()
        height = label_bounds.height()
        safe_image = image_rect.adjusted(2, 2, -2, -2)
        candidates = []

        def add(x, y):
            candidates.append(QRectF(float(x), float(y), width, height))

        # Prefer the four sides of the detection box.  Staggering the anchor
        # along each side prevents rows of adjacent labels from colliding.
        x_offsets = [
            0.0,
            (bounds.width() - width) / 2.0,
            bounds.width() - width,
        ]
        y_offsets = [
            0.0,
            (bounds.height() - height) / 2.0,
            bounds.height() - height,
        ]
        for offset in x_offsets:
            add(bounds.left() + offset, bounds.top() - height - gap)
            add(bounds.left() + offset, bounds.bottom() + gap)
        for offset in y_offsets:
            add(bounds.right() + gap, bounds.top() + offset)
            add(bounds.left() - width - gap, bounds.top() + offset)

        # Add progressively farther positions to create a local label lane.
        step = max(14.0, height + gap)
        for ring in range(1, 9):
            distance = ring * step
            for offset in x_offsets:
                add(bounds.left() + offset - distance, bounds.top() - height - gap)
                add(bounds.left() + offset + distance, bounds.top() - height - gap)
                add(bounds.left() + offset - distance, bounds.bottom() + gap)
                add(bounds.left() + offset + distance, bounds.bottom() + gap)
            for offset in y_offsets:
                add(bounds.right() + gap, bounds.top() + offset - distance)
                add(bounds.right() + gap, bounds.top() + offset + distance)
                add(bounds.left() - width - gap, bounds.top() + offset - distance)
                add(bounds.left() - width - gap, bounds.top() + offset + distance)

        def is_free(rect):
            if not safe_image.contains(rect) or rect.intersects(bounds):
                return False
            return not any(rect.intersects(old) for old in occupied)

        for rect in candidates:
            if is_free(rect):
                return rect.left(), rect.top(), rect

        # The local candidates may all be outside a narrow crop.  Search a
        # coarse grid around the box before accepting any overlap.
        grid_step = max(16.0, min(48.0, step))
        radius = 10
        center_x = bounds.center().x()
        center_y = bounds.center().y()
        for row in range(-radius, radius + 1):
            for column in range(-radius, radius + 1):
                add(
                    center_x - width / 2.0 + column * grid_step,
                    center_y - height / 2.0 + row * grid_step,
                )
                rect = candidates[-1]
                if is_free(rect):
                    return rect.left(), rect.top(), rect

        # Last resort: choose the in-image candidate with the least overlap,
        # while keeping it as close as possible to its detection box.
        best = None
        best_cost = None
        for rect in candidates:
            if not safe_image.contains(rect):
                continue
            overlap = sum(
                rect.intersected(old).width() * rect.intersected(old).height()
                for old in occupied
            )
            distance = abs(rect.center().x() - bounds.center().x()) + abs(
                rect.center().y() - bounds.center().y()
            )
            cost = overlap * 1000.0 + distance
            if best_cost is None or cost < best_cost:
                best = rect
                best_cost = cost
        if best is not None:
            return best.left(), best.top(), best

        x = min(max(safe_image.left(), bounds.left()), safe_image.right() - width)
        y = min(max(safe_image.top(), bounds.top() - height), safe_image.bottom() - height)
        fallback = QRectF(x, y, width, height)
        return x, y, fallback

    def set_detections(self, detections: list[dict]):
        self.clear_detections()
        self.detections = detections
        if not self.pixmap_item:
            return
        image_rect = self.pixmap_item.boundingRect()
        occupied = []
        for index, item in enumerate(detections):
            points = item.get("polygon", [])
            if len(points) < 3:
                continue
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
            outline = QGraphicsPolygonItem(polygon)
            outline.setPen(QPen(QColor("#19d85b"), 3))
            outline.setBrush(QBrush(Qt.NoBrush))
            outline.setZValue(3)
            outline.setData(0, index)
            outline.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.scene.addItem(outline)
            self.overlay_items.append(outline)

            if self.display_mode == "clean":
                continue
            text = str(item.get("text", "")) or "[空文本]"
            score = float(item.get("score", 0.0))
            label = QGraphicsTextItem()
            safe = html.escape(f"{text}  {score:.2f}")
            label.setHtml(f'<span style="color:#19e85c; background-color:#000000;">{safe}</span>')
            # Keep outside labels readable without dominating dense map areas.
            label.setFont(QFont("Microsoft YaHei", 12))
            label.setZValue(4)
            label.setData(0, index)
            label.setAcceptedMouseButtons(Qt.NoButton)
            self.scene.addItem(label)
            label_bounds = label.boundingRect()
            x, y, occupied_rect = self._label_position(
                polygon.boundingRect(), label_bounds, image_rect, occupied
            )
            label.setPos(x, y)
            # Reserve a small margin around every label so adjacent labels do
            # not touch even when their visible text backgrounds are narrow.
            occupied.append(occupied_rect.adjusted(-3, -3, 3, 3))
            self.overlay_items.append(label)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self.add_mode and event.button() == Qt.LeftButton:
            self._selection_start = scene_pos
            self._rubber_band = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
            self._rubber_band.setPen(QPen(QColor("#ffd166"), 3, Qt.DashLine))
            self._rubber_band.setBrush(QBrush(QColor(255, 209, 102, 45)))
            self._rubber_band.setZValue(10)
            self.scene.addItem(self._rubber_band)
            return
        item = self.scene.itemAt(scene_pos, self.transform())
        if item is not None:
            index = item.data(0)
            if index is not None:
                self.detection_clicked.emit(int(index))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.add_mode and self._selection_start is not None and self._rubber_band is not None:
            current = self.mapToScene(event.pos())
            self._rubber_band.setRect(QRectF(self._selection_start, current).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.add_mode and self._selection_start is not None and event.button() == Qt.LeftButton:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._selection_start, current).normalized()
            self._selection_start = None
            if self._rubber_band is not None:
                self.scene.removeItem(self._rubber_band)
                self._rubber_band = None
            self.enable_add_mode(False)
            if rect.width() >= 5 and rect.height() >= 5:
                self.rectangle_created.emit(rect.left(), rect.top(), rect.width(), rect.height())
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.add_mode and event.button() == Qt.LeftButton:
            self._apply_zoom(1.35)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def enable_add_mode(self, enabled: bool):
        self.add_mode = enabled
        self._selection_start = None
        if not enabled and self._rubber_band is not None:
            self.scene.removeItem(self._rubber_band)
            self._rubber_band = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def highlight(self, index: int):
        for item in self.overlay_items:
            if not isinstance(item, QGraphicsPolygonItem):
                continue
            selected = item.data(0) == index
            item.setPen(QPen(QColor("#ff4d4f" if selected else "#19d85b"), 5 if selected else 3))
