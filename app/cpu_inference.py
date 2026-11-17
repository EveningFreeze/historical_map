"""Offline Windows CPU inference using the exported TorchScript model."""

import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from .config import APP_ROOT
from .cpu_postprocess import (
    boundary_ranked_items,
    geometry_filter,
    polygon_nms,
    public_items,
    restore_scores,
)
from .preprocessing import read_image


def _starts(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    values = list(range(0, size - tile + 1, stride))
    last = size - tile
    if values[-1] != last:
        values.append(last)
    return values


def _tile_boxes(width, height, tile, overlap):
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in _starts(height, tile, overlap)
        for x in _starts(width, tile, overlap)
    ]


def _decode(indices, probabilities, labels):
    chars = []
    emitted = []
    last = -1
    for position, index in enumerate(indices):
        index = int(index)
        if index < len(labels) and index != last:
            chars.append(labels[index])
            emitted.append(float(probabilities[position, index]))
        last = index
    confidence = float(np.mean(emitted)) if emitted else 0.0
    return "".join(chars), confidence


class TorchScriptCpuBackend:
    """Backend compatible with the existing InferenceWorker interface."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_path = APP_ROOT / "models" / "recognizer.ts"
        self.dictionary_path = APP_ROOT / "models" / "tw25_cls_list.txt"
        self.model = None
        self.labels = None
        self.backend_label = "Windows CPU"

    def _load(self):
        if self.model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"CPU 模型不存在：{self.model_path}")
            if not self.dictionary_path.is_file():
                raise FileNotFoundError(f"CPU 字符词表不存在：{self.dictionary_path}")
            default_threads = min(8, max(1, os.cpu_count() or 4))
            threads = int(os.environ.get("HMP_CPU_THREADS", str(default_threads)))
            torch.set_num_threads(max(1, threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Another backend may already have started Torch work.
                pass
            self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
            self.labels = self.dictionary_path.read_text(encoding="utf-8").splitlines()

    def _predict_tile(self, tile, x0, y0, model_threshold, model_input_size=1280):
        self._load()
        original_height, original_width = tile.shape[:2]
        if original_height != model_input_size or original_width != model_input_size:
            model_tile = cv2.resize(
                tile, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA
            )
        else:
            model_tile = tile
        # The exported model keeps DeepSolo's configured BGR normalization.
        tensor = torch.from_numpy(np.ascontiguousarray(model_tile))
        tensor = tensor.permute(2, 0, 1).float().unsqueeze(0)
        with torch.inference_mode():
            cls_logits, text_logits, ctrl_points, bd_points = self.model(tensor)
            scores = torch.sigmoid(cls_logits.mean(dim=2))[0, :, 0]
            text_probabilities = torch.softmax(text_logits[0], dim=-1)
            text_indices = text_probabilities.argmax(dim=-1)
        rows = []
        for index in torch.where(scores >= model_threshold)[0].tolist():
            boundary = bd_points[0, index].cpu().numpy().copy()
            boundary[:, 0::2] *= original_width
            boundary[:, 1::2] *= original_height
            side_a, side_b = np.hsplit(boundary, 2)
            polygon = np.vstack([side_a, side_b[::-1]])
            polygon[:, 0] += x0
            polygon[:, 1] += y0
            text, rec_confidence = _decode(
                text_indices[index].cpu().tolist(),
                text_probabilities[index].cpu().numpy(),
                self.labels,
            )
            rows.append({
                "polygon": polygon.tolist(),
                "text": text,
                "score": float(scores[index].item()),
                "rec_confidence": rec_confidence,
                "source_tile": [x0, y0, x0 + original_width, y0 + original_height],
            })
        return rows

    def run(
        self,
        image_path: Path,
        output_path: Path,
        tile_size: int,
        overlap: float,
        options: dict | None = None,
        log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict]:
        options = options or {}
        image = read_image(image_path)
        height, width = image.shape[:2]
        model_threshold = float(options.get("model_score_threshold", 0.15))
        score_threshold = float(options.get("score_threshold", 0.35))
        recognition_threshold = float(options.get("recognition_threshold", 0.30))
        nms_threshold = float(options.get("nms_threshold", 0.30))
        max_dimension = float(options.get("max_dimension_per_char", 150.0))
        model_input_size = int(options.get("model_input_size", 1280))
        boxes = _tile_boxes(width, height, tile_size, overlap)
        if log:
            log(
                f"{self.backend_label} 参数：线程 {torch.get_num_threads()}，"
                f"切片 {tile_size}，模型输入 {model_input_size}"
            )
        raw = []
        for current, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            if should_stop and should_stop():
                raise RuntimeError("用户停止了推理任务")
            raw.extend(self._predict_tile(
                image[y0:y1, x0:x1], x0, y0, model_threshold, model_input_size
            ))
            if log and (current == 1 or current == len(boxes) or current % 5 == 0):
                log(f"{self.backend_label} 推理：{current}/{len(boxes)} 个切片")

        if options.get("multi_scale"):
            secondary = int(options.get("secondary_tile_size", 1600))
            secondary_boxes = _tile_boxes(width, height, secondary, overlap)
            for current, (x0, y0, x1, y1) in enumerate(secondary_boxes, start=1):
                if should_stop and should_stop():
                    raise RuntimeError("用户停止了推理任务")
                raw.extend(self._predict_tile(
                    image[y0:y1, x0:x1], x0, y0, model_threshold, model_input_size
                ))
                if log and (current == len(secondary_boxes) or current % 5 == 0):
                    log(f"{self.backend_label} 1600 补检：{current}/{len(secondary_boxes)} 个切片")
            boxes = boxes + secondary_boxes

        score_candidates = [item for item in raw if item["score"] >= score_threshold]
        candidates = [
            item for item in score_candidates
            if item["rec_confidence"] >= recognition_threshold
        ]
        ranked = boundary_ranked_items(candidates)
        kept, _ = polygon_nms(ranked, nms_threshold)
        kept = restore_scores(kept)
        kept, _ = geometry_filter(kept, max_dimension)
        detections = public_items(kept)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([{
            "image_id": image_path.stem,
            "file_name": str(image_path),
            "instances": detections,
        }], ensure_ascii=False), encoding="utf-8")
        if log:
            log(f"{self.backend_label} 推理完成：{len(detections)} 个文字实例，切片 {len(boxes)} 个")
        return detections
