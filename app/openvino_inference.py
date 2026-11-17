"""Native OpenVINO CPU inference for the exported deployment graph."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import APP_ROOT
from .cpu_postprocess import (
    boundary_ranked_items,
    geometry_filter,
    polygon_nms,
    public_items,
    restore_scores,
)
from .preprocessing import read_image


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


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


class OpenVINOCpuBackend:
    """Run the deployment graph with OpenVINO's native Intel CPU executor."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_path = APP_ROOT / "models" / "recognizer.onnx"
        self.dictionary_path = APP_ROOT / "models" / "tw25_cls_list.txt"
        self.compiled_model = None
        self.infer_request = None
        self.input_port = None
        self.labels = None
        self.provider_name = "OpenVINO CPU"
        self.worker_count = max(
            1, int(os.environ.get("HMP_OPENVINO_WORKERS", "1"))
        )
        self._thread_local = threading.local()

    def _load(self):
        if self.compiled_model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO 模型不存在：{self.model_path}")
        if not self.dictionary_path.is_file():
            raise FileNotFoundError(f"CPU 字符词表不存在：{self.dictionary_path}")
        try:
            import openvino as ov
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 OpenVINO，请先运行：.\\scripts\\setup_windows_cpu.ps1"
            ) from exc

        core = ov.Core()
        model = core.read_model(str(self.model_path))
        # OpenVINO's default CPU scheduling was faster on this hybrid Intel
        # processor than forcing the LATENCY hint. Optional tuning remains
        # available through environment variables for other machines.
        properties = {}
        performance_hint = os.environ.get("HMP_OPENVINO_HINT")
        if performance_hint:
            properties["PERFORMANCE_HINT"] = performance_hint
        thread_count = os.environ.get("HMP_OPENVINO_THREADS")
        if thread_count:
            properties["INFERENCE_NUM_THREADS"] = max(1, int(thread_count))
        self.compiled_model = core.compile_model(model, "CPU", properties)
        self.input_port = self.compiled_model.input(0)
        self.labels = self.dictionary_path.read_text(encoding="utf-8").splitlines()

    def _request_for_current_thread(self):
        request = getattr(self._thread_local, "request", None)
        if request is None:
            request = self.compiled_model.create_infer_request()
            self._thread_local.request = request
        return request

    def _predict_tile(self, tile, x0, y0, model_threshold, model_input_size=1280):
        self._load()
        original_height, original_width = tile.shape[:2]
        if original_height != model_input_size or original_width != model_input_size:
            model_tile = cv2.resize(
                tile, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA
            )
        else:
            model_tile = tile
        tensor = np.ascontiguousarray(model_tile.transpose(2, 0, 1)[None]).astype(np.float32)
        output_values = self._request_for_current_thread().infer(
            {self.input_port: tensor}
        )
        outputs = tuple(output_values[output] for output in self.compiled_model.outputs)
        cls_logits, text_logits, ctrl_points, bd_points = outputs
        scores = 1.0 / (1.0 + np.exp(-cls_logits.mean(axis=2)))[0, :, 0]
        text_probabilities = _softmax(text_logits[0], axis=-1)
        text_indices = np.argmax(text_probabilities, axis=-1)
        rows = []
        for index in np.flatnonzero(scores >= model_threshold).tolist():
            boundary = bd_points[0, index].copy()
            boundary[:, 0::2] *= original_width
            boundary[:, 1::2] *= original_height
            side_a, side_b = np.hsplit(boundary, 2)
            polygon = np.vstack([side_a, side_b[::-1]])
            polygon[:, 0] += x0
            polygon[:, 1] += y0
            text, rec_confidence = _decode(
                text_indices[index].tolist(), text_probabilities[index], self.labels
            )
            rows.append({
                "polygon": polygon.tolist(),
                "text": text,
                "score": float(scores[index]),
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
        self._load()
        if log:
            log(
                f"Windows CPU 参数：{self.provider_name}，"
                f"切片 {tile_size}，模型输入 {model_input_size}，"
                f"并发 {self.worker_count}"
            )
        raw = []
        def run_box(box):
            x0, y0, x1, y1 = box
            if should_stop and should_stop():
                raise RuntimeError("用户停止了推理任务")
            return self._predict_tile(
                image[y0:y1, x0:x1], x0, y0, model_threshold, model_input_size
            )

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            for current, rows in enumerate(executor.map(run_box, boxes), start=1):
                raw.extend(rows)
                if log and (current == 1 or current == len(boxes) or current % 5 == 0):
                    log(f"OpenVINO CPU 推理：{current}/{len(boxes)} 个切片")

        if options.get("multi_scale"):
            secondary = int(options.get("secondary_tile_size", 1600))
            secondary_boxes = _tile_boxes(width, height, secondary, overlap)
            def run_secondary_box(box):
                x0, y0, x1, y1 = box
                if should_stop and should_stop():
                    raise RuntimeError("用户停止了推理任务")
                return self._predict_tile(
                    image[y0:y1, x0:x1], x0, y0, model_threshold, model_input_size
                )

            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                for current, rows in enumerate(executor.map(run_secondary_box, secondary_boxes), start=1):
                    raw.extend(rows)
                    if log and (current == len(secondary_boxes) or current % 5 == 0):
                        log(f"OpenVINO CPU 1600 补检：{current}/{len(secondary_boxes)} 个切片")
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
            log(f"OpenVINO CPU 推理完成：{len(detections)} 个文字实例，切片 {len(boxes)} 个")
        return detections
