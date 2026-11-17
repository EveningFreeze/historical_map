"""Benchmark OpenVINO CPU scheduling properties on eight real map tiles."""

import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.cpu_inference import _tile_boxes  # noqa: E402


image = cv2.imread(
    str(ROOT / "data" / "tasks" / "de9091554dde4416a8eb283dc0a155a4" / "preprocessed.png"),
    cv2.IMREAD_COLOR,
)
height, width = image.shape[:2]
tensors = []
for x0, y0, x1, y1 in _tile_boxes(width, height, 1280, 0.15)[:8]:
    tile = cv2.resize(image[y0:y1, x0:x1], (1280, 1280), interpolation=cv2.INTER_AREA)
    tensors.append(np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(np.float32))

core = ov.Core()
model = core.read_model(str(ROOT / "models" / "recognizer.onnx"))

variants = {
    "default": {},
    "latency": {"PERFORMANCE_HINT": "LATENCY"},
    "throughput": {"PERFORMANCE_HINT": "THROUGHPUT"},
    "streams2": {"NUM_STREAMS": 2},
    "threads16": {"INFERENCE_NUM_THREADS": 16},
}
for name, properties in variants.items():
    compiled = core.compile_model(model, "CPU", properties)
    request = compiled.create_infer_request()
    request.infer({compiled.input(0): tensors[0]})
    start = time.perf_counter()
    for tensor in tensors:
        request.infer({compiled.input(0): tensor})
    elapsed = time.perf_counter() - start
    print(f"{name}={elapsed:.3f}s", flush=True)
