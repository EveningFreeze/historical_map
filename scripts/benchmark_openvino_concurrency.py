"""Measure parallel OpenVINO requests without changing the v9 graph."""

import concurrent.futures
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.cpu_inference import _tile_boxes  # noqa: E402


image = cv2.imread(
    str(ROOT / "data" / "tasks" / "de9091554dde4416a8eb283dc0a155a4" / "preprocessed.png"),
    cv2.IMREAD_COLOR,
)
height, width = image.shape[:2]
boxes = _tile_boxes(width, height, 1280, 0.15)[:8]
tensors = []
for x0, y0, x1, y1 in boxes:
    tile = image[y0:y1, x0:x1]
    tile = cv2.resize(tile, (1280, 1280), interpolation=cv2.INTER_AREA)
    tensors.append(np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(np.float32))

core = ov.Core()
model = core.read_model(str(ROOT / "models" / "recognizer.onnx"))
compiled = core.compile_model(model, "CPU")
input_port = compiled.input(0)


def infer_one(tensor):
    request = compiled.create_infer_request()
    values = request.infer({input_port: tensor})
    return tuple(values[output] for output in compiled.outputs)


infer_one(tensors[0])

for workers in (1, 2, 4):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(infer_one, tensors))
    elapsed = time.perf_counter() - start
    print(
        f"workers={workers} total={elapsed:.3f}s per_tile={elapsed / len(tensors):.3f}s",
        flush=True,
    )

print("same_first_shape", outputs[0][1].shape)
