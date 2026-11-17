"""Benchmark native OpenVINO CPU execution against the v9 TorchScript output."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(int(os.environ.get("HMP_CPU_THREADS", "8")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
image_path = sorted((ROOT / "data" / "tasks").glob("*/preprocessed.png"))[0]
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
tile = cv2.resize(image, (1280, 1280), interpolation=cv2.INTER_AREA)
x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(np.float32)

torch_model = torch.jit.load(str(ROOT / "models" / "recognizer.ts"), map_location="cpu").eval()
core = ov.Core()
ov_model = core.read_model(str(ROOT / "models" / "recognizer.onnx"))
compiled = core.compile_model(ov_model, "CPU")
request = compiled.create_infer_request()


def torch_run():
    with torch.inference_mode():
        return tuple(v.detach().cpu().numpy() for v in torch_model(torch.from_numpy(x)))


def ov_run():
    values = request.infer({compiled.input(0): x})
    return tuple(values[output] for output in compiled.outputs)


torch_run()
ov_run()
torch_times = []
ov_times = []
for _ in range(3):
    start = time.perf_counter()
    torch_values = torch_run()
    torch_times.append(time.perf_counter() - start)
    start = time.perf_counter()
    ov_values = ov_run()
    ov_times.append(time.perf_counter() - start)
print(f"torchscript={np.median(torch_times):.3f}s")
print(f"openvino_native={np.median(ov_times):.3f}s")
print(
    "max_abs=" + str([
        float(np.max(np.abs(left - right)))
        for left, right in zip(torch_values, ov_values)
    ])
)
print(
    "text_argmax_mismatch=" + str(
        int(np.count_nonzero(np.argmax(torch_values[1], axis=-1) != np.argmax(ov_values[1], axis=-1)))
    )
)
