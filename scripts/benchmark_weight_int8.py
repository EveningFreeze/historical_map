"""Compare FP32 OpenVINO and weight-only INT8 outputs and latency."""

import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


ROOT = Path(__file__).resolve().parents[1]
image = cv2.imread(
    str(ROOT / "data" / "tasks" / "0270f110582a41af82beb657d71c3a3f" / "preprocessed.png"),
    cv2.IMREAD_COLOR,
)
tile = cv2.resize(image, (1280, 1280), interpolation=cv2.INTER_AREA)
x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None]).astype(np.float32)
core = ov.Core()


def make(path):
    compiled = core.compile_model(core.read_model(str(path)), "CPU")
    request = compiled.create_infer_request()
    request.infer({compiled.input(0): x})
    return compiled, request


fp32, fp32_request = make(ROOT / "models" / "recognizer.onnx")
int8, int8_request = make(ROOT / "models" / "deepsolo_v9_cpu_weight_int8.xml")


def run(compiled, request):
    values = request.infer({compiled.input(0): x})
    return tuple(values[output] for output in compiled.outputs)


fp32_times = []
int8_times = []
for _ in range(3):
    start = time.perf_counter()
    fp32_values = run(fp32, fp32_request)
    fp32_times.append(time.perf_counter() - start)
    start = time.perf_counter()
    int8_values = run(int8, int8_request)
    int8_times.append(time.perf_counter() - start)

print(f"fp32={np.median(fp32_times):.3f}s")
print(f"weight_int8={np.median(int8_times):.3f}s")
print("max_abs=" + str([
    float(np.max(np.abs(left - right)))
    for left, right in zip(fp32_values, int8_values)
]))
print("text_argmax_mismatch=" + str(
    int(np.count_nonzero(
        np.argmax(fp32_values[1], axis=-1) != np.argmax(int8_values[1], axis=-1)
    ))
))
