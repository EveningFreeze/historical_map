"""Create a fast weight-only INT8 OpenVINO candidate without touching FP32."""

import json
from pathlib import Path

import nncf
import openvino as ov


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "recognizer.onnx"
OUTPUT = ROOT / "models" / "deepsolo_v9_cpu_weight_int8.xml"

core = ov.Core()
model = core.read_model(str(INPUT))
compressed = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
    compression_format=nncf.CompressionFormat.DQ,
)
ov.serialize(compressed, str(OUTPUT), str(OUTPUT.with_suffix(".bin")))
metadata = {
    "format": "openvino_weight_only_int8",
    "source": INPUT.name,
    "xml": OUTPUT.name,
    "bin": OUTPUT.with_suffix(".bin").name,
    "note": "Weight-only INT8 candidate; activation quantization is not applied.",
}
OUTPUT.with_suffix(".json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(metadata, ensure_ascii=False))
