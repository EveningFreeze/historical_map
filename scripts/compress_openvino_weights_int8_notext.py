"""Weight-only INT8 candidate with the six final recognition heads in FP32."""

import json
from pathlib import Path

import nncf
import openvino as ov


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "recognizer.onnx"
OUTPUT = ROOT / "models" / "deepsolo_v9_cpu_weight_int8_notext.xml"
TEXT_HEADS = [
    "/detector/0_1/MatMul",
    "/detector/MatMul_4",
    "/detector/MatMul_12",
    "/detector/MatMul_20",
    "/detector/MatMul_28",
    "/detector/MatMul_36",
]

core = ov.Core()
model = core.read_model(str(INPUT))
compressed = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
    compression_format=nncf.CompressionFormat.DQ,
    ignored_scope=nncf.IgnoredScope(names=TEXT_HEADS),
)
ov.serialize(compressed, str(OUTPUT), str(OUTPUT.with_suffix(".bin")))
metadata = {
    "format": "openvino_weight_only_int8_mixed",
    "source": INPUT.name,
    "ignored_text_heads": TEXT_HEADS,
}
OUTPUT.with_suffix(".json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(metadata, ensure_ascii=False))
