"""Post-training INT8 quantization for the v9 OpenVINO graph.

The calibration images are real TW25 training tiles. This script never
overwrites the FP32 ONNX/ TorchScript artifacts.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import nncf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parent / "train" / "tw25" / "train"


def preprocess(path: Path, size: int) -> dict[str, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取校准图像：{path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32)
    return {"image": tensor}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "models" / "recognizer.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "deepsolo_v9_cpu_int8.xml")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--size", type=int, default=1280)
    parser.add_argument("--calibration-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()

    paths = sorted(args.data.glob("*.jpg")) + sorted(args.data.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"找不到校准图像：{args.data}")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.calibration_size]

    core = ov.Core()
    model = core.read_model(str(args.input))
    input_name = model.inputs[0].get_any_name()
    dataset = nncf.Dataset(
        paths,
        lambda path: {input_name: preprocess(Path(path), args.size)["image"]},
    )
    quantized = nncf.quantize(
        model,
        dataset,
        subset_size=len(paths),
        target_device=nncf.TargetDevice.CPU,
        preset=nncf.QuantizationPreset.PERFORMANCE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.output.with_suffix(".bin")
    ov.serialize(quantized, str(args.output), str(bin_path))
    metadata = {
        "format": "openvino_int8",
        "source": str(args.input.name),
        "calibration_dir": str(args.data),
        "calibration_size": len(paths),
        "input_size": args.size,
        "seed": args.seed,
        "xml": str(args.output.name),
        "bin": str(bin_path.name),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
