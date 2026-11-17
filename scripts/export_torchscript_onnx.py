"""Convert the verified v9 TorchScript artifact to an ONNX CPU artifact."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "models" / "recognizer.ts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "recognizer.onnx",
    )
    parser.add_argument("--size", type=int, default=1280)
    args = parser.parse_args()

    model = torch.jit.load(str(args.input), map_location="cpu").eval()
    example = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        opset_version=17,
        input_names=["image"],
        output_names=[
            "pred_logits",
            "pred_text_logits",
            "pred_ctrl_points",
            "pred_bd_points",
        ],
        do_constant_folding=True,
        dynamo=False,
    )
    metadata = {
        "format": "onnx",
        "source": str(args.input.name),
        "tile_size": args.size,
        "sha256": sha256(args.output),
        "notes": "Raw v9 outputs only; platform keeps the same decoding and post-processing.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
