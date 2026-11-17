"""Create a CUDA-loadable copy of the CPU-exported TorchScript graph.

The original export was traced on CPU and contains explicit CPU Device
constants in DeepSolo's positional encodings.  Replacing those graph
constants preserves the weights and outputs while allowing the same graph to
run after its parameters are moved to CUDA.
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("models/recognizer.ts"))
    parser.add_argument("--output", type=Path, default=Path("models/recognizer_cuda.ts"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    model = torch.jit.load(str(args.source), map_location="cpu").eval()
    changed = 0
    for _, module in model.named_modules():
        try:
            method_names = module._c._method_names()
        except (AttributeError, RuntimeError):
            continue
        for method_name in method_names:
            graph = module._c._get_method(method_name).graph
            for node in graph.nodes():
                if node.kind() != "prim::Constant":
                    continue
                if "value" not in node.attributeNames():
                    continue
                if str(node.output().type()) != "Device":
                    continue
                if node.s("value") == "cpu":
                    node.s_("value", "cuda")
                    changed += 1

    if changed == 0:
        raise RuntimeError("没有找到 CPU Device 常量，未生成 CUDA 模型")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(model, str(args.output))
    metadata = {
        "format": "torchscript",
        "source": str(args.source),
        "device_constants_rewritten": changed,
        "device_used_for_export": "cuda",
        "weights_unchanged": True,
        "notes": "CPU source graph rewritten only for explicit Device constants.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
