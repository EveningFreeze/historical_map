"""Export the current DeepSolo v9 model to a CPU-loadable TorchScript artifact.

This is an experiment for the Windows offline deployment path.  It exports
the neural network raw outputs only; tile fusion, polygon reconstruction, NMS,
and CTC decoding remain in the platform CPU backend.

Run from WSL with the existing DeepSolo environment, for example:

    /home/freeze/tw25-venv/bin/python \
      /mnt/d/地名提取/historical_map_platform/scripts/export_deepsolo_torchscript.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image

# Detectron2 0.6 still references the Pillow 9.x compatibility alias.
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "DeepSolo-main" / "DeepSolo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "tools"))

from adet.config import get_cfg  # noqa: E402
from adet.layers.ms_deform_attn import (  # noqa: E402
    MSDeformAttn,
    ms_deform_attn_core_pytorch,
)
from detectron2.engine.defaults import DefaultPredictor  # noqa: E402
from detectron2.structures import ImageList  # noqa: E402


def pytorch_deformable_forward(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
):
    """Replace the custom C++/CUDA operator with the PyTorch reference path."""
    n, len_q, _ = query.shape
    _, len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    value = value.view(n, len_in, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(query).view(
        n, len_q, self.n_heads, self.n_levels, self.n_points, 2
    )
    attention_weights = self.attention_weights(query).view(
        n, len_q, self.n_heads, self.n_levels * self.n_points
    )
    attention_weights = torch.softmax(attention_weights, -1).view(
        n, len_q, self.n_heads, self.n_levels, self.n_points
    )
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / self.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError("reference_points must have last dimension 2 or 4")
    output = ms_deform_attn_core_pytorch(
        value, input_spatial_shapes, sampling_locations, attention_weights
    )
    return self.output_proj(output)


class RawDeepSolo(nn.Module):
    """Expose only tensor outputs so Detectron2 Instances stay outside export."""

    def __init__(self, detector, mean, std):
        super().__init__()
        self.detector = detector
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        normalized = (image - self.mean) / self.std
        height = image.shape[2]
        width = image.shape[3]
        samples = ImageList(normalized, [(height, width)] * image.shape[0])
        output = self.detector(samples)
        return (
            output["pred_logits"],
            output["pred_text_logits"],
            output["pred_ctrl_points"],
            output["pred_bd_points"],
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "historical_map_platform" / "models" / "deepsolo_v9_raw.ts",
    )
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="保留的检测查询数；100 为原始 v9，较小值仅用于实验性轻量模型。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = ROOT / "tw25_deepsolo" / "tw25_r50_v9_recognition_head.yaml"
    weights_path = ROOT / "tw25_deepsolo_output_v9_recognition_head" / "model_0000499.pth"
    dictionary_path = ROOT / "tw25_deepsolo" / "data" / "tw25_cls_list"

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MIN_SIZE_TEST = args.tile_size
    cfg.INPUT.MAX_SIZE_TEST = args.tile_size
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    predictor.model.eval()

    original_num_queries = predictor.model.detection_transformer.num_proposals
    if args.num_queries <= 0 or args.num_queries > original_num_queries:
        raise ValueError(
            f"--num-queries 必须在 1 到 {original_num_queries} 之间，当前为 {args.num_queries}"
        )
    if args.num_queries != original_num_queries:
        detector = predictor.model.detection_transformer
        # The checkpoint is still loaded with all 100 queries.  Keep the first
        # N learned point embeddings, while changing encoder top-k and decoder
        # query count so the compute graph is genuinely smaller.
        point_weights = detector.point_embed.weight.detach().clone()
        detector.num_proposals = args.num_queries
        detector.transformer.num_proposals = args.num_queries
        detector.point_embed = nn.Embedding(
            args.num_queries * detector.num_points, detector.d_model
        ).to(point_weights.device)
        detector.point_embed.weight.data.copy_(
            point_weights[: args.num_queries * detector.num_points]
        )

    # The reference implementation uses grid_sample and standard tensor ops.
    # It avoids exporting the adet._C custom operator.
    MSDeformAttn.forward = pytorch_deformable_forward
    wrapper = RawDeepSolo(
        predictor.model.detection_transformer,
        cfg.MODEL.PIXEL_MEAN,
        cfg.MODEL.PIXEL_STD,
    ).eval()
    device = torch.device(args.device)
    wrapper.to(device)
    example = torch.zeros((1, 3, args.tile_size, args.tile_size), device=device)

    with torch.no_grad():
        eager_outputs = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        traced_outputs = traced(example)
        max_abs_diff = max(
            float((left - right).abs().max().item())
            for left, right in zip(eager_outputs, traced_outputs)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    metadata = {
        "format": "torchscript",
        "raw_outputs_only": True,
        "tile_size": args.tile_size,
        "vocabulary": str(dictionary_path),
        "weights": str(weights_path),
        "device_used_for_export": args.device,
        "num_queries": args.num_queries,
        "original_num_queries": original_num_queries,
        "max_abs_diff_eager_vs_traced": max_abs_diff,
        "notes": "CPU backend must implement polygon reconstruction, NMS, and CTC decoding.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
