"""Native Windows NVIDIA CUDA inference for the exported deployment graph."""

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import APP_ROOT
from .cpu_inference import TorchScriptCpuBackend, _decode


class TorchScriptCudaBackend(TorchScriptCpuBackend):
    """Run the deployment TorchScript graph on a local NVIDIA CUDA GPU.

    The post-processing and character dictionary are intentionally shared with
    the CPU backend. Only the model forward pass and tensor preparation move to
    CUDA, which keeps the Windows GPU result format compatible with the CPU and
    OpenVINO backends.
    """

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.backend_label = "Windows GPU（CUDA）"
        self.device = torch.device("cuda")
        cuda_model_path = APP_ROOT / "models" / "recognizer_cuda.ts"
        if cuda_model_path.is_file():
            self.model_path = cuda_model_path

    def _load(self):
        if self.model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA CUDA GPU。请安装 NVIDIA 驱动和 GPU 版 PyTorch，"
                "或切换到 Windows CPU（OpenVINO 原生）。"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GPU 模型不存在：{self.model_path}")
        if not self.dictionary_path.is_file():
            raise FileNotFoundError(f"GPU 字符词表不存在：{self.dictionary_path}")
        self.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        self.model = torch.jit.load(
            str(self.model_path), map_location=self.device
        ).eval()
        self.labels = self.dictionary_path.read_text(encoding="utf-8").splitlines()

    def _predict_tile(self, tile, x0, y0, model_threshold, model_input_size=1280):
        self._load()
        original_height, original_width = tile.shape[:2]
        if original_height != model_input_size or original_width != model_input_size:
            model_tile = cv2.resize(
                tile, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA
            )
        else:
            model_tile = tile
        tensor = torch.from_numpy(np.ascontiguousarray(model_tile))
        tensor = tensor.permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                cls_logits, text_logits, ctrl_points, bd_points = self.model(tensor)
                scores = torch.sigmoid(cls_logits.mean(dim=2))[0, :, 0]
                text_probabilities = torch.softmax(text_logits[0], dim=-1)
                text_indices = text_probabilities.argmax(dim=-1)
                scores_np = scores.detach().cpu().numpy()
                text_probabilities_np = text_probabilities.detach().cpu().numpy()
                text_indices_np = text_indices.detach().cpu().numpy()
                bd_points_np = bd_points.detach().cpu().numpy()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA 显存不足。请切换到 2048 下采样模式、降低并发，"
                    "或改用 Windows CPU（OpenVINO 原生）。"
                ) from exc
            raise

        rows = []
        for index in np.flatnonzero(scores_np >= model_threshold).tolist():
            boundary = bd_points_np[0, index].copy()
            boundary[:, 0::2] *= original_width
            boundary[:, 1::2] *= original_height
            side_a, side_b = np.hsplit(boundary, 2)
            polygon = np.vstack([side_a, side_b[::-1]])
            polygon[:, 0] += x0
            polygon[:, 1] += y0
            text, rec_confidence = _decode(
                text_indices_np[index].tolist(),
                text_probabilities_np[index],
                self.labels,
            )
            rows.append({
                "polygon": polygon.tolist(),
                "text": text,
                "score": float(scores_np[index]),
                "rec_confidence": rec_confidence,
                "source_tile": [x0, y0, x0 + original_width, y0 + original_height],
            })
        return rows
