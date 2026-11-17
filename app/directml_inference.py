"""Windows DirectML GPU inference for the ONNX deployment graph."""

from pathlib import Path

import cv2
import numpy as np

from .openvino_inference import OpenVINOCpuBackend, _decode, _softmax


class DirectMLBackend(OpenVINOCpuBackend):
    """Run the ONNX graph through the Windows DirectML execution provider."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.provider_name = "Windows GPU（DirectML）"
        self.session = None
        self.input_name = None
        self.output_names = None

    def _load(self):
        if self.session is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GPU 模型不存在：{self.model_path}")
        if not self.dictionary_path.is_file():
            raise FileNotFoundError(f"GPU 字符词表不存在：{self.dictionary_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 DirectML GPU 依赖，请运行 .\\scripts\\setup_windows_directml.ps1"
            ) from exc
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" not in providers:
            raise RuntimeError(
                "当前环境没有可用的 DirectML 执行器，请安装 onnxruntime-directml。"
            )
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        if "DmlExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("DirectML 执行器加载失败，请检查显卡驱动。")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
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
        tensor = np.ascontiguousarray(model_tile.transpose(2, 0, 1)[None]).astype(np.float32)
        try:
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
        except Exception as exc:  # noqa: BLE001 - provider errors need a GUI message
            raise RuntimeError(f"DirectML GPU 推理失败：{exc}") from exc
        cls_logits, text_logits, ctrl_points, bd_points = outputs
        scores = 1.0 / (1.0 + np.exp(-cls_logits.mean(axis=2)))[0, :, 0]
        text_probabilities = _softmax(text_logits[0], axis=-1)
        text_indices = np.argmax(text_probabilities, axis=-1)
        rows = []
        for index in np.flatnonzero(scores >= model_threshold).tolist():
            boundary = bd_points[0, index].copy()
            boundary[:, 0::2] *= original_width
            boundary[:, 1::2] *= original_height
            side_a, side_b = np.hsplit(boundary, 2)
            polygon = np.vstack([side_a, side_b[::-1]])
            polygon[:, 0] += x0
            polygon[:, 1] += y0
            text, rec_confidence = _decode(
                text_indices[index].tolist(), text_probabilities[index], self.labels
            )
            rows.append({
                "polygon": polygon.tolist(),
                "text": text,
                "score": float(scores[index]),
                "rec_confidence": rec_confidence,
                "source_tile": [x0, y0, x0 + original_width, y0 + original_height],
            })
        return rows
