from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    grayscale: bool = False
    clahe: bool = False
    denoise: bool = False
    sharpen: bool = False
    threshold: bool = False


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法保存图像：{path}")
    encoded.tofile(str(path))


def apply_preprocessing(image: np.ndarray, options: PreprocessOptions) -> np.ndarray:
    result = image.copy()
    if options.grayscale:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if options.clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
        result = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if options.denoise:
        result = cv2.GaussianBlur(result, (3, 3), 0)

    if options.sharpen:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.2)
        result = cv2.addWeighted(result, 1.45, blurred, -0.45, 0)

    if options.threshold:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return result
