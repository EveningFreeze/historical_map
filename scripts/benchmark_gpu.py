"""Benchmark the native Windows CUDA backend on one representative map."""

from pathlib import Path
import json
import argparse
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "tasks" / "gpu_benchmark.json"
sys.path.insert(0, str(PROJECT_ROOT))

from app.cuda_inference import TorchScriptCudaBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the native Windows CUDA backend")
    parser.add_argument("image", nargs="?", type=Path, help="要测试的地图图片路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="基准结果 JSON 输出路径",
    )
    args = parser.parse_args()
    image_path = args.image
    if image_path is None:
        candidates = sorted(
            path for path in (PROJECT_ROOT / "data" / "tasks").glob("**/*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        )
        if not candidates:
            raise FileNotFoundError(
                "未找到示例图片，请执行：python scripts/benchmark_gpu.py <图片路径>"
            )
        image_path = candidates[0]
    image_path = image_path.resolve()
    output_path = args.output.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    backend = TorchScriptCudaBackend(PROJECT_ROOT)
    started = time.perf_counter()
    detections = backend.run(
        image_path,
        output_path,
        tile_size=1280,
        overlap=0.15,
        options={
            "model_input_size": 1280,
            "model_score_threshold": 0.15,
            "score_threshold": 0.35,
            "recognition_threshold": 0.30,
            "nms_threshold": 0.30,
            "max_dimension_per_char": 150.0,
        },
        log=print,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "backend": backend.backend_label,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "image": str(image_path),
        "detections": len(detections),
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
