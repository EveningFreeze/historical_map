import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Callable


def to_wsl_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    drive = path.drive[:1].lower()
    if not drive:
        raise ValueError(f"无法转换为 WSL 路径：{path}")
    return f"/mnt/{drive}{path.as_posix()[2:]}"


class DeepSoloBackend:
    def __init__(self, project_root: Path, distro: str = "Ubuntu"):
        self.project_root = project_root
        self.distro = os.environ.get("HMP_WSL_DISTRO", distro)
        self.python = os.environ.get("HMP_WSL_PYTHON", "/home/freeze/tw25-venv/bin/python")
        self.script = project_root / "tw25_deepsolo" / "platform_infer_tw25.py"
        self.config = project_root / "tw25_deepsolo" / "platform_inference_config.json"

    def command(
        self,
        image_path: Path,
        output_path: Path,
        tile_size: int,
        overlap: float,
        options: dict | None = None,
    ):
        script = to_wsl_path(self.script)
        config = to_wsl_path(self.config)
        image = to_wsl_path(image_path)
        output = to_wsl_path(output_path)
        options = options or {}
        args = [
            "wsl.exe", "-d", self.distro, "--",
            self.python, script,
            "--input", image,
            "--output", output,
            "--config", config,
            "--mode", "optimized",
            "--tile-size", str(tile_size),
            "--overlap", str(overlap),
            "--no-name-postprocess",
        ]
        optional_flags = (
            ("model_score_threshold", "--model-score-threshold"),
            ("score_threshold", "--score-threshold"),
            ("nms_threshold", "--nms-threshold"),
            ("max_dimension_per_char", "--max-dimension-per-char"),
            ("recognition_threshold", "--recognition-threshold"),
            ("secondary_tile_size", "--secondary-tile-size"),
            ("secondary_trigger_max_instances", "--secondary-trigger-max-instances"),
        )
        for key, flag in optional_flags:
            if key in options:
                args.extend([flag, str(options[key])])
        if options.get("multi_scale", False):
            args.append("--multi-scale")
        else:
            args.append("--no-multi-scale")
        return args

    def run(
        self,
        image_path: Path,
        output_path: Path,
        tile_size: int,
        overlap: float,
        options: dict | None = None,
        log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command(image_path, output_path, tile_size, overlap, options)
        if log:
            log("启动 WSL DeepSolo 推理：" + " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout or []:
                if log and line.strip():
                    log(line.rstrip())
                if should_stop and should_stop():
                    process.terminate()
                    raise RuntimeError("用户停止了推理任务")
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
        if return_code != 0:
            raise RuntimeError(f"DeepSolo 推理失败，退出码：{return_code}")
        if not output_path.is_file():
            raise FileNotFoundError(f"推理没有生成结果文件：{output_path}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return payload[0].get("instances", []) if payload else []


def export_geojson(detections: list[dict], output_path: Path, task_id: str) -> None:
    features = []
    for index, item in enumerate(detections, start=1):
        polygon = item.get("polygon", [])
        if len(polygon) < 3:
            continue
        ring = polygon + [polygon[0]]
        features.append(
            {
                "type": "Feature",
                "id": index,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "text": item.get("text", ""),
                    "score": item.get("score", 0.0),
                    "rec_confidence": item.get("rec_confidence", 0.0),
                    "task_id": task_id,
                    "coordinate_reference": "image_pixel",
                },
            }
        )
    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_csv(detections: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["序号", "地名", "检测置信度", "识别置信度", "多边形坐标"])
        for index, item in enumerate(detections, start=1):
            writer.writerow(
                [
                    index,
                    item.get("text", ""),
                    item.get("score", 0.0),
                    item.get("rec_confidence", 0.0),
                    json.dumps(item.get("polygon", []), ensure_ascii=False),
                ]
            )
