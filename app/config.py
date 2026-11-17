from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
DATA_ROOT = APP_ROOT / "data"
TASKS_ROOT = DATA_ROOT / "tasks"
DB_PATH = DATA_ROOT / "history.sqlite3"

DEEPSOLO_SCRIPT = PROJECT_ROOT / "tw25_deepsolo" / "platform_infer_tw25.py"
DEEPSOLO_CONFIG = PROJECT_ROOT / "tw25_deepsolo" / "platform_inference_config.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

INFERENCE_MODES = {
    "平台默认（1280 切片）": {"tile_size": 1280, "overlap": 0.15},
    "CPU快速预览（2048像素窗口）": {
        "tile_size": 2048,
        "overlap": 0.375,
        "model_input_size": 1280,
    },
    "CPU极速预览（2048像素窗口，低重叠）": {
        "tile_size": 2048,
        "overlap": 0.25,
        "model_input_size": 1280,
    },
    "高召回（1280 低阈值）": {
        "tile_size": 1280,
        "overlap": 0.25,
        "model_score_threshold": 0.08,
        "score_threshold": 0.20,
        "nms_threshold": 0.25,
        "max_dimension_per_char": 200.0,
        "recognition_threshold": 0.15,
    },
    "高召回（1280+1600 双尺度）": {
        "tile_size": 1280,
        "overlap": 0.25,
        "model_score_threshold": 0.08,
        "score_threshold": 0.20,
        "nms_threshold": 0.30,
        "max_dimension_per_char": 200.0,
        "recognition_threshold": 0.15,
        "multi_scale": True,
        "secondary_tile_size": 1600,
        "secondary_trigger_max_instances": 9999,
    },
    "快速（2048像素窗口）": {"tile_size": 2048, "overlap": 0.375},
    "标准（1536像素窗口）": {"tile_size": 1536, "overlap": 0.333},
    "精细（1024像素窗口）": {"tile_size": 1024, "overlap": 0.25},
}


def ensure_app_dirs() -> None:
    TASKS_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
