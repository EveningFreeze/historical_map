# 平台架构

```text
历史地图
   │
   ├─ 图像预处理：灰度 / CLAHE / 去噪 / 锐化 / OTSU
   │
   ├─ 滑动窗口：快速 / 标准 / 精细 / 平台默认
   │
   ├─ WSL DeepSolo 后端
   │      └─ 1280 切片 + 边界感知 NMS + 识别置信度 + 几何过滤
   │
   ├─ 坐标恢复与结果融合
   │
   ├─ 地图交互：缩放 / 拖动 / 框选结果 / 框外文字
   │
   ├─ 人机协同：直接改名 / 合并碎片 / 手动添加
   │
   └─ 输出：JSON / CSV / 像素坐标 GeoJSON + SQLite 任务历史
```

## 模块边界

- `app/preprocessing.py`：只处理图像，不依赖 DeepSolo。
- `app/inference.py`：只负责调用推理后端和导出格式；后续可替换 PaddleOCR 或 FastAPI。
- `app/widgets.py`：只负责地图画布和检测框交互。
- `app/database.py`：只负责任务和检测结果的本地持久化。
- `tw25_deepsolo/platform_infer_tw25.py`：保留为独立、可命令行调用的模型后端。

## 坐标说明

当前 GeoJSON 使用图像像素坐标，属性中写入 `coordinate_reference=image_pixel`。
要变成真实 GIS 图层，需要增加地图配准控制点，并在导出前应用像素到地理坐标的
仿射或薄板样条变换。
