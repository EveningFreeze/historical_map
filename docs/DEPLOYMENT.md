# 部署清单

## Windows 桌面端

1. 安装 Python 3.10 或更高版本。
2. 创建桌面端虚拟环境并安装 `requirements.txt`。
3. 确认 `python -c "import PyQt5, cv2"` 成功。
4. 确认 `wsl -l -v` 中存在 Ubuntu。
5. 确认 WSL 中的 DeepSolo 虚拟环境可以运行当前项目的官方验证命令。
6. 根据实际路径设置 `HMP_WSL_PYTHON`，执行 `python run.py`。

## GitHub 提交建议

- 不提交 `historical_map_platform/data/tasks/` 下的任务图片和模型推理输出。
- 不提交 `.venv/`、WSL 虚拟环境或 `.pth` 权重。
- 在仓库 Release 或单独模型下载说明中提供权重来源和 SHA256。
- 保留 `tw25_deepsolo/platform_inference_config.json`，让模型、词表、配置路径可审查。

## 后续服务化

如果需要多人使用，将 `DeepSoloBackend.run()` 提取为 FastAPI 后台任务接口；
前端可以迁移到 Vue3 + OpenLayers，SQLite 迁移到 PostgreSQL/PostGIS，桌面端
仍可作为单机离线版本保留。
