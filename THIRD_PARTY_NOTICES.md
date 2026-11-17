# Third-party notices

本文件用于说明本项目中源码、模型、数据集和第三方依赖的授权边界。它不是对第三方许可证的替代，发布前应保留各依赖的原始许可证和版权声明。

## 1. 本项目原创内容

除另有说明外，本仓库中由项目维护者原创的 Python、PowerShell、配置、文档和界面代码使用根目录 `LICENSE` 中的 MIT License。

MIT 许可不自动覆盖模型权重、训练数据、历史地图图像或第三方代码。

## 2. DeepSolo、AdelaiDet 和 Detectron2

本项目的研究模型、导出脚本和部分模型处理流程参考或依赖 DeepSolo、AdelaiDet 和 Detectron2。重新分发相关代码或模型前，应同时遵守对应上游项目的许可证、非商业限制和版权声明。

特别注意：本地研究环境中的 AdelaiDet 许可证包含非商业使用说明。当前部署模型和研究后端是否可以用于商业发布，需要根据实际模型来源、训练权重来源和上游许可证进一步确认。未经确认，不要把模型或研究后端宣传为可自由商业使用。

参考项目：

- [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## 3. 模型权重

`models/recognizer_cuda.ts`、`models/recognizer.ts` 和 `models/recognizer.onnx` 是项目部署模型文件，不应默认理解为 MIT 许可内容。模型发布时应在 GitHub Release 的说明中注明：

- 权重来源和训练检查点来源；
- 使用的训练数据集及其授权；
- 是否允许商业使用；
- 是否允许再分发和修改；
- 对应的字符词表版本和 SHA256。

在上述信息未确认前，建议将模型 Release 标注为“research use / non-commercial pending license review”。

## 4. TW25 数据集和历史地图图像

TW25 数据集、SynthMap+ 数据集以及用户自行导入的历史地图图像分别受其来源方的使用条款、版权和数据许可约束。本仓库不应在没有授权的情况下重新分发原始数据集或历史地图原图。

训练标注、示例图片和截图发布前，应检查是否包含受限制的地图扫描件、个人信息或其他不适合公开的内容。

## 5. Python 和运行时依赖

运行环境使用 PyQt5、PyTorch、OpenVINO、OpenCV、NumPy、Shapely 等第三方软件。它们各自有独立许可证，用户和发行者应根据实际安装或打包方式履行相应义务。

其中 PyQt5 的开源使用和商业分发条件需要特别确认；如果以商业方式分发包含 PyQt5 的桌面程序，应检查 GPL 或商业许可要求。

依赖版本和来源记录在：

- `requirements.txt`
- `requirements-gpu.txt`
- `requirements-cpu.txt`
- `pyproject.toml`

## 6. 发布前检查清单

- 确认根目录 `LICENSE` 是否符合项目实际授权意愿。
- 为模型 Release 添加模型来源和使用限制。
- 保留 DeepSolo、AdelaiDet、Detectron2 和运行时依赖的版权声明。
- 确认 TW25、SynthMap+ 和示例地图图片允许公开分发的范围。
- 为每个模型版本记录 SHA256 和对应评测结果。
- 如果发布 exe，检查 PyQt5、PyTorch、OpenVINO 和其他打包依赖的分发要求。
