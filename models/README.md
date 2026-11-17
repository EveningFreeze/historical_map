# 部署模型说明

模型目录包含 TorchScript 和 ONNX 两种部署表示，分别供 Windows GPU 和 OpenVINO
CPU 后端使用。两个模型输入固定为 `1×3×1280×1280`，输出检测 logits、字符 logits、
控制点和边界点。平台在 Windows 端负责切片、CTC 解码、边界恢复、NMS 和几何过滤。

当前它们是离线 Windows 部署版本，不应直接宣称与不同硬件后端逐项完全一致。
导出时使用参考实现替代了训练框架中的自定义可变形注意力算子；在当前测试
切片上文字序列一致，但完整验证图出现过实例数量差异。因此正式发布前必须完成
官方验证集的 CPU/GPU 指标对齐，并重新校准阈值。

模型文件使用 GitHub Release 分发，不以普通 Git 大文件提交。首次部署请在项目根目录运行：

```powershell
.\scripts\download_models.ps1 -Backend all
```

脚本会读取 `model_manifest.json`，下载与版本匹配的模型和词表，并校验文件大小与 SHA256。
