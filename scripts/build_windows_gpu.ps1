$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$python = Join-Path (Get-Location) ".venv_gpu\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "请先运行 .\scripts\setup_windows_gpu.ps1"
    exit 1
}
& $python -m pip install pyinstaller
& $python -m PyInstaller --noconfirm --clean --onedir `
    --name HistoricalMapPlatform-GPU `
    --add-data "models\recognizer.ts;models" `
    --add-data "models\recognizer_torchscript.json;models" `
    --add-data "models\recognizer.onnx;models" `
    --add-data "models\recognizer_openvino.json;models" `
    --add-data "models\tw25_cls_list.txt;models" `
    --hidden-import app.cuda_inference `
    --hidden-import app.openvino_inference `
    --hidden-import shapely `
    --collect-all torch `
    --collect-all openvino `
    run.py
Write-Host "已生成 dist\HistoricalMapPlatform-GPU。运行需要 NVIDIA 驱动；无 NVIDIA GPU 的电脑请使用 CPU 发布包。"
