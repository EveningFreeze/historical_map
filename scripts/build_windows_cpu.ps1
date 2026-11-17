$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$python = Join-Path (Get-Location) ".venv_cpu\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "请先运行 .\scripts\setup_windows_cpu.ps1"
    exit 1
}
& $python -m pip install pyinstaller
& $python -m PyInstaller --noconfirm --clean --onedir `
    --name HistoricalMapPlatform `
    --add-data "models\recognizer.ts;models" `
    --add-data "models\recognizer_torchscript.json;models" `
    --add-data "models\recognizer.onnx;models" `
    --add-data "models\recognizer_openvino.json;models" `
    --add-data "models\tw25_cls_list.txt;models" `
    --hidden-import app.cuda_inference `
    --hidden-import app.openvino_inference `
    --hidden-import shapely `
    --collect-all openvino `
    run.py
Write-Host "已生成 dist\HistoricalMapPlatform，可复制到其他 Windows 电脑离线使用。"
