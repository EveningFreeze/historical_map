$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv_gpu\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "未找到 GPU 虚拟环境，请先运行：.\scripts\setup_windows_gpu.ps1"
    exit 1
}
$env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $PSScriptRoot ".venv_gpu\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
& $python run.py
