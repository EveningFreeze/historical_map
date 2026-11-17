Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv_cpu\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "未找到 CPU 虚拟环境，请先运行：.\scripts\setup_windows_cpu.ps1"
    exit 1
}
$env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $PSScriptRoot ".venv_cpu\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
& $python run.py
