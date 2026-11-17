$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$venv = Join-Path (Get-Location) ".venv_dml"
if (-not (Test-Path $venv)) {
    $pythonLauncher = Get-Command python -ErrorAction Stop
    & $pythonLauncher.Source -m venv $venv
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-gpu-directml.txt
& $python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__); print('Providers:', ort.get_available_providers())"
Write-Host "DirectML GPU 环境已准备。可运行 .\run_platform.ps1。"
