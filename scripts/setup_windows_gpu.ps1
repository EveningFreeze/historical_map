$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$venv = Join-Path (Get-Location) ".venv_gpu"
if (-not (Test-Path $venv)) {
    $pythonLauncher = Get-Command python -ErrorAction Stop
    & $pythonLauncher.Source -m venv $venv
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-gpu.txt
& $python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Write-Host "GPU 环境已准备。运行 .\run_platform.ps1 启动平台。"
