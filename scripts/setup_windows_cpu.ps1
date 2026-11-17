$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$root = (Get-Location).Path
$venv = Join-Path $root ".venv_cpu"
$cache = Join-Path $root ".pip-cache"
$temp = Join-Path $root ".pip-tmp"
New-Item -ItemType Directory -Force -Path $cache,$temp | Out-Null
$env:PIP_CACHE_DIR = $cache
$env:TEMP = $temp
$env:TMP = $temp

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    python -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-cpu.txt
& $python -c "import torch; assert not torch.cuda.is_available(); print('CPU torch:', torch.__version__)"
& $python -c "import openvino; print('OpenVINO:', openvino.__version__)"
Write-Host "Windows CPU 环境安装完成。运行："
Write-Host ".\run_platform.ps1"
