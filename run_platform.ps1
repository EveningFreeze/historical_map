$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$candidates = @(
    (Join-Path $PSScriptRoot ".venv_gpu\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv_dml\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv_cpu\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
)
$systemPython = Get-Command python -ErrorAction SilentlyContinue
if ($systemPython) {
    $candidates += $systemPython.Source
}

$python = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) {
        continue
    }
    $pythonRoot = Split-Path (Split-Path $candidate -Parent) -Parent
    $pyqtPackage = Join-Path $pythonRoot "Lib\site-packages\PyQt5"
    if ($candidate -like "*\.venv_gpu\*" -and -not (Test-Path (Join-Path $pythonRoot "Lib\site-packages\torch"))) {
        continue
    }
    if (Test-Path $pyqtPackage) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    Write-Error "未找到 Python。请先运行 .\scripts\setup_windows_gpu.ps1 或 .\scripts\setup_windows_cpu.ps1"
    exit 1
}

$qtPlatforms = Join-Path (Split-Path $python) "..\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
if (Test-Path $qtPlatforms) {
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = (Resolve-Path $qtPlatforms).Path
}
Write-Host "使用 Python：$python"
& $python (Join-Path $PSScriptRoot "run.py")
