$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$root = (Resolve-Path .).Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "历史地图识别平台.lnk"

$gpuExe = Join-Path $root "dist\HistoricalMapPlatform-GPU\HistoricalMapPlatform-GPU.exe"
$cpuExe = Join-Path $root "dist\HistoricalMapPlatform\HistoricalMapPlatform.exe"
$launcher = Join-Path $root "run_platform.ps1"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.WorkingDirectory = $root
$shortcut.Description = "历史地图地名提取平台"

if (Test-Path $gpuExe) {
    $shortcut.TargetPath = $gpuExe
    $shortcut.Arguments = ""
    $shortcut.IconLocation = $gpuExe
} elseif (Test-Path $cpuExe) {
    $shortcut.TargetPath = $cpuExe
    $shortcut.Arguments = ""
    $shortcut.IconLocation = $cpuExe
} else {
    $shortcut.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcut.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
}

$shortcut.Save()
Write-Host "桌面快捷方式已创建：$shortcutPath"
