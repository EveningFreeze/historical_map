param(
    [ValidateSet("all", "cuda", "openvino", "torchscript")]
    [string]$Backend = "all",
    [string]$ManifestPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$root = (Get-Location).Path
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $root "models\model_manifest.json"
} elseif (-not [System.IO.Path]::IsPathRooted($ManifestPath)) {
    $ManifestPath = Join-Path $root $ManifestPath
}

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "模型清单不存在：$ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseBaseUrl = [string]$manifest.release_base_url
if ([string]::IsNullOrWhiteSpace($releaseBaseUrl)) {
    throw "模型清单缺少 release_base_url：$ManifestPath"
}

$targetRoot = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

if ($Backend -eq "all") {
    $artifacts = @($manifest.artifacts)
} else {
    $artifacts = @($manifest.artifacts | Where-Object {
        $_.backend -eq $Backend -or $_.backend -eq "shared"
    })
}

if ($artifacts.Count -eq 0) {
    throw "没有找到后端 '$Backend' 对应的模型文件。"
}

$headers = @{ "User-Agent" = "HistoricalMapPlatform/ModelDownloader" }
foreach ($artifact in $artifacts) {
    $name = [string]$artifact.name
    $expectedHash = ([string]$artifact.sha256).ToUpperInvariant()
    $expectedSize = [int64]$artifact.size_bytes
    $target = Join-Path $targetRoot $name
    $temporary = "$target.download"
    $url = if ($artifact.url) {
        [string]$artifact.url
    } else {
        "$($releaseBaseUrl.TrimEnd('/'))/$name"
    }

    if (Test-Path -LiteralPath $target) {
        $existing = Get-Item -LiteralPath $target
        $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($existingHash -eq $expectedHash -and $existing.Length -eq $expectedSize) {
            Write-Host "已校验，跳过：$name"
            continue
        }
        if (-not $Force) {
            throw "文件已存在但校验失败：$target。确认模型版本后使用 -Force 覆盖。"
        }
        Remove-Item -LiteralPath $target -Force
    }

    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Force
    }

    Write-Host "下载：$name"
    Write-Host "地址：$url"
    Invoke-WebRequest -Uri $url -OutFile $temporary -Headers $headers -UseBasicParsing

    $downloaded = Get-Item -LiteralPath $temporary
    $downloadedHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($downloaded.Length -ne $expectedSize -or $downloadedHash -ne $expectedHash) {
        Remove-Item -LiteralPath $temporary -Force
        throw "下载文件校验失败：$name。期望大小 $expectedSize，实际大小 $($downloaded.Length)。"
    }

    Move-Item -LiteralPath $temporary -Destination $target -Force
    Write-Host "完成并校验：$name"
}

Write-Host "模型准备完成：版本 $($manifest.model_version)，后端 $Backend"
