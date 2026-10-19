$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Push-Location $PSScriptRoot
try {
    $uv = Join-Path $PSScriptRoot '.bootstrap\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        $uv = (Get-Command uv -ErrorAction Stop).Source
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime\python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $PSScriptRoot '.runtime\bin'
    $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
    $env:TEMP = Join-Path $PSScriptRoot '.runtime\temp'
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    $env:UV_HTTP_TIMEOUT = '180'
    & $uv python install 3.12.15
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $wheel = Join-Path $PSScriptRoot '.cache\downloads\torch-2.13.0+cu126-cp312-cp312-win_amd64.whl'
    if (Test-Path -LiteralPath $wheel) {
        if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne '380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2') {
            throw 'PyTorch wheel checksum differs from uv.lock.'
        }
        & $uv sync --frozen --extra cu126 --python 3.12.15 --no-default-groups --no-install-package torch
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
        & $uv pip install --offline --no-deps $wheel
    } else {
        & $uv sync --frozen --extra cu126 --python 3.12.15 --no-default-groups
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $uv pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    & .\run-course.ps1 check
    & .\run-course.ps1 download
} finally { Pop-Location }
