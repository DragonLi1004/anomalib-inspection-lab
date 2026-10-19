param(
    [ValidateSet('check', 'download', 'baseline', 'predict', 'gui', 'download-full')][string]$Action = 'check',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$ImagePath,
    [switch]$OpenBrowser
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Push-Location $PSScriptRoot
try {
    $env:TEMP = Join-Path $PSScriptRoot '.runtime\temp'
    $env:TMP = $env:TEMP
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache\matplotlib'
    $env:XDG_CACHE_HOME = Join-Path $PSScriptRoot '.cache'
    $env:TORCH_HOME = Join-Path $PSScriptRoot 'course\models\torch'
    $env:HF_HOME = Join-Path $PSScriptRoot 'course\models\huggingface'
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup-course.ps1 first.' }
    switch ($Action) {
        'gui' {
            if ($OpenBrowser) { & $python course\gui.py --open-browser }
            else { & $python course\gui.py }
        }
        'download-full' { & $python course\download_mvtec.py }
        'check' { & $python course\check_env.py }
        'download' { & $python course\prepare_assets.py }
        'baseline' { & $python course\baseline.py --device $Device }
        'predict' {
            if (-not $ImagePath) { throw 'Specify -ImagePath for prediction.' }
            & $python course\predict.py --device $Device --image $ImagePath
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Course command failed: $LASTEXITCODE" }
} finally { Pop-Location }
