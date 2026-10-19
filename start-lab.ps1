$ErrorActionPreference = 'Stop'
$url = 'http://127.0.0.1:8765'
try {
    $state = Invoke-RestMethod "$url/api/state" -TimeoutSec 2
    if ($state.token -and $null -ne $state.runs) { Start-Process $url; exit 0 }
} catch { }
& "$PSScriptRoot\run-course.ps1" gui -OpenBrowser
