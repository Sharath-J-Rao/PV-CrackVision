param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,
    [ValidateRange(10, 900)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$healthUri = "http://127.0.0.1:$Port/api/health"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

Write-Host "Waiting for PV CrackVision API at $healthUri ..."
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Method Get -Uri $healthUri -TimeoutSec 2
        if ($health.ok -eq $true) {
            $appUri = "http://127.0.0.1:$Port"
            Write-Host "PV CrackVision is healthy. Opening $appUri"
            Start-Process $appUri
            exit 0
        }
    }
    catch {
        # Server is still starting; retry until the timeout.
    }
    Start-Sleep -Seconds 1
}

Write-Error "PV CrackVision did not become healthy within $TimeoutSeconds seconds. Check the start.bat window for the actual Node/Vite startup error."
exit 1
