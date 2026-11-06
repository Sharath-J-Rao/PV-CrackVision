# Downloads a portable Node.js 22 LTS into <project>\.node (no admin rights, no installer, no prompts).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
$base = 'https://nodejs.org/dist/latest-v22.x'
$sums = (Invoke-WebRequest "$base/SHASUMS256.txt" -UseBasicParsing).Content -split "`n"
$line = $sums | Where-Object { $_ -match "node-v[\d\.]+-win-$arch\.zip\s*$" } | Select-Object -First 1
if (-not $line) { throw "Could not find a Node.js win-$arch build at $base" }
$file = ($line -split '\s+')[-1].Trim()
$root = Split-Path $PSScriptRoot -Parent
$zip = Join-Path $env:TEMP $file
$tmp = Join-Path $root '.node_tmp'
Write-Host "Downloading $file (one time, ~30 MB)..."
Invoke-WebRequest "$base/$file" -OutFile $zip -UseBasicParsing
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
Expand-Archive $zip -DestinationPath $tmp -Force
if (Test-Path "$root\.node") { Remove-Item "$root\.node" -Recurse -Force }
Move-Item (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName "$root\.node"
Remove-Item $tmp -Recurse -Force
Remove-Item $zip -Force
Write-Host "Node.js ready in .node"
