[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { throw 'Node.js is required to build. Install Node.js 22.12+ or 24+ first.' }
$version = (& node -p 'process.versions.node').Trim()
$parts = $version.Split('.')
$major = [int]$parts[0]
$minor = [int]$parts[1]
if ($major -lt 22 -or ($major -eq 22 -and $minor -lt 12)) { throw "Node.js 22.12+ or 24+ is required. Found $version." }

Write-Host 'Installing project dependencies...'
& npm install --no-audit --no-fund
if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
Write-Host 'Installing Electron and the Windows packager...'
& npm install --no-save --no-package-lock electron@45.2.3 electron-builder@26.18.1
if ($LASTEXITCODE -ne 0) { throw 'Electron tooling installation failed.' }
Write-Host 'Building production assets...'
& npm run build
if ($LASTEXITCODE -ne 0) { throw 'Vite production build failed.' }
Write-Host 'Building Windows installer and portable EXE...'
& npx electron-builder --win --x64 --config electron-builder.config.cjs --publish never
if ($LASTEXITCODE -ne 0) { throw 'Windows desktop packaging failed.' }
Write-Host 'Desktop deliverables:' -ForegroundColor Green
Get-ChildItem (Join-Path $PSScriptRoot 'desktop-dist') -Filter *.exe -File | Select-Object Name, Length, FullName
