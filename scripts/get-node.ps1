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
# v1.4.15.0: extract in a SHORT temp path (the project folder can be deeply nested; Windows PowerShell's Expand-Archive
# fails past 260 characters), then copy with robocopy, which handles long paths.
$tmp = Join-Path $env:TEMP ('pvcv_node_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
Write-Host "Downloading $file (one time, ~30 MB)..."
Invoke-WebRequest "$base/$file" -OutFile $zip -UseBasicParsing
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (Test-Path $tar) { & $tar -xf $zip -C $tmp; if ($LASTEXITCODE -ne 0) { throw "tar extraction failed ($LASTEXITCODE)" } }
else { Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp) }
$extracted = (Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1).FullName
if (-not $extracted) { throw 'The Node.js archive was empty.' }
$dest = Join-Path $root '.node'
if (Test-Path -LiteralPath $dest) { & cmd /c rmdir /s /q "`"$dest`"" }
& robocopy $extracted $dest /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying Node.js into the project folder failed (robocopy $LASTEXITCODE). Move the project to a shorter path such as D:\PVCV and run again." }
Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
Write-Host "Node.js ready in .node"
