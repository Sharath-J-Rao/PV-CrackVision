[CmdletBinding()]
param(
    [string]$ArchivePath = '',
    [string]$RepositoryPath = 'D:\Python_Projects\PV-Crack Vision'
)

$ErrorActionPreference = 'Stop'
if (-not $ArchivePath) { $ArchivePath = Join-Path $PSScriptRoot 'PV_CrackVision_v1.4.8.9_Windows_Desktop_Source.zip' }
$Branch = 'release/v1.4.8.9-desktop'
$Worktree = Join-Path $env:TEMP ("PV-CrackVision-v1.4.8.9-desktop-" + [guid]::NewGuid().ToString('N'))
$ExtractRoot = Join-Path $env:TEMP ("PV-CrackVision-v1.4.8.9-source-" + [guid]::NewGuid().ToString('N'))
$WorktreeAdded = $false

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    & git -C $RepositoryPath @GitArgs
    if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) { throw "Release ZIP not found: $ArchivePath" }
if (-not (Test-Path (Join-Path $RepositoryPath '.git'))) { throw "Git repository not found: $RepositoryPath" }
$origin = (& git -C $RepositoryPath remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -notmatch 'github\.com[:/]Sharath-J-Rao/PV-CrackVision(?:\.git)?$') {
    throw "The selected clone does not have the expected GitHub origin. Found: $origin"
}

try {
    New-Item -ItemType Directory -Path $ExtractRoot -Force | Out-Null
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractRoot -Force
    $source = Get-ChildItem -LiteralPath $ExtractRoot -Directory -Recurse | Where-Object {
        (Test-Path (Join-Path $_.FullName 'electron/main.cjs')) -and
        (Test-Path (Join-Path $_.FullName '.github/workflows/windows-desktop.yml'))
    } | Select-Object -First 1
    if (-not $source) { throw 'The supplied ZIP is not the desktop-ready v1.4.8.9 source package.' }
    $SourceRoot = $source.FullName

    Invoke-Git fetch origin
    $remoteMain = (& git -C $RepositoryPath rev-parse 'origin/main').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remoteMain) { throw 'Could not resolve origin/main.' }
    $remoteBranch = & git -C $RepositoryPath ls-remote --heads origin "refs/heads/$Branch"
    if ($LASTEXITCODE -ne 0) { throw 'Could not check the release branch on GitHub.' }
    if ($remoteBranch) { throw "Remote branch '$Branch' already exists. Refusing to overwrite it." }

    Invoke-Git worktree add -b $Branch $Worktree origin/main
    $WorktreeAdded = $true

    # Copy all release source except local environments, data, secrets and generated output.
    $excludeDirs = @('.git', 'node_modules', 'data', '.node', '.node_tmp', 'desktop-dist', 'dist', 'coverage', '.aistudio')
    Get-ChildItem -LiteralPath $SourceRoot -Force | Where-Object {
        $_.Name -notin $excludeDirs -and $_.Name -notin @('.env.local', '.env', '.env.production')
    } | ForEach-Object {
        $destination = Join-Path $Worktree $_.Name
        if ($_.PSIsContainer) {
            & robocopy $_.FullName $destination /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD .git node_modules data .node .node_tmp desktop-dist dist coverage __pycache__ /XF .env .env.local .env.production
            if ($LASTEXITCODE -gt 7) { throw "Failed to copy source folder '$($_.Name)' (robocopy exit $LASTEXITCODE)." }
        } else {
            Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        }
    }

    & git -C $Worktree add -A
    if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
    & git -C $Worktree diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged release contains whitespace errors; nothing was pushed.' }
    & git -C $Worktree diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { throw 'No source changes were found to commit.' }

    & git -C $Worktree commit -m 'Release PV CrackVision v1.4.8.9 Windows desktop'
    if ($LASTEXITCODE -ne 0) { throw 'Git commit failed.' }
    & git -C $Worktree push -u origin $Branch
    if ($LASTEXITCODE -ne 0) { throw 'Git push failed. Verify Git Credential Manager / GitHub authentication, then retry.' }

    Write-Host ''
    Write-Host 'Release branch pushed successfully:' -ForegroundColor Green
    Write-Host "https://github.com/Sharath-J-Rao/PV-CrackVision/tree/$Branch"
    Write-Host 'The Windows build workflow is triggered automatically by the branch push.' -ForegroundColor Green
    Write-Host 'Open GitHub Actions and download both EXE files from the artifact named PV-CrackVision-v1.4.8.9-Windows-x64.'
}
finally {
    if ($WorktreeAdded) { & git -C $RepositoryPath worktree remove --force $Worktree 2>$null }
    if (Test-Path $ExtractRoot) { Remove-Item $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue }
}
