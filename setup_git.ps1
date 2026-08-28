<#
.SYNOPSIS
    Initialises the thesis repository for Git: .gitignore, .gitattributes, folder
    scaffolding, a large-file guard and the first commit.

.DESCRIPTION
    Safe to re-run. It never overwrites an existing .gitignore or .gitattributes
    unless -Force is given, and it never deletes anything.

.PARAMETER RepoPath
    Repository root. Defaults to the current directory.

.PARAMETER RemoteUrl
    Optional. If given, sets 'origin' to this URL and pushes the branch.

.PARAMETER Branch
    Branch name for the first commit. Default: main.

.PARAMETER Force
    Overwrite an existing .gitignore / .gitattributes with the versions below.

.EXAMPLE
    .\setup_git.ps1

.EXAMPLE
    .\setup_git.ps1 -RemoteUrl "https://github.com/<user>/thesis-plasma-surrogate.git"
#>

[CmdletBinding()]
param(
    [string]$RepoPath      = (Get-Location).Path,
    [string]$RemoteUrl     = "",
    [string]$Branch        = "main",
    [string]$CommitMessage = "Initial commit: plasma reactor surrogate modelling pipeline",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok  ($msg) { Write-Host "    [ok]   $msg" -ForegroundColor Green }
function Write-Note($msg) { Write-Host "    [note] $msg" -ForegroundColor DarkGray }
function Write-Warn($msg) { Write-Host "    [warn] $msg" -ForegroundColor Yellow }

function Write-TextNoBom {
    # .gitignore breaks if the first line carries a UTF-8 BOM, so bypass Set-Content.
    param([string]$Path, [string]$Content)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function New-Placeholder {
    param([string]$Directory)
    if (-not (Test-Path $Directory)) { New-Item -ItemType Directory -Path $Directory | Out-Null }
    $keep = Join-Path $Directory ".gitkeep"
    if (-not (Test-Path $keep)) { New-Item -ItemType File -Path $keep | Out-Null }
}

# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------
$gitignore = @'
# ---------------------------------------------------------------------------
# Generated artefacts: everything here is reproducible from the raw dataset.
# The folders are kept (via .gitkeep), their contents are not.
# ---------------------------------------------------------------------------
data/*
!data/.gitkeep
models/*
!models/.gitkeep
results/*
!results/.gitkeep

# COMSOL and MATLAB exports, wherever they land
*.mph
*.mph.recovery*
cluster_dpOnt_*.txt
AGGREGATED_OPTIMIZED_SEQUENCE_*.txt
all_clusters_COMBINED_MASTER*.txt

# Trained weights, Optuna studies, cached objects
*.pt
*.pth
*.ckpt
*.db
*.sqlite
*.sqlite3
*.pkl
*.joblib

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.venv/
venv/
env/

# ---------------------------------------------------------------------------
# MATLAB
# ---------------------------------------------------------------------------
*.asv
*.m~
*.mat
*.mex*
*.slxc
slprj/

# ---------------------------------------------------------------------------
# LaTeX (if the thesis source is ever added here)
# ---------------------------------------------------------------------------
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.lof
*.lot
*.out
*.synctex.gz
*.toc

# ---------------------------------------------------------------------------
# OS / editors
# ---------------------------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
'@

$gitattributes = @'
# Normalise line endings on commit; check out native on each platform.
* text=auto

*.py   text
*.m    text
*.md   text
*.txt  text
*.csv  text
*.ps1  text eol=crlf
*.bat  text eol=crlf
*.sh   text eol=lf

*.pt   binary
*.pth  binary
*.db   binary
*.mph  binary
*.mat  binary
*.png  binary
*.jpg  binary
*.pdf  binary
*.xlsx binary
'@

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not on PATH. Install it from https://git-scm.com/download/win and reopen PowerShell."
}
Write-Ok (git --version)

if (-not (Test-Path $RepoPath)) { throw "RepoPath does not exist: $RepoPath" }
Set-Location $RepoPath
Write-Ok "Repository root: $RepoPath"

$userName  = git config --get user.name
$userEmail = git config --get user.email
if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Warn "Git identity is not set. Run these once, then re-run this script:"
    Write-Host '        git config --global user.name  "Your Name"'
    Write-Host '        git config --global user.email "you@example.com"'
    throw "Missing git identity."
}
Write-Ok "Committing as $userName <$userEmail>"

# ---------------------------------------------------------------------------
# 1. Initialise the repository
# ---------------------------------------------------------------------------
Write-Step "Initialising the repository"

if (Test-Path (Join-Path $RepoPath ".git")) {
    Write-Note "Already a git repository - continuing with the remaining steps."
} else {
    git init --initial-branch=$Branch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        # git < 2.28 has no --initial-branch
        git init | Out-Null
        git symbolic-ref HEAD "refs/heads/$Branch" | Out-Null
    }
    Write-Ok "Created a repository on branch '$Branch'"
}

# ---------------------------------------------------------------------------
# 2. .gitignore / .gitattributes
# ---------------------------------------------------------------------------
Write-Step "Writing .gitignore and .gitattributes"

foreach ($pair in @(
        @{ Name = ".gitignore";     Body = $gitignore },
        @{ Name = ".gitattributes"; Body = $gitattributes })) {

    $path = Join-Path $RepoPath $pair.Name
    if ((Test-Path $path) -and (-not $Force)) {
        Write-Note "$($pair.Name) already exists - left untouched (use -Force to replace)."
    } else {
        Write-TextNoBom -Path $path -Content $pair.Body
        Write-Ok "Wrote $($pair.Name)"
    }
}

# ---------------------------------------------------------------------------
# 3. Folder scaffolding
# ---------------------------------------------------------------------------
Write-Step "Creating the generated-output folders"

foreach ($dir in @("data", "models", "results")) {
    New-Placeholder (Join-Path $RepoPath $dir)
    Write-Ok "$dir\ ready (contents ignored, folder tracked)"
}
New-Item -ItemType Directory -Force -Path (Join-Path $RepoPath "data\processed\spatial_weights") | Out-Null

# ---------------------------------------------------------------------------
# 4. Stage and check what is about to be committed
# ---------------------------------------------------------------------------
Write-Step "Staging files"

git add -A
if ($LASTEXITCODE -ne 0) { throw "git add failed." }

$tracked = @(git ls-files)
if ($tracked.Count -eq 0) {
    Write-Warn "Nothing to commit - the working tree is empty or fully ignored."
    return
}

$oversize = @()
$large    = @()
foreach ($rel in $tracked) {
    $full = Join-Path $RepoPath $rel
    if (Test-Path $full) {
        $mb = (Get-Item $full).Length / 1MB
        if     ($mb -ge 95) { $oversize += ("{0}  ({1:N1} MB)" -f $rel, $mb) }
        elseif ($mb -ge 50) { $large    += ("{0}  ({1:N1} MB)" -f $rel, $mb) }
    }
}

if ($oversize.Count -gt 0) {
    git reset | Out-Null
    Write-Warn "These files are at or above GitHub's 100 MB hard limit:"
    $oversize | ForEach-Object { Write-Host "        $_" }
    throw "Staging was undone. Add these paths to .gitignore (or set up Git LFS) and re-run."
}
if ($large.Count -gt 0) {
    Write-Warn "Over GitHub's 50 MB warning threshold - consider Git LFS:"
    $large | ForEach-Object { Write-Host "        $_" }
}

Write-Ok "$($tracked.Count) file(s) staged"
git status --short

# ---------------------------------------------------------------------------
# 5. Commit
# ---------------------------------------------------------------------------
Write-Step "Committing"

$hasCommits = $true
git rev-parse --verify HEAD 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { $hasCommits = $false }

$pending = @(git diff --cached --name-only)
if ($pending.Count -eq 0 -and $hasCommits) {
    Write-Note "No staged changes - nothing to commit."
} else {
    $msg = if ($hasCommits) { "Update: repository housekeeping" } else { $CommitMessage }
    git commit -m $msg | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
    git branch -M $Branch
    Write-Ok "Committed on '$Branch': $msg"
}

# ---------------------------------------------------------------------------
# 6. Remote (optional)
# ---------------------------------------------------------------------------
if (-not [string]::IsNullOrWhiteSpace($RemoteUrl)) {
    Write-Step "Configuring the remote"

    $remotes = @(git remote)
    if ($remotes -contains "origin") {
        git remote set-url origin $RemoteUrl
        Write-Ok "Updated origin -> $RemoteUrl"
    } else {
        git remote add origin $RemoteUrl
        Write-Ok "Added origin -> $RemoteUrl"
    }

    Write-Note "Pushing (a browser or credential prompt may appear)..."
    git push -u origin $Branch
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Push failed. Check that the remote repository exists and is empty, then run: git push -u origin $Branch"
    } else {
        Write-Ok "Pushed to origin/$Branch"
    }
} else {
    Write-Step "Next step"
    Write-Host "    Create an empty repository on GitHub (no README, no .gitignore), then run:" -ForegroundColor DarkGray
    Write-Host "        git remote add origin https://github.com/<user>/<repo>.git"
    Write-Host "        git push -u origin $Branch"
}

Write-Host "`nDone.`n" -ForegroundColor Green
