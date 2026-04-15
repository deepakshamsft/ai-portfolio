# setup.ps1 — ML Notes Environment Setup (Windows / PowerShell)
# Verifies dependencies, installs missing ones, and launches Jupyter at notes/ML
# Run from anywhere:
#   .\notes\scripts\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# notes/scripts/ → notes/ → repo root
$RepoRoot  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$NotesPath = Join-Path $RepoRoot "notes\ML"
$VenvPath  = Join-Path $RepoRoot ".venv"

$RequiredPackages = @(
    "notebook",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "tensorboard",
    "scipy",
    "ipykernel"
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }

# ─── 1. Python ────────────────────────────────────────────────────────────────

Write-Step "Checking Python"

$Python = $null
foreach ($candidate in @("python", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                $Python = $candidate
                Write-Ok "$ver"
                break
            } else {
                Write-Warn "$ver found but Python 3.9+ is required"
            }
        }
    } catch { }
}

if (-not $Python) {
    Write-Fail "Python 3.9+ not found. Install from https://www.python.org/downloads/ and re-run this script."
}

# ─── 2. pip ───────────────────────────────────────────────────────────────────

Write-Step "Checking pip"
try {
    $pipVer = & $Python -m pip --version 2>&1
    Write-Ok $pipVer
} catch {
    Write-Fail "pip not available. Run: $Python -m ensurepip --upgrade"
}

# ─── 3. Virtual environment ───────────────────────────────────────────────────

Write-Step "Virtual environment"

if (Test-Path $VenvPath) {
    Write-Ok "Existing venv found at .venv"
} else {
    Write-Warn "No venv found — creating one at .venv ..."
    & $Python -m venv $VenvPath
    Write-Ok "Created .venv"
}

# Activate
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Fail "Could not find venv activation script at $ActivateScript"
}
. $ActivateScript
Write-Ok "Activated .venv"

# Upgrade pip inside the venv quietly
& python -m pip install --upgrade pip --quiet

# ─── 4. Packages ──────────────────────────────────────────────────────────────

Write-Step "Checking / installing packages"

$InstalledRaw = & python -m pip list --format=columns 2>&1
$Installed = ($InstalledRaw | Select-Object -Skip 2) -replace '\s+.*', '' | ForEach-Object { $_.ToLower() }

foreach ($pkg in $RequiredPackages) {
    $pkgLower = $pkg.ToLower()
    if ($Installed -contains $pkgLower) {
        Write-Ok "$pkg already installed"
    } else {
        Write-Warn "$pkg missing — installing ..."
        & python -m pip install $pkg --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to install $pkg"
        }
        Write-Ok "$pkg installed"
    }
}

# ─── 5. Register kernel ───────────────────────────────────────────────────────

Write-Step "Registering Jupyter kernel"
& python -m ipykernel install --user --name "ml-notes" --display-name "ML Notes (venv)" 2>&1 | Out-Null
Write-Ok "Kernel 'ml-notes' registered"

# ─── 6. Verify notes/ML exists ────────────────────────────────────────────────

Write-Step "Verifying notes/ML path"
if (-not (Test-Path $NotesPath)) {
    Write-Warn "notes/ML does not exist yet — creating directory ..."
    New-Item -ItemType Directory -Path $NotesPath | Out-Null
}
Write-Ok "notes/ML is ready at $NotesPath"

# ─── 7. Launch Jupyter ────────────────────────────────────────────────────────

Write-Step "Launching Jupyter Notebook"
Write-Host ""
Write-Host "  Opening browser at http://localhost:8888" -ForegroundColor White
Write-Host "  Root directory : $NotesPath" -ForegroundColor White
Write-Host "  Press Ctrl+C to stop the server.`n" -ForegroundColor White

& jupyter notebook --notebook-dir="$NotesPath"
