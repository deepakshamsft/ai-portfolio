# setup.ps1 — AI Infrastructure Notes: Dependency Setup (Windows / PowerShell)
# Installs every package needed to run the Jupyter notebooks across all 10
# chapters under notes/AIInfrastructure/.
#
# Run from the repo root or from this scripts/ folder:
#   .\notes\AIInfrastructure\scripts\setup.ps1
#
# What it does:
#   1. Verifies Python 3.9+ (3.11+ recommended)
#   2. Creates or reuses the repo-level .venv
#   3. Installs the AIInfrastructure notebook dependency stack
#   4. Registers a Jupyter kernel named "ai-infrastructure"
#   5. Verifies all core imports work

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Locate repo root (this script lives at notes/AIInfrastructure/scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$VenvPath  = Join-Path $RepoRoot ".venv"

# ─── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }
function Write-Group { param($msg) Write-Host "`n  ── $msg" -ForegroundColor DarkCyan }

# ─── Header ────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI Infrastructure Notes — Notebook Dependency Setup" -ForegroundColor White
Write-Host "  Windows / PowerShell                               " -ForegroundColor White
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray

# ─── STEP 1: Python ────────────────────────────────────────────────────────────

Write-Step "Checking Python 3.9+"

$Python = $null
foreach ($candidate in @("python", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                $Python = $candidate
                Write-Ok "$ver"
                break
            } elseif ($major -ge 3 -and $minor -ge 9) {
                Write-Warn "$ver found — Python 3.11+ recommended; continuing"
                $Python = $candidate
                break
            } else {
                Write-Warn "$ver found — Python 3.9+ required"
            }
        }
    } catch { }
}

if (-not $Python) {
    Write-Warn "Python not found — attempting install via winget ..."
    try {
        winget install --id Python.Python.3.11 --source winget --silent `
            --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $Python = "python"
        Write-Ok "Installed: $(& python --version 2>&1)"
    } catch {
        Write-Fail "Could not install Python automatically. Install Python 3.11+ from https://www.python.org/downloads/ and re-run."
    }
}

# ─── STEP 2: Virtual environment ───────────────────────────────────────────────

Write-Step "Virtual environment (.venv at repo root)"

if (Test-Path $VenvPath) {
    Write-Ok "Existing .venv found at $VenvPath — reusing"
} else {
    Write-Warn "No .venv found — creating at $VenvPath ..."
    & $Python -m venv $VenvPath
    Write-Ok "Created .venv"
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Fail "Cannot find venv activation script at $ActivateScript"
}
. $ActivateScript
Write-Ok "Activated .venv"

& python -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok "pip / setuptools / wheel up to date"

# ─── STEP 3: Package installation ──────────────────────────────────────────────

Write-Step "Installing AI Infrastructure notebook dependencies"

function Install-Group {
    param(
        [string]   $GroupName,
        [string[]] $Packages
    )
    Write-Group $GroupName
    $installed = (& python -m pip list --format=columns 2>&1 |
                  Select-Object -Skip 2) -replace '\s+.*', '' |
                  ForEach-Object { $_.ToLower() }

    foreach ($pkg in $Packages) {
        $key = ($pkg -replace '\[.*\]', '' -replace '[>=<!\s].*', '').ToLower().Trim()
        if ($installed -contains $key) {
            Write-Ok "$pkg already installed"
        } else {
            Write-Warn "$pkg missing — installing ..."
            & python -m pip install $pkg --quiet
            if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install $pkg" }
            Write-Ok "$pkg installed"
        }
    }
}

# ── Core scientific stack (Chapters 1–10, all notebooks) ──
# numpy      : all arithmetic intensity calculations, roofline models, simulators
# pandas     : GPU spec databases, result tables, cost models
# matplotlib : all plots — roofline, batching curves, cost comparisons
# scipy      : used in Ch.3 (quantization error statistics) and Ch.12 (clustering metrics)
Install-Group "Core scientific stack (all chapters)" @(
    "numpy",
    "pandas",
    "scipy",
    "matplotlib"
)

# ── Jupyter tooling ──
# ipykernel      : required to run .ipynb notebooks
# ipywidgets     : interactive sliders for sensitivity analyses (optional but used in exercises)
# jupyter        : notebook server (skip if using VS Code's built-in Jupyter extension)
Install-Group "Jupyter tooling" @(
    "ipykernel",
    "ipywidgets",
    "jupyter"
)

# ── MLOps chapter (Ch.9) ──
# mlflow         : local experiment tracking, model registry, artifact logging
Install-Group "MLOps — experiment tracking (Ch.9)" @(
    "mlflow"
)

# ── Optional: heavier libraries ──
# These are NOT required for any notebook but unlock the optional cells in Ch.3
# (quantization) and Ch.9 (model registry) that call real model APIs.
# Commented out by default to keep install fast for the calculator-only workflow.
#
# Install-Group "Optional: real model inference (Ch.3 optional cells)" @(
#     "torch",                 # PyTorch CPU — needed only for the optional GPTQ live demo
#     "transformers",          # HuggingFace — optional Ch.3 perplexity measurement
#     "bitsandbytes"           # INT8/INT4 via bitsandbytes — optional Ch.3 quantization
# )

# ─── STEP 4: Jupyter kernel ────────────────────────────────────────────────────

Write-Step "Registering Jupyter kernel 'ai-infrastructure'"

try {
    & python -m ipykernel install --user --name "ai-infrastructure" `
        --display-name "Python (AI Infrastructure)"
    Write-Ok "Kernel registered: 'ai-infrastructure'"
} catch {
    Write-Warn "Could not register Jupyter kernel — you can still open notebooks in VS Code directly"
}

# ─── STEP 5: Smoke test ────────────────────────────────────────────────────────

Write-Step "Smoke-testing core imports"

$SmokeTest = @"
import sys
errors = []
libs = {
    "numpy":      "import numpy as np; np.zeros(3)",
    "pandas":     "import pandas as pd; pd.DataFrame()",
    "matplotlib": "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
    "scipy":      "import scipy",
    "mlflow":     "import mlflow",
    "ipykernel":  "import ipykernel",
}
for name, code in libs.items():
    try:
        exec(code)
        print(f'  OK  {name}')
    except Exception as e:
        errors.append(name)
        print(f'  FAIL {name}: {e}')
if errors:
    print(f'\nFailed imports: {errors}')
    sys.exit(1)
else:
    print('\nAll imports OK.')
"@

& python -c $SmokeTest
if ($LASTEXITCODE -ne 0) {
    Write-Fail "One or more imports failed — check the output above"
}
Write-Ok "All core imports verified"

# ─── Done ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  To start a notebook:" -ForegroundColor White
Write-Host "    • Open the .ipynb file in VS Code and select the" -ForegroundColor White
Write-Host "      'Python (AI Infrastructure)' kernel, OR" -ForegroundColor White
Write-Host "    • Run: jupyter notebook" -ForegroundColor White
Write-Host ""
Write-Host "  Chapters ready to run:" -ForegroundColor White
Write-Host "    Ch.1  GPU Architecture          GPUArchitecture/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.2  Memory Budgets             MemoryAndComputeBudgets/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.3  Quantization               QuantizationAndPrecision/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.4  Distributed Training       ParallelismAndDistributedTraining/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.5  Inference Optimization     InferenceOptimization/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.6  Serving Frameworks         ServingFrameworks/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.7  Networking & Clusters      NetworkingAndClusterArchitecture/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.8  Cloud AI Infrastructure    CloudAIInfrastructure/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.9  MLOps                      MLOpsAndExperimentManagement/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.10 Production AI Platform     ProductionAIPlatform/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
