# setup.ps1 — AI/ML Dev Environment Setup (Windows / PowerShell)
# Provisions Python, a full AI/ML library stack, VS Code, Ollama, and a local SLM.
# Run from anywhere:
#   .\scripts\setup.ps1
#
# Steps implemented so far:
#   1. Python + AI/ML libraries  ✔
#   2. VS Code install            ✔
#   3. Twinny (Ollama Copilot) extension  ✔
#   4. Ollama server install & first launch  ✔
#   5. Lifecycle wiring (Ollama runs with VS Code)  ✔
#   6. Pull best SLM for coding/reasoning  ✔

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# scripts/ → repo root
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }
function Write-Group { param($msg) Write-Host "`n  ── $msg" -ForegroundColor DarkCyan }

# ─── STEP 1: Python + AI/ML Libraries ────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 1/6" -ForegroundColor White
Write-Host "  Python + AI/ML Library Stack" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

# ─── 1a. Python ───────────────────────────────────────────────────────────────

Write-Step "Checking Python 3.11+"

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
                Write-Warn "$ver found — Python 3.11+ is recommended; continuing with $ver"
                $Python = $candidate
                break
            } else {
                Write-Warn "$ver found but Python 3.9+ is required"
            }
        }
    } catch { }
}

if (-not $Python) {
    Write-Warn "Python not found — attempting install via winget ..."
    try {
        winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
        # Refresh PATH for this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $Python = "python"
        $ver = & python --version 2>&1
        Write-Ok "Installed: $ver"
    } catch {
        Write-Fail "Could not install Python automatically. Install Python 3.11+ from https://www.python.org/downloads/ and re-run."
    }
}

# ─── 1b. pip ──────────────────────────────────────────────────────────────────

Write-Step "Checking pip"
try {
    $pipVer = & $Python -m pip --version 2>&1
    Write-Ok $pipVer
} catch {
    Write-Warn "pip not available — bootstrapping ..."
    & $Python -m ensurepip --upgrade
    $pipVer = & $Python -m pip --version 2>&1
    Write-Ok $pipVer
}

# ─── 1c. Virtual environment ──────────────────────────────────────────────────

Write-Step "Virtual environment (.venv)"

if (Test-Path $VenvPath) {
    Write-Ok "Existing .venv found — reusing"
} else {
    Write-Warn "No .venv found — creating ..."
    & $Python -m venv $VenvPath
    Write-Ok "Created .venv at $VenvPath"
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Fail "Cannot find venv activation script at $ActivateScript"
}
. $ActivateScript
Write-Ok "Activated .venv"

# Upgrade pip + build tools quietly
& python -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok "pip / setuptools / wheel up to date"

# ─── 1d. Package installation ─────────────────────────────────────────────────

Write-Step "Installing AI/ML package stack"

function Install-Group {
    param(
        [string]   $GroupName,
        [string[]] $Packages,
        [string]   $ExtraArgs = ""
    )
    Write-Group $GroupName
    $installed = (& python -m pip list --format=columns 2>&1 |
                  Select-Object -Skip 2) -replace '\s+.*', '' |
                  ForEach-Object { $_.ToLower() }

    foreach ($pkg in $Packages) {
        # Normalise: strip extras/version for display key
        $key = ($pkg -replace '\[.*\]', '' -replace '[>=<!\s].*', '').ToLower().Trim()
        if ($installed -contains $key) {
            Write-Ok "$pkg already installed"
        } else {
            Write-Warn "$pkg missing — installing ..."
            if ($ExtraArgs) {
                & python -m pip install $pkg $ExtraArgs.Split(" ") --quiet
            } else {
                & python -m pip install $pkg --quiet
            }
            if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install $pkg" }
            Write-Ok "$pkg installed"
        }
    }
}

# Core scientific stack
Install-Group "Core scientific stack" @(
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn"
)

# Machine learning
Install-Group "Machine learning" @(
    "scikit-learn",
    "xgboost",
    "lightgbm"
)

# Deep learning — TensorFlow
Install-Group "Deep learning / TensorFlow" @(
    "tensorflow",
    "tensorboard",
    "keras"
)

# PyTorch — CPU-safe build (no CUDA required)
Install-Group "PyTorch (CPU build)" @(
    "torch",
    "torchvision",
    "torchaudio"
) "--index-url https://download.pytorch.org/whl/cpu"

# Notebook tooling
Install-Group "Notebook tooling" @(
    "notebook",
    "ipykernel",
    "ipywidgets",
    "jupyterlab"
)

# Generative AI / LLM utilities
Install-Group "Generative AI / LLM utilities" @(
    "transformers",
    "diffusers",
    "accelerate",
    "datasets",
    "tokenizers",
    "huggingface-hub",
    "openai",
    "langchain",
    "langchain-community",
    "sentence-transformers",
    "faiss-cpu",
    "chromadb"
)

# General utilities
Install-Group "Utilities" @(
    "python-dotenv",
    "tqdm",
    "pillow",
    "requests",
    "httpx",
    "pydantic"
)

# Docs / study site (MkDocs Material — browse notes/ in a web browser)
# mkdocs-jupyter renders every notebook.ipynb as a page alongside the .md files.
Install-Group "Docs site (MkDocs Material)" @(
    "mkdocs-material",
    "pymdown-extensions",
    "mkdocs-jupyter"
)

# Notebook extras — dependencies pulled in by per-notes setup scripts
#   notes/AIInfrastructure : mlflow
#   notes/MultiAgentAI     : tiktoken, mcp, fastapi, uvicorn, anyio, redis,
#                            langgraph, langchain-core, langchain-openai,
#                            autogen-agentchat, semantic-kernel, ollama
Install-Group "Notebook extras (AIInfrastructure + MultiAgentAI)" @(
    "mlflow",
    "tiktoken",
    "mcp",
    "fastapi",
    "uvicorn[standard]",
    "anyio",
    "redis",
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "autogen-agentchat",
    "semantic-kernel",
    "ollama"
)

# ─── 1e. Register Jupyter kernels ─────────────────────────────────────────────

Write-Step "Registering Jupyter kernels"
& python -m ipykernel install --user --name "ai-ml-dev"          --display-name "AI/ML Dev (venv)"           2>&1 | Out-Null
Write-Ok "Kernel 'ai-ml-dev' registered"
& python -m ipykernel install --user --name "ml-notes"           --display-name "ML Notes (venv)"            2>&1 | Out-Null
Write-Ok "Kernel 'ml-notes' registered"
& python -m ipykernel install --user --name "ai-infrastructure"  --display-name "Python (AI Infrastructure)" 2>&1 | Out-Null
Write-Ok "Kernel 'ai-infrastructure' registered"
& python -m ipykernel install --user --name "multi-agent-ai"     --display-name "Multi-Agent AI"             2>&1 | Out-Null
Write-Ok "Kernel 'multi-agent-ai' registered"

Write-Step "Setting default kernel on every notebook under notes/"
& python (Join-Path $PSScriptRoot "set_default_kernel.py")
if ($LASTEXITCODE -ne 0) { Write-Warn "set_default_kernel.py exited with code $LASTEXITCODE" }

# ─── STEP 2: Visual Studio Code ─────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 2/6" -ForegroundColor White
Write-Host "  Visual Studio Code" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

Write-Step "Checking for Visual Studio Code"

$CodeCmd = $null
foreach ($candidate in @("code", "code-insiders")) {
    try {
        $codeVer = & $candidate --version 2>&1 | Select-Object -First 1
        if ($codeVer -match '\d+\.\d+') {
            $CodeCmd = $candidate
            Write-Ok "VS Code $codeVer already installed ($candidate)"
            break
        }
    } catch { }
}

if (-not $CodeCmd) {
    Write-Warn "VS Code not found — installing via winget ..."
    try {
        winget install --id Microsoft.VisualStudioCode `
            --source winget `
            --silent `
            --accept-package-agreements `
            --accept-source-agreements

        # Refresh PATH so 'code' is immediately available
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Confirm
        $codeVer = & code --version 2>&1 | Select-Object -First 1
        if ($codeVer -match '\d+\.\d+') {
            $CodeCmd = "code"
            Write-Ok "VS Code $codeVer installed successfully"
        } else {
            Write-Warn "winget completed but 'code' is not yet on PATH."
            Write-Warn "Restart your terminal after this script finishes, then re-run for remaining steps."
            $CodeCmd = "code"   # optimistically continue; extensions added in Step 3
        }
    } catch {
        Write-Fail "Could not install VS Code automatically. Download from https://code.visualstudio.com/ and re-run."
    }
} else {
    Write-Ok "Skipping install — VS Code already present"
}

# ─── STEP 3: Twinny (Ollama AI Copilot) Extension ───────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 3/6" -ForegroundColor White
Write-Host "  Twinny — Ollama AI Copilot Extension" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$TwinnyExtId = "rjmacarthy.twinny"

Write-Step "Checking Twinny extension ($TwinnyExtId)"

$extensionInstalled = $false
try {
    $extList = & $CodeCmd --list-extensions 2>&1
    if ($extList -match [regex]::Escape($TwinnyExtId)) {
        Write-Ok "Twinny already installed"
        $extensionInstalled = $true
    }
} catch {
    Write-Warn "Could not query VS Code extensions — will attempt install anyway"
}

if (-not $extensionInstalled) {
    Write-Warn "Twinny not found — installing ..."
    try {
        & $CodeCmd --install-extension $TwinnyExtId --force 2>&1 | Out-Null
        # Verify
        $extList = & $CodeCmd --list-extensions 2>&1
        if ($extList -match [regex]::Escape($TwinnyExtId)) {
            Write-Ok "Twinny installed successfully"
        } else {
            Write-Warn "Install command ran but extension not detected yet — it may appear after VS Code restarts"
        }
    } catch {
        Write-Warn "Could not install Twinny automatically."
        Write-Warn "Install manually: open VS Code → Extensions → search 'Twinny' → Install"
    }
}

Write-Step "Twinny post-install configuration note"
Write-Host "" 
Write-Host "  After launching VS Code:" -ForegroundColor White
Write-Host "    1. Open the Twinny sidebar (robot icon on the Activity Bar)" -ForegroundColor DarkGray
Write-Host "    2. Click the settings cog → choose provider: Ollama" -ForegroundColor DarkGray
Write-Host "    3. Set hostname: localhost   port: 11434" -ForegroundColor DarkGray
Write-Host "    4. Set the chat model and FIM model (Step 6 will pull the model)" -ForegroundColor DarkGray
Write-Host ""

# ─── STEP 4: Ollama Server Install & First Launch ────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 4/6" -ForegroundColor White
Write-Host "  Ollama Local Inference Server" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$OllamaPort = 11434
$OllamaBaseUrl = "http://localhost:$OllamaPort"

# ── 4a. Install Ollama ────────────────────────────────────────────────────────

Write-Step "Checking Ollama binary"

$OllamaInstalled = $false
try {
    $ollamaVer = & ollama --version 2>&1
    if ($ollamaVer -match 'ollama') {
        Write-Ok "Ollama already installed: $ollamaVer"
        $OllamaInstalled = $true
    }
} catch { }

if (-not $OllamaInstalled) {
    Write-Warn "Ollama not found — installing via winget ..."
    try {
        winget install --id Ollama.Ollama `
            --source winget `
            --silent `
            --accept-package-agreements `
            --accept-source-agreements

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        $ollamaVer = & ollama --version 2>&1
        Write-Ok "Ollama installed: $ollamaVer"
    } catch {
        Write-Fail "Could not install Ollama automatically. Download from https://ollama.com/download and re-run."
    }
}

# ── 4b. Start the Ollama server ───────────────────────────────────────────────

Write-Step "Starting Ollama server"

# Check if it is already listening
$serverRunning = $false
try {
    $resp = Invoke-WebRequest -Uri $OllamaBaseUrl -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) {
        Write-Ok "Ollama server already running at $OllamaBaseUrl"
        $serverRunning = $true
    }
} catch { }

if (-not $serverRunning) {
    Write-Warn "Ollama server not running — starting in background ..."

    # Launch as a hidden background job so this script keeps running
    $OllamaJob = Start-Job -ScriptBlock { ollama serve } -Name "OllamaServe"
    Write-Ok "Ollama server started (background job id: $($OllamaJob.Id))"

    # Save PID file so the lifecycle step (Step 5) can stop it
    $pidFile = Join-Path $RepoRoot ".ollama.pid"
    # The job spawns a child process; wait briefly then capture the PID
    Start-Sleep -Seconds 3
    $ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ollamaProc) {
        $ollamaProc.Id | Set-Content $pidFile
        Write-Ok "Ollama PID $($ollamaProc.Id) saved to .ollama.pid"
    } else {
        Write-Warn "Could not locate ollama process to save PID — lifecycle management in Step 5 may need manual setup"
    }

    # Health-check with retries
    $maxRetries = 10
    $retries = 0
    $healthy = $false
    while ($retries -lt $maxRetries -and -not $healthy) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-WebRequest -Uri $OllamaBaseUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) { $healthy = $true }
        } catch { }
        $retries++
    }

    if ($healthy) {
        Write-Ok "Ollama server is healthy at $OllamaBaseUrl"
    } else {
        Write-Warn "Ollama server did not respond within ${maxRetries}s — it may still be starting up"
        Write-Warn "Run 'ollama serve' manually and check http://localhost:$OllamaPort"
    }
}

# ─── STEP 5: Ollama Lifecycle Wiring ──────────────────────────────────────────
#
# Strategy: write .vscode/tasks.json with a folderOpen task that starts
# ollama serve, and a companion stop task.  VS Code has no native onClose
# hook, so we also write a small wrapper script that monitors the 'code'
# process and stops Ollama when it exits.

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 5/6" -ForegroundColor White
Write-Host "  Ollama Lifecycle Wiring" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$VscodeDirPath  = Join-Path $RepoRoot ".vscode"
$TasksJsonPath  = Join-Path $VscodeDirPath "tasks.json"
$WatcherScript  = Join-Path $RepoRoot "scripts\ollama-watcher.ps1"

# ── 5a. Write .vscode/tasks.json ──────────────────────────────────────────────

Write-Step "Configuring .vscode/tasks.json"

if (-not (Test-Path $VscodeDirPath)) {
    New-Item -ItemType Directory -Path $VscodeDirPath | Out-Null
}

# Only write if tasks.json doesn't already contain our tasks
$writeTasksJson = $true
if (Test-Path $TasksJsonPath) {
    $existing = Get-Content $TasksJsonPath -Raw
    if ($existing -match 'ollama-start') {
        Write-Ok "tasks.json already contains ollama tasks — skipping"
        $writeTasksJson = $false
    } else {
        Write-Warn "tasks.json exists but has no ollama tasks — merging not supported; backing up and rewriting"
        Copy-Item $TasksJsonPath "$TasksJsonPath.bak" -Force
    }
}

if ($writeTasksJson) {
    $tasksJson = @'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ollama-start",
            "type": "shell",
            "command": "powershell",
            "args": [
                "-NonInteractive",
                "-File",
                "${workspaceFolder}/scripts/ollama-watcher.ps1"
            ],
            "runOptions": {
                "runOn": "folderOpen"
            },
            "presentation": {
                "reveal": "never",
                "panel": "dedicated",
                "showReuseMessage": false
            },
            "problemMatcher": []
        },
        {
            "label": "ollama-stop",
            "type": "shell",
            "command": "powershell",
            "args": [
                "-NonInteractive",
                "-Command",
                "$p = Get-Content '${workspaceFolder}/.ollama.pid' -ErrorAction SilentlyContinue; if ($p) { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue }"
            ],
            "presentation": {
                "reveal": "never",
                "showReuseMessage": false
            },
            "problemMatcher": []
        }
    ]
}
'@
    Set-Content -Path $TasksJsonPath -Value $tasksJson -Encoding UTF8
    Write-Ok "Written: .vscode/tasks.json"
    Write-Warn "ACTION REQUIRED: open VS Code → Terminal → Run Task → 'Allow Automatic Tasks' to enable folderOpen trigger"
}

# ── 5b. Write the watcher script ──────────────────────────────────────────────

Write-Step "Writing ollama-watcher.ps1"

if (Test-Path $WatcherScript) {
    Write-Ok "ollama-watcher.ps1 already exists — skipping"
} else {
    $watcherContent = @'
# ollama-watcher.ps1
# Launched automatically when this VS Code workspace opens (via tasks.json folderOpen).
# - Starts ollama serve if not already running
# - Monitors the VS Code process
# - Stops ollama serve when VS Code exits

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$PidFile   = Join-Path $RepoRoot ".ollama.pid"
$OllamaUrl = "http://localhost:11434"

function Is-OllamaRunning {
    try { $null = Invoke-WebRequest $OllamaUrl -UseBasicParsing -TimeoutSec 2 -EA Stop; return $true } catch { return $false }
}

# Start ollama if not already up
if (-not (Is-OllamaRunning)) {
    $job = Start-Job -ScriptBlock { ollama serve }
    Start-Sleep -Seconds 3
    $proc = Get-Process -Name "ollama" -EA SilentlyContinue | Select-Object -First 1
    if ($proc) { $proc.Id | Set-Content $PidFile }
}

# Wait until all VS Code windows are closed
while ($true) {
    Start-Sleep -Seconds 5
    $codeRunning = Get-Process -Name "code" -ErrorAction SilentlyContinue
    if (-not $codeRunning) {
        # VS Code has exited — stop Ollama
        $savedPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($savedPid) {
            Stop-Process -Id ([int]$savedPid) -Force -ErrorAction SilentlyContinue
        }
        # Belt-and-suspenders: kill any remaining ollama processes
        Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        break
    }
}
'@
    Set-Content -Path $WatcherScript -Value $watcherContent -Encoding UTF8
    Write-Ok "Written: scripts/ollama-watcher.ps1"
}

# ── 5c. Workspace settings: make notebooks read-only in VS Code ─────────────
#
# Rationale: notebooks are edited LIVE in Jupyter Lab at http://localhost:8888.
# If VS Code also opens the same .ipynb with its own kernel, you get:
#   - concurrent writes that clobber each other
#   - two kernels holding GPU/RAM for the same notebook
# Marking *.ipynb read-only in this workspace keeps VS Code as a preview only.

Write-Step "Writing .vscode/settings.json (notebooks read-only in VS Code)"

$SettingsJsonPath = Join-Path $VscodeDirPath "settings.json"
$readOnlyPatch = [ordered]@{
    "files.readonlyInclude" = [ordered]@{ "**/*.ipynb" = $true }
    "notebook.defaultKernel" = "ai-ml-dev"
}

if (Test-Path $SettingsJsonPath) {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $existing = Get-Content $SettingsJsonPath -Raw | ConvertFrom-Json -AsHashtable
        } else {
            $jsonObj = Get-Content $SettingsJsonPath -Raw | ConvertFrom-Json
            $existing = @{}
            $jsonObj.PSObject.Properties | ForEach-Object { $existing[$_.Name] = $_.Value }
        }
    } catch {
        $existing = @{}
    }
    # Merge files.readonlyInclude map
    if (-not $existing.ContainsKey("files.readonlyInclude")) {
        $existing["files.readonlyInclude"] = @{}
    }
    $existing["files.readonlyInclude"]["**/*.ipynb"] = $true
    $existing["notebook.defaultKernel"] = "ai-ml-dev"
    $existing | ConvertTo-Json -Depth 10 | Set-Content $SettingsJsonPath -Encoding UTF8
    Write-Ok "Merged read-only rule into existing .vscode/settings.json"
} else {
    $readOnlyPatch | ConvertTo-Json -Depth 10 | Set-Content $SettingsJsonPath -Encoding UTF8
    Write-Ok "Written: .vscode/settings.json"
}

# ─── STEP 6: Pull Best SLM for AI/ML Coding ──────────────────────────────────
#
# Primary:  qwen2.5-coder:7b  (~4.7 GB, needs ~8 GB free RAM)
# Fallback: phi3.5            (~2.2 GB, needs ~4 GB free RAM)
# Selection is automatic based on detected system RAM.

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 6/6" -ForegroundColor White
Write-Host "  Pull Best Local SLM" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

# ── 6a. Detect system RAM ────────────────────────────────────────────────────

Write-Step "Detecting system RAM"

$TotalRamGB = 0
try {
    $cs = Get-CimInstance -ClassName Win32_ComputerSystem
    $TotalRamGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    Write-Ok "Total RAM: ${TotalRamGB} GB"
} catch {
    Write-Warn "Could not detect RAM — defaulting to conservative model"
    $TotalRamGB = 0
}

# Choose model based on available RAM
$PrimaryModel  = "qwen2.5-coder:7b"
$FallbackModel = "phi3.5"
$ChosenModel   = if ($TotalRamGB -ge 10) { $PrimaryModel } else { $FallbackModel }

if ($TotalRamGB -ge 10) {
    Write-Ok "RAM ≥ 10 GB — selecting primary model: $ChosenModel"
} else {
    Write-Warn "RAM < 10 GB — selecting fallback model: $ChosenModel"
}

# ── 6b. Check if model already pulled ───────────────────────────────────────────

Write-Step "Checking if $ChosenModel is already available"

$modelPresent = $false
try {
    $modelList = & ollama list 2>&1
    # Normalise: qwen2.5-coder:7b appears as "qwen2.5-coder:7b" in the list
    $modelKey = $ChosenModel.Split(":")[0].ToLower()
    if ($modelList -match [regex]::Escape($modelKey)) {
        Write-Ok "$ChosenModel already present in Ollama"
        $modelPresent = $true
    }
} catch {
    Write-Warn "Could not query ollama list — will attempt pull"
}

# ── 6c. Pull the model ────────────────────────────────────────────────────────

if (-not $modelPresent) {
    Write-Step "Pulling $ChosenModel (this may take a few minutes on first run)"
    Write-Host "  Downloading model — progress shown below:" -ForegroundColor DarkGray
    Write-Host ""
    & ollama pull $ChosenModel
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "$ChosenModel pulled successfully"
    } else {
        Write-Warn "Pull exited with code $LASTEXITCODE — check your internet connection and retry: ollama pull $ChosenModel"
    }
} else {
    Write-Step "Skipping pull — $ChosenModel already present"
}

# ── 6d. Configure Twinny to use the model ────────────────────────────────────────

Write-Step "Writing Twinny model settings to VS Code user settings"

$VsUserSettingsDir  = Join-Path $env:APPDATA "Code\User"
$VsSettingsPath     = Join-Path $VsUserSettingsDir "settings.json"

if (-not (Test-Path $VsUserSettingsDir)) {
    New-Item -ItemType Directory -Path $VsUserSettingsDir -Force | Out-Null
}

$twinnySettings = [ordered]@{
    "twinny.ollamaApiHostname"    = "localhost"
    "twinny.ollamaApiPort"        = 11434
    "twinny.chatModelName"        = $ChosenModel
    "twinny.fimModelName"         = $ChosenModel
    "twinny.apiProvider"          = "ollama"
}

if (Test-Path $VsSettingsPath) {
    try {
        # ConvertFrom-Json -AsHashtable requires PS 6+; handle PS 5.1 (Windows default) too
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $existing = Get-Content $VsSettingsPath -Raw | ConvertFrom-Json -AsHashtable
        } else {
            $jsonObj = Get-Content $VsSettingsPath -Raw | ConvertFrom-Json
            $existing = @{}
            $jsonObj.PSObject.Properties | ForEach-Object { $existing[$_.Name] = $_.Value }
        }
    } catch {
        $existing = @{}
        Write-Warn "Could not parse existing settings.json — will merge carefully"
    }
    foreach ($key in $twinnySettings.Keys) {
        $existing[$key] = $twinnySettings[$key]
    }
    $existing | ConvertTo-Json -Depth 10 | Set-Content $VsSettingsPath -Encoding UTF8
    Write-Ok "Twinny settings merged into existing settings.json"
} else {
    $twinnySettings | ConvertTo-Json -Depth 10 | Set-Content $VsSettingsPath -Encoding UTF8
    Write-Ok "settings.json created with Twinny model settings"
}

# ─── STEP 7: Launch Study Servers (Jupyter Lab + MkDocs) ─────────────────────
#
# Fixed local ports so bookmarks stay stable:
#   • Jupyter Lab  → http://localhost:8888   (hands-on coding in notebooks)
#   • MkDocs site  → http://localhost:8000   (read notes/ in a web browser)
#
# Both run as detached background processes so this script can exit.
# PIDs are saved to .jupyter.pid / .mkdocs.pid for a later stop command.

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  AI/ML Dev Environment Setup — Step 7/7" -ForegroundColor White
Write-Host "  Launch Study Servers (Jupyter + MkDocs)" -ForegroundColor White
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray

$JupyterPort  = 8888
$MkdocsPort   = 8000
$JupyterPid   = Join-Path $RepoRoot ".jupyter.pid"
$MkdocsPid    = Join-Path $RepoRoot ".mkdocs.pid"
$JupyterLog   = Join-Path $RepoRoot ".jupyter.log"
$MkdocsLog    = Join-Path $RepoRoot ".mkdocs.log"
$VenvPython   = Join-Path $VenvPath "Scripts\python.exe"

function Test-PortInUse { param([int]$Port)
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return [bool]$conn
    } catch { return $false }
}

# ── 7a. Jupyter Lab ──────────────────────────────────────────────────────────

Write-Step "Starting Jupyter Lab on port $JupyterPort"

if (Test-PortInUse -Port $JupyterPort) {
    Write-Ok "Port $JupyterPort already in use — assuming Jupyter Lab is running"
} else {
    $jupyterArgs = @(
        "-m", "jupyter", "lab",
        "--no-browser",
        "--ServerApp.ip=127.0.0.1",
        "--ServerApp.port=$JupyterPort",
        "--ServerApp.port_retries=0",
        "--ServerApp.root_dir=$RepoRoot",
        "--ServerApp.open_browser=False"
    )
    $jupyterProc = Start-Process -FilePath $VenvPython `
        -ArgumentList $jupyterArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $JupyterLog `
        -RedirectStandardError  "$JupyterLog.err" `
        -WindowStyle Hidden `
        -PassThru
    $jupyterProc.Id | Set-Content $JupyterPid
    Write-Ok "Jupyter Lab started (PID $($jupyterProc.Id)) — log: .jupyter.log"
    Write-Host "    Check .jupyter.log for the one-time login token/URL." -ForegroundColor DarkGray
}

# ── 7b. MkDocs site ──────────────────────────────────────────────────────────

Write-Step "Starting MkDocs site on port $MkdocsPort"

if (Test-PortInUse -Port $MkdocsPort) {
    Write-Ok "Port $MkdocsPort already in use — assuming MkDocs is running"
} else {
    $mkdocsArgs = @(
        "-m", "mkdocs", "serve",
        "-f", (Join-Path $RepoRoot "mkdocs.yml"),
        "-a", "127.0.0.1:$MkdocsPort"
    )
    $mkdocsProc = Start-Process -FilePath $VenvPython `
        -ArgumentList $mkdocsArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $MkdocsLog `
        -RedirectStandardError  "$MkdocsLog.err" `
        -WindowStyle Hidden `
        -PassThru
    $mkdocsProc.Id | Set-Content $MkdocsPid
    Write-Ok "MkDocs started (PID $($mkdocsProc.Id)) — log: .mkdocs.log"
}

# ─── ALL DONE ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  Setup complete (all 7 steps)" -ForegroundColor Green
Write-Host "" 
Write-Host "  Python env  : $VenvPath" -ForegroundColor White
Write-Host "  Activate    : .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  VS Code     : $CodeCmd" -ForegroundColor White
Write-Host "  Twinny ext  : $TwinnyExtId" -ForegroundColor White
Write-Host "  Ollama      : $OllamaBaseUrl" -ForegroundColor White
Write-Host "  SLM model   : $ChosenModel" -ForegroundColor White
Write-Host ""
Write-Host "  Study servers (running in background):" -ForegroundColor Cyan
Write-Host "    Hands-on notebooks  → http://localhost:$JupyterPort" -ForegroundColor White
Write-Host "    Reading (MkDocs)    → http://localhost:$MkdocsPort"  -ForegroundColor White
Write-Host ""
Write-Host "  To stop them:" -ForegroundColor DarkGray
Write-Host "    Get-Content .jupyter.pid,.mkdocs.pid | % { Stop-Process -Id ([int]`$_) -Force }" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Next: open VS Code in this folder — Ollama will start automatically." -ForegroundColor Cyan
Write-Host "  If prompted, click 'Allow Automatic Tasks' to enable the watcher." -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "" 
Write-Host "  Python env  : $VenvPath" -ForegroundColor White
Write-Host "  Activate    : .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  VS Code     : $CodeCmd" -ForegroundColor White
Write-Host "  Twinny ext  : $TwinnyExtId" -ForegroundColor White
Write-Host "  Ollama      : $OllamaBaseUrl" -ForegroundColor White
Write-Host "  SLM model   : $ChosenModel" -ForegroundColor White
Write-Host ""
Write-Host "  Next: open VS Code in this folder — Ollama will start automatically." -ForegroundColor Cyan
Write-Host "  If prompted, click 'Allow Automatic Tasks' to enable the watcher." -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""