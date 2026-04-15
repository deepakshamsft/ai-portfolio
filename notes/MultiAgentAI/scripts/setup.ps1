# setup.ps1 — Multi-Agent AI Notes: Dependency Setup (Windows / PowerShell)
# Installs every package needed to run the Jupyter notebooks across all 7
# chapters under notes/MultiAgentAI/.
#
# Run from the repo root or from this scripts/ folder:
#   .\notes\MultiAgentAI\scripts\setup.ps1
#
# What it does:
#   1. Verifies Python 3.9+ (3.11+ recommended)
#   2. Creates or reuses the repo-level .venv
#   3. Installs the MultiAgentAI notebook dependency stack
#   4. Registers a Jupyter kernel named "multi-agent-ai"
#   5. Generates all 7 chapter notebooks from the generation script
#   6. Verifies all core imports work

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Locate repo root (this script lives at notes/MultiAgentAI/scripts/)
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
Write-Host "  Multi-Agent AI Notes — Notebook Dependency Setup   " -ForegroundColor White
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

# ─── STEP 3: Package groups ────────────────────────────────────────────────────

Write-Step "Installing Multi-Agent AI packages"

Write-Group "Core notebook stack"
& pip install jupyter ipykernel --quiet
Write-Ok "jupyter, ipykernel"

Write-Group "Ch.1 — Message Formats (token counting)"
& pip install tiktoken --quiet
Write-Ok "tiktoken"

Write-Group "Ch.2 — MCP (Model Context Protocol SDK)"
& pip install mcp --quiet
Write-Ok "mcp"

Write-Group "Ch.3 — A2A protocol (HTTP / SSE)"
& pip install fastapi "uvicorn[standard]" httpx anyio --quiet
Write-Ok "fastapi, uvicorn, httpx, anyio"

Write-Group "Ch.4 — Event-driven agents (Redis Streams)"
& pip install redis --quiet
Write-Ok "redis"

Write-Group "Ch.5 — Shared Memory (Redis blackboard)"
# Already covered by redis above — add redis-py extras
& pip install redis --quiet
Write-Ok "redis (shared memory blackboard)"

Write-Group "Ch.6 — Trust & Sandboxing (validation)"
& pip install pydantic --quiet
Write-Ok "pydantic"

Write-Group "Ch.7 — Agent Frameworks (LangGraph, AutoGen, Semantic Kernel)"
& pip install langgraph langchain-core langchain-openai --quiet
Write-Ok "langgraph, langchain-core, langchain-openai"

& pip install "autogen-agentchat" --quiet
Write-Ok "autogen-agentchat"

& pip install semantic-kernel --quiet
Write-Ok "semantic-kernel"

Write-Group "Local model inference (Ollama Python client)"
& pip install ollama --quiet
Write-Ok "ollama"

# ─── STEP 4: Jupyter kernel registration ───────────────────────────────────────

Write-Step "Registering 'multi-agent-ai' Jupyter kernel"

& python -m ipykernel install --user --name multi-agent-ai --display-name "Multi-Agent AI"
Write-Ok "Kernel 'multi-agent-ai' registered"

# ─── STEP 5: Generate notebooks ────────────────────────────────────────────────

Write-Step "Generating chapter notebooks"

$GeneratorScript = Join-Path $ScriptDir "generate_notebooks.py"
if (Test-Path $GeneratorScript) {
    & python $GeneratorScript
    Write-Ok "All 7 chapter notebooks generated"
} else {
    Write-Warn "Generator script not found at $GeneratorScript — skipping notebook generation"
}

# ─── STEP 6: Smoke test imports ────────────────────────────────────────────────

Write-Step "Verifying imports"

$smokeTest = @"
import sys, importlib

results = []
packages = [
    ("tiktoken",           "Ch.1 token counting"),
    ("fastapi",            "Ch.3 A2A server"),
    ("httpx",              "Ch.3 A2A client"),
    ("redis",              "Ch.4/5 event-driven messaging + blackboard"),
    ("pydantic",           "Ch.6 schema validation"),
    ("langgraph",          "Ch.7 LangGraph"),
    ("langchain_core",     "Ch.7 LangChain core"),
    ("autogen",            "Ch.7 AutoGen"),
    ("semantic_kernel",    "Ch.7 Semantic Kernel"),
    ("ollama",             "local model client"),
]

for pkg, label in packages:
    try:
        importlib.import_module(pkg)
        results.append(("ok", pkg, label))
    except ImportError as e:
        results.append(("missing", pkg, label))

print()
for status, pkg, label in results:
    icon = "✓" if status == "ok" else "!"
    print(f"  {icon} {pkg:<25} {label}")

missing = [pkg for s, pkg, _ in results if s == "missing"]
if missing:
    print(f"\n  Some packages missing: {missing}")
    print("  Re-run this script or pip install manually.")
    sys.exit(1)
else:
    print("\n  All imports verified.")
"@

& python -c $smokeTest

# ─── Done ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  Setup complete.                                     " -ForegroundColor Green
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Open any chapter notebook in VS Code and select the 'Multi-Agent AI' kernel."
Write-Host ""
Write-Host "  Chapter notebooks:"
Write-Host "    Ch.1  Message Formats         MessageFormats/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.2  MCP                     MCP/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.3  A2A Protocol             A2A/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.4  Event-Driven Agents      EventDrivenAgents/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.5  Shared Memory            SharedMemory/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.6  Trust & Sandboxing       TrustAndSandboxing/notebook.ipynb" -ForegroundColor DarkGray
Write-Host "    Ch.7  Agent Frameworks         AgentFrameworks/notebook.ipynb" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Optional: run Ollama for live model cells in Ch.7:" -ForegroundColor DarkGray
Write-Host "    winget install Ollama.Ollama" -ForegroundColor DarkGray
Write-Host "    ollama pull phi3:mini" -ForegroundColor DarkGray
Write-Host ""
