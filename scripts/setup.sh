#!/usr/bin/env bash
# setup.sh — AI/ML Dev Environment Setup (macOS / Linux)
# Provisions Python, a full AI/ML library stack, VS Code, Ollama, and a local SLM.
# Run from anywhere:
#   bash scripts/setup.sh
#
# Steps implemented so far:
#   1. Python + AI/ML libraries  ✔
#   2. VS Code install            ✔
#   3. Kilo Code (agentic AI) extension  ✔
#   4. Ollama server install & first launch  ✔
#   5. Lifecycle wiring (Ollama runs with VS Code)  ✔
#   6. Pull DeepSeek-R1 reasoning SLM for Kilo Code  ✔

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → repo root
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$REPO_ROOT/.venv"

# ─── Helpers ──────────────────────────────────────────────────────────────────

step()  { echo; echo "▶ $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ! $*"; }
fail()  { echo "  ✗ $*" >&2; exit 1; }
group() { echo; echo "  ── $*"; }

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    case "${ID:-}" in
        ubuntu|debian|linuxmint|pop)  OS="debian" ;;
        fedora|rhel|centos|rocky|alma) OS="fedora" ;;
        arch|manjaro)                  OS="arch"   ;;
        *)                             OS="linux"  ;;
    esac
fi

# ─── STEP 1: Python + AI/ML Libraries ────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 1/6"
echo "  Python + AI/ML Library Stack"
echo "══════════════════════════════════════════════"

# ─── 1a. Python ───────────────────────────────────────────────────────────────

step "Checking Python 3.11+"

PYTHON=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        major=$(echo "$ver" | sed 's/Python \([0-9]*\)\..*/\1/')
        minor=$(echo "$ver" | sed 's/Python [0-9]*\.\([0-9]*\)\..*/\1/')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            ok "$ver (meets 3.11+ requirement)"
            break
        elif [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            warn "$ver found — Python 3.11+ recommended; continuing with $ver"
            PYTHON="$candidate"
            break
        else
            warn "$ver found but Python 3.9+ is required"
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python not found — attempting install ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install python@3.11
                PYTHON="python3.11"
            else
                fail "Homebrew not found. Install it from https://brew.sh/ then re-run, or install Python manually from https://www.python.org/downloads/"
            fi
            ;;
        debian)
            sudo apt-get update -qq
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
            PYTHON="python3.11"
            ;;
        fedora)
            sudo dnf install -y python3.11
            PYTHON="python3.11"
            ;;
        arch)
            sudo pacman -Sy --noconfirm python
            PYTHON="python"
            ;;
        *)
            fail "Cannot auto-install Python on this system. Install Python 3.11+ manually from https://www.python.org/downloads/ and re-run."
            ;;
    esac
    ok "Python installed: $("$PYTHON" --version 2>&1)"
fi

# ─── 1b. pip ──────────────────────────────────────────────────────────────────

step "Checking pip"
if "$PYTHON" -m pip --version &>/dev/null; then
    ok "$("$PYTHON" -m pip --version)"
else
    warn "pip not available — bootstrapping ..."
    "$PYTHON" -m ensurepip --upgrade 2>/dev/null || {
        case "$OS" in
            debian) sudo apt-get install -y python3-pip ;;
            fedora) sudo dnf install -y python3-pip ;;
            macos)  brew install python@3.11 ;;
            *)      fail "Cannot bootstrap pip. Install it manually and re-run." ;;
        esac
    }
    ok "$("$PYTHON" -m pip --version)"
fi

# ─── 1c. Virtual environment ──────────────────────────────────────────────────

step "Virtual environment (.venv)"

if [ -d "$VENV_PATH" ]; then
    ok "Existing .venv found — reusing"
else
    warn "No .venv found — creating ..."
    "$PYTHON" -m venv "$VENV_PATH"
    ok "Created .venv at $VENV_PATH"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
ok "Activated .venv"

# Upgrade build tools quietly
python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip / setuptools / wheel up to date"

# ─── 1d. Package installation ─────────────────────────────────────────────────

step "Installing AI/ML package stack"

# Helper: install a group of packages
install_group() {
    local group_name="$1"
    shift
    local extra_args=()
    # Collect packages until "--extra-args" sentinel
    local packages=()
    local in_extra=0
    for arg in "$@"; do
        if [ "$arg" = "--extra-args" ]; then
            in_extra=1
        elif [ "$in_extra" -eq 1 ]; then
            extra_args+=("$arg")
        else
            packages+=("$arg")
        fi
    done

    group "$group_name"
    local installed
    installed=$(python -m pip list --format=columns 2>/dev/null | tail -n +3 | awk '{print tolower($1)}')

    for pkg in "${packages[@]}"; do
        # Normalise: strip extras/version specifiers for lookup
        local key
        key=$(echo "$pkg" | sed 's/\[.*\]//' | sed 's/[>=<!].*//' | tr '[:upper:]' '[:lower:]' | xargs)
        if echo "$installed" | grep -qx "$key"; then
            ok "$pkg already installed"
        else
            warn "$pkg missing — installing ..."
            if [ ${#extra_args[@]} -gt 0 ]; then
                python -m pip install "$pkg" "${extra_args[@]}" --quiet
            else
                python -m pip install "$pkg" --quiet
            fi
            ok "$pkg installed"
        fi
    done
}

# Core scientific stack
install_group "Core scientific stack" \
    numpy pandas scipy matplotlib seaborn

# Machine learning
install_group "Machine learning" \
    scikit-learn xgboost lightgbm

# Deep learning — TensorFlow
install_group "Deep learning / TensorFlow" \
    tensorflow tensorboard keras

# PyTorch — CPU-safe build (no CUDA required on stock machines)
install_group "PyTorch (CPU build)" \
    torch torchvision torchaudio \
    --extra-args --index-url https://download.pytorch.org/whl/cpu

# Notebook tooling
install_group "Notebook tooling" \
    notebook ipykernel ipywidgets jupyterlab

# Generative AI / LLM utilities
install_group "Generative AI / LLM utilities" \
    transformers diffusers accelerate datasets tokenizers \
    huggingface-hub openai langchain langchain-community \
    sentence-transformers faiss-cpu chromadb

# General utilities
install_group "Utilities" \
    python-dotenv tqdm pillow requests httpx pydantic

# Docs / study site (MkDocs Material — browse notes/ in a web browser)
# mkdocs-jupyter renders every notebook.ipynb as a page alongside the .md files.
install_group "Docs site (MkDocs Material)" \
    mkdocs-material pymdown-extensions mkdocs-jupyter

# Notebook extras — dependencies pulled in by per-notes setup scripts
#   notes/AIInfrastructure : mlflow
#   notes/MultiAgentAI     : tiktoken, mcp, fastapi, uvicorn, anyio, redis,
#                            langgraph, langchain-core, langchain-openai,
#                            autogen-agentchat, semantic-kernel, ollama
install_group "Notebook extras (AIInfrastructure + MultiAgentAI)" \
    mlflow tiktoken mcp fastapi "uvicorn[standard]" anyio redis \
    langgraph langchain-core langchain-openai \
    autogen-agentchat semantic-kernel ollama

# ─── 1e. Register Jupyter kernels ─────────────────────────────────────────────

step "Registering Jupyter kernels"
python -m ipykernel install --user --name "ai-ml-dev"         --display-name "AI/ML Dev (venv)"           &>/dev/null
ok "Kernel 'ai-ml-dev' registered"
python -m ipykernel install --user --name "ml-notes"          --display-name "ML Notes (venv)"            &>/dev/null
ok "Kernel 'ml-notes' registered"
python -m ipykernel install --user --name "ai-infrastructure" --display-name "Python (AI Infrastructure)" &>/dev/null
ok "Kernel 'ai-infrastructure' registered"
python -m ipykernel install --user --name "multi-agent-ai"    --display-name "Multi-Agent AI"             &>/dev/null
ok "Kernel 'multi-agent-ai' registered"

step "Setting default kernel on every notebook under notes/"
python "$SCRIPT_DIR/set_default_kernel.py" || warn "set_default_kernel.py exited non-zero"

# ─── Done ─────────────────────────────────────────────────────────────────────

# ─── STEP 2: Visual Studio Code ─────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 2/6"
echo "  Visual Studio Code"
echo "══════════════════════════════════════════════"

step "Checking for Visual Studio Code"

CODE_CMD=""
for candidate in code code-insiders; do
    if command -v "$candidate" &>/dev/null; then
        code_ver=$("$candidate" --version 2>&1 | head -1)
        if echo "$code_ver" | grep -qE '[0-9]+\.[0-9]+'; then
            CODE_CMD="$candidate"
            ok "VS Code $code_ver already installed ($candidate)"
            break
        fi
    fi
done

if [ -z "$CODE_CMD" ]; then
    warn "VS Code not found — installing ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install --cask visual-studio-code
            else
                fail "Homebrew not found. Install VS Code manually from https://code.visualstudio.com/ and re-run."
            fi
            ;;
        debian)
            # Add Microsoft apt repository
            sudo apt-get install -y wget gpg
            wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
                | gpg --dearmor \
                | sudo tee /usr/share/keyrings/microsoft-archive-keyring.gpg > /dev/null
            echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] \
https://packages.microsoft.com/repos/code stable main" \
                | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
            sudo apt-get update -qq
            sudo apt-get install -y code
            ;;
        fedora)
            sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
            sudo tee /etc/yum.repos.d/vscode.repo > /dev/null <<'EOF'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
EOF
            sudo dnf install -y code
            ;;
        arch)
            # Use AUR helper if available, otherwise advise manual install
            if command -v yay &>/dev/null; then
                yay -S --noconfirm visual-studio-code-bin
            elif command -v paru &>/dev/null; then
                paru -S --noconfirm visual-studio-code-bin
            else
                warn "AUR helper not found. Install VS Code manually: https://code.visualstudio.com/"
                warn "Continuing without VS Code — Steps 3–5 may fail."
            fi
            ;;
        *)
            warn "Cannot auto-install VS Code on this OS. Download from https://code.visualstudio.com/"
            warn "Continuing — Steps 3–5 may fail if 'code' is not on PATH."
            ;;
    esac

    # Refresh PATH and verify
    export PATH="$PATH:/usr/bin:/usr/local/bin"
    if command -v code &>/dev/null; then
        CODE_CMD="code"
        code_ver=$(code --version 2>&1 | head -1)
        ok "VS Code $code_ver installed successfully"
    else
        warn "'code' not on PATH yet. Restart your terminal after this script finishes, then re-run for remaining steps."
        CODE_CMD="code"   # optimistically continue
    fi
else
    ok "Skipping install — VS Code already present"
fi

# ─── STEP 3: Kilo Code (Agentic AI) Extension ───────────────────────────────
#
# Kilo Code is an open-source agentic coding assistant (fork of Roo/Cline) that
# can plan, edit files, and run commands. We point it at a locally-hosted
# DeepSeek-R1 reasoning SLM via Ollama (configured in Step 6d).

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 3/6"
echo "  Kilo Code — Agentic AI Extension"
echo "══════════════════════════════════════════════"

KILO_EXT_ID="kilocode.kilo-code"

step "Checking Kilo Code extension ($KILO_EXT_ID)"

EXTENSION_INSTALLED=false
if command -v "${CODE_CMD}" &>/dev/null; then
    if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$KILO_EXT_ID"; then
        ok "Kilo Code already installed"
        EXTENSION_INSTALLED=true
    fi
else
    warn "'${CODE_CMD}' not on PATH — skipping extension check"
fi

if [ "$EXTENSION_INSTALLED" = false ]; then
    warn "Kilo Code not found — installing ..."
    if command -v "${CODE_CMD}" &>/dev/null; then
        "${CODE_CMD}" --install-extension "$KILO_EXT_ID" --force &>/dev/null || true
        # Verify
        if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$KILO_EXT_ID"; then
            ok "Kilo Code installed successfully"
        else
            warn "Install ran but extension not detected yet — it may appear after VS Code restarts"
        fi
    else
        warn "Cannot install Kilo Code: 'code' not on PATH."
        warn "Install manually: open VS Code → Extensions → search 'Kilo Code' → Install"
    fi
fi

step "Kilo Code post-install configuration note"
echo ""
echo "  After launching VS Code:"
echo "    1. Open the Kilo Code sidebar (kangaroo icon on the Activity Bar)"
echo "    2. Click 'Settings' → API Provider: Ollama"
echo "    3. Base URL: http://localhost:11434   (the auto-discover button works too)"
echo "    4. Model: deepseek-r1:8b  (or deepseek-r1:1.5b on low-RAM machines)"
echo "    5. Save — Kilo Code will now drive agentic edits with DeepSeek-R1 reasoning"
echo ""

# ─── STEP 4: Ollama Server Install & First Launch ────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 4/6"
echo "  Ollama Local Inference Server"
echo "══════════════════════════════════════════════"

OLLAMA_PORT=11434
OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
PID_FILE="$REPO_ROOT/.ollama.pid"

# ── 4a. Install Ollama ────────────────────────────────────────────────────────

step "Checking Ollama binary"

OLLAMA_INSTALLED=false
if command -v ollama &>/dev/null; then
    ollama_ver=$(ollama --version 2>&1 || true)
    ok "Ollama already installed: $ollama_ver"
    OLLAMA_INSTALLED=true
fi

if [ "$OLLAMA_INSTALLED" = false ]; then
    warn "Ollama not found — installing ..."
    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                brew install ollama
            else
                fail "Homebrew not found. Install Ollama manually from https://ollama.com/download and re-run."
            fi
            ;;
        debian|fedora|arch|linux)
            # Official install script (works on most Linux distros)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            fail "Cannot auto-install Ollama on this OS. Download from https://ollama.com/download and re-run."
            ;;
    esac
    export PATH="$PATH:/usr/local/bin:/usr/bin"
    if command -v ollama &>/dev/null; then
        ok "Ollama installed: $(ollama --version 2>&1 || true)"
    else
        fail "Ollama installation completed but 'ollama' not found on PATH. Restart terminal and re-run."
    fi
fi

# ── 4b. Start the Ollama server ───────────────────────────────────────────────

step "Starting Ollama server"

# Check if already listening
SERVER_RUNNING=false
if curl -sf --max-time 3 "$OLLAMA_BASE_URL" &>/dev/null; then
    ok "Ollama server already running at $OLLAMA_BASE_URL"
    SERVER_RUNNING=true
fi

if [ "$SERVER_RUNNING" = false ]; then
    warn "Ollama server not running — starting in background ..."

    # Pin Ollama to a single loaded model with no parallelism, so the 8B
    # reasoning SLM owns the GPU/RAM exclusively while Kilo Code is working.
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_CONTEXT_LENGTH=4096

    # Start server detached, redirect output to a log file
    nohup env OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 \
        ollama serve > "$REPO_ROOT/.ollama.log" 2>&1 &
    OLLAMA_BG_PID=$!
    echo "$OLLAMA_BG_PID" > "$PID_FILE"
    ok "Ollama server started (PID $OLLAMA_BG_PID, log: .ollama.log)"
    ok "Single-model mode: OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_NUM_PARALLEL=1, OLLAMA_CONTEXT_LENGTH=4096"
    ok "PID saved to .ollama.pid"

    # Health-check with retries
    MAX_RETRIES=12
    RETRIES=0
    HEALTHY=false
    while [ "$RETRIES" -lt "$MAX_RETRIES" ] && [ "$HEALTHY" = false ]; do
        sleep 1
        if curl -sf --max-time 2 "$OLLAMA_BASE_URL" &>/dev/null; then
            HEALTHY=true
        fi
        RETRIES=$((RETRIES + 1))
    done

    if [ "$HEALTHY" = true ]; then
        ok "Ollama server is healthy at $OLLAMA_BASE_URL"
    else
        warn "Ollama server did not respond within ${MAX_RETRIES}s — it may still be starting up"
        warn "Check manually: curl $OLLAMA_BASE_URL  or  cat .ollama.log"
    fi
fi

# ─── STEP 5: Ollama Lifecycle Wiring ──────────────────────────────────────────
#
# Strategy: write .vscode/tasks.json with a folderOpen task that starts
# ollama serve, and a companion stop task.  VS Code has no native onClose
# hook, so we also write a small watcher script that monitors the 'code'
# process and stops Ollama when it exits.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 5/6"
echo "  Ollama Lifecycle Wiring"
echo "══════════════════════════════════════════════"

VSCODE_DIR="$REPO_ROOT/.vscode"
TASKS_JSON="$VSCODE_DIR/tasks.json"
WATCHER_SCRIPT="$REPO_ROOT/scripts/ollama-watcher.sh"

# ── 5a. Write .vscode/tasks.json ──────────────────────────────────────────────

step "Configuring .vscode/tasks.json"

mkdir -p "$VSCODE_DIR"

WRITE_TASKS=true
if [ -f "$TASKS_JSON" ]; then
    if grep -q 'ollama-start' "$TASKS_JSON" 2>/dev/null; then
        ok "tasks.json already contains ollama tasks — skipping"
        WRITE_TASKS=false
    else
        warn "tasks.json exists but has no ollama tasks — backing up and rewriting"
        cp "$TASKS_JSON" "${TASKS_JSON}.bak"
    fi
fi

if [ "$WRITE_TASKS" = true ]; then
    cat > "$TASKS_JSON" << 'TASKSJSON'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ollama-start",
            "type": "shell",
            "command": "bash",
            "args": [
                "${workspaceFolder}/scripts/ollama-watcher.sh"
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
            "command": "bash",
            "args": [
                "-c",
                "pid=$(cat '${workspaceFolder}/.ollama.pid' 2>/dev/null); [ -n \"$pid\" ] && kill \"$pid\" 2>/dev/null; pkill -x ollama 2>/dev/null; true"
            ],
            "presentation": {
                "reveal": "never",
                "showReuseMessage": false
            },
            "problemMatcher": []
        }
    ]
}
TASKSJSON
    ok "Written: .vscode/tasks.json"
    warn "ACTION REQUIRED: open VS Code → Terminal → Run Task → 'Allow Automatic Tasks'"
fi

# ── 5b. Write the watcher script ──────────────────────────────────────────────

step "Writing ollama-watcher.sh"

# Always rewrite: the watcher content evolves (env vars, model pins, ...) and
# we want re-running setup.sh to keep it in sync.
if [ -f "$WATCHER_SCRIPT" ]; then
    ok "Overwriting existing ollama-watcher.sh"
fi

cat > "$WATCHER_SCRIPT" << 'WATCHER'
#!/usr/bin/env bash
# ollama-watcher.sh
# Launched automatically when this VS Code workspace opens (via tasks.json folderOpen).
# - Starts ollama serve if not already running
# - Monitors the VS Code process
# - Stops ollama serve when VS Code exits

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$REPO_ROOT")"  # scripts/ -> repo root
PID_FILE="$REPO_ROOT/.ollama.pid"
OLLAMA_URL="http://localhost:11434"

is_ollama_running() {
    curl -sf --max-time 2 "$OLLAMA_URL" &>/dev/null
}

# Start ollama if not already up
if ! is_ollama_running; then
    # Pin to a single loaded model with no parallelism and a 4096-token ctx.
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_CONTEXT_LENGTH=4096
    nohup env OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 \
        ollama serve >> "$REPO_ROOT/.ollama.log" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
fi

# Wait until all VS Code windows are gone
while true; do
    sleep 5
    if ! pgrep -x code &>/dev/null && ! pgrep -x "Code" &>/dev/null; then
        # VS Code has exited — stop Ollama
        saved_pid=$(cat "$PID_FILE" 2>/dev/null || true)
        [ -n "$saved_pid" ] && kill "$saved_pid" 2>/dev/null || true
        pkill -x ollama 2>/dev/null || true
        rm -f "$PID_FILE"
        break
    fi
done
WATCHER
chmod +x "$WATCHER_SCRIPT"
ok "Written: scripts/ollama-watcher.sh"

# ── 5c. Workspace settings: make notebooks read-only in VS Code ─────────────
#
# Rationale: notebooks are edited LIVE in Jupyter Lab at http://localhost:8888.
# If VS Code also opens the same .ipynb with its own kernel, you get:
#   - concurrent writes that clobber each other
#   - two kernels holding GPU/RAM for the same notebook
# Marking *.ipynb read-only in this workspace keeps VS Code as a preview only.

step "Writing .vscode/settings.json (notebooks read-only in VS Code)"

SETTINGS_JSON="$VSCODE_DIR/settings.json"

if [ -f "$SETTINGS_JSON" ]; then
    python - "$SETTINGS_JSON" << 'PYEOF'
import json, sys
p = sys.argv[1]
try:
    with open(p, "r") as f:
        data = json.load(f)
except Exception:
    data = {}
ro = data.setdefault("files.readonlyInclude", {})
ro["**/*.ipynb"] = True
data["notebook.defaultKernel"] = "ai-ml-dev"
with open(p, "w") as f:
    json.dump(data, f, indent=4)
print("  \u2713 Merged read-only rule into existing .vscode/settings.json")
PYEOF
else
    cat > "$SETTINGS_JSON" << 'SETTINGSJSON'
{
    "files.readonlyInclude": {
        "**/*.ipynb": true
    },
    "notebook.defaultKernel": "ai-ml-dev"
}
SETTINGSJSON
    ok "Written: .vscode/settings.json"
fi

# ─── STEP 6: Pull DeepSeek-R1 Reasoning SLM ──────────────────────────────────
#
# DeepSeek-R1 is the reasoning model that powers Kilo Code's agentic planning.
#   Primary:  deepseek-r1:8b-llama-distill-q4_K_M  (~5 GB, needs ~10 GB free RAM)
#   Fallback: deepseek-r1:1.5b-qwen-distill-q4_0   (~1.1 GB, needs ~3 GB free RAM)
# Selection is automatic based on detected system RAM.
#
# After pulling, we derive a companion model tagged '-ctx4k' with
# `PARAMETER num_ctx 4096` baked in, so every client (Kilo Code, curl, raw API)
# gets a 4096-token context window without having to pass num_ctx explicitly.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 6/6"
echo "  Pull DeepSeek-R1 Reasoning SLM"
echo "══════════════════════════════════════════════"

PRIMARY_BASE="deepseek-r1:8b-llama-distill-q4_K_M"
FALLBACK_BASE="deepseek-r1:1.5b-qwen-distill-q4_0"
CTX_TOKENS=4096

# ── 6a. Detect system RAM ────────────────────────────────────────────────────

step "Detecting system RAM"

TOTAL_RAM_GB=0
case "$OS" in
    macos)
        total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        TOTAL_RAM_GB=$(( total_bytes / 1073741824 ))
        ;;
    debian|fedora|arch|linux|*)
        # /proc/meminfo reports kB
        mem_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
        TOTAL_RAM_GB=$(( mem_kb / 1048576 ))
        ;;
esac

ok "Total RAM: ${TOTAL_RAM_GB} GB"

if [ "$TOTAL_RAM_GB" -ge 10 ]; then
    BASE_MODEL="$PRIMARY_BASE"
    ok "RAM ≥ 10 GB — selecting primary base: $BASE_MODEL"
else
    BASE_MODEL="$FALLBACK_BASE"
    warn "RAM < 10 GB — selecting fallback base: $BASE_MODEL"
fi

# Derived tag: same base with '-ctx4k' suffix. This is what Kilo Code targets.
CHOSEN_MODEL="$(echo "$BASE_MODEL" | cut -d: -f1):$(echo "$BASE_MODEL" | cut -d: -f2)-ctx4k"
ok "Derived model (num_ctx=${CTX_TOKENS}): $CHOSEN_MODEL"

# ── 6b. Check if base model already pulled ──────────────────────────────────

step "Checking if $BASE_MODEL is already available"

BASE_PRESENT=false
if ollama list 2>/dev/null | grep -qF "$BASE_MODEL"; then
    ok "$BASE_MODEL already present in Ollama"
    BASE_PRESENT=true
fi

# ── 6c. Pull the base model ──────────────────────────────────────────────────

if [ "$BASE_PRESENT" = false ]; then
    step "Pulling $BASE_MODEL (this may take a few minutes on first run)"
    echo "  Downloading model — progress shown below:"
    echo ""
    if ollama pull "$BASE_MODEL"; then
        ok "$BASE_MODEL pulled successfully"
    else
        warn "Pull failed — check your internet connection and retry: ollama pull $BASE_MODEL"
    fi
else
    step "Skipping pull — $BASE_MODEL already present"
fi

# ── 6c'. Create the derived -ctx4k model with num_ctx=4096 baked in ─────────

step "Creating derived model $CHOSEN_MODEL with num_ctx=${CTX_TOKENS}"

if ollama list 2>/dev/null | grep -qF "$CHOSEN_MODEL"; then
    ok "$CHOSEN_MODEL already present — skipping create"
else
    MODELFILE="$REPO_ROOT/.ollama.Modelfile"
    cat > "$MODELFILE" << MFEOF
FROM $BASE_MODEL
PARAMETER num_ctx $CTX_TOKENS
MFEOF
    if ollama create "$CHOSEN_MODEL" -f "$MODELFILE"; then
        ok "Created $CHOSEN_MODEL (num_ctx=${CTX_TOKENS})"
    else
        warn "ollama create failed — Kilo Code will have to pass num_ctx itself"
        CHOSEN_MODEL="$BASE_MODEL"
    fi
    rm -f "$MODELFILE"
fi

# ── 6d. Configure Kilo Code to use the DeepSeek-R1 model ─────────────────────
#
# Kilo Code stores its API provider in extension global state (not plain
# settings.json), so this step writes a best-effort settings scaffold plus a
# Kilo Code profile JSON that the user can import from the sidebar.

step "Writing Kilo Code provider settings"

# Locate VS Code user settings directory
case "$OS" in
    macos)
        VSCODE_SETTINGS_DIR="$HOME/Library/Application Support/Code/User"
        ;;
    *)
        VSCODE_SETTINGS_DIR="$HOME/.config/Code/User"
        ;;
esac
VSCODE_SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

mkdir -p "$VSCODE_SETTINGS_DIR"

if [ -f "$VSCODE_SETTINGS_FILE" ]; then
    python3 - << PYEOF
import json
p = '$VSCODE_SETTINGS_FILE'
try:
    with open(p, 'r') as f:
        existing = json.load(f)
except Exception:
    existing = {}
patch = {
    'kilo-code.apiProvider': 'ollama',
    'kilo-code.ollamaBaseUrl': 'http://localhost:11434',
    'kilo-code.ollamaModelId': '${CHOSEN_MODEL}',
    'kilo-code.ollamaNumCtx': ${CTX_TOKENS},
    'kilo-code.modelMaxTokens': ${CTX_TOKENS},
    'kilo-code.autoApprovalEnabled': False,
}
existing.update(patch)
with open(p, 'w') as f:
    json.dump(existing, f, indent=4)
print('  ✓ Kilo Code settings merged into existing settings.json')
PYEOF
else
    cat > "$VSCODE_SETTINGS_FILE" << KILOEOF
{
  "kilo-code.apiProvider": "ollama",
  "kilo-code.ollamaBaseUrl": "http://localhost:11434",
  "kilo-code.ollamaModelId": "${CHOSEN_MODEL}",
  "kilo-code.ollamaNumCtx": ${CTX_TOKENS},
  "kilo-code.modelMaxTokens": ${CTX_TOKENS},
  "kilo-code.autoApprovalEnabled": false
}
KILOEOF
    ok "settings.json created with Kilo Code provider settings"
fi

# Write an importable Kilo Code profile (Settings → Profiles → Import in sidebar)
KILO_PROFILE="$REPO_ROOT/.vscode/kilo-code-profile.json"
mkdir -p "$(dirname "$KILO_PROFILE")"
cat > "$KILO_PROFILE" << KILOPROFEOF
{
  "name": "Local DeepSeek-R1 (Ollama, 4k ctx)",
  "apiProvider": "ollama",
  "ollamaBaseUrl": "http://localhost:11434",
  "ollamaModelId": "${CHOSEN_MODEL}",
  "ollamaNumCtx": ${CTX_TOKENS},
  "modelMaxTokens": ${CTX_TOKENS}
}
KILOPROFEOF
ok "Kilo Code profile written: .vscode/kilo-code-profile.json (import from Kilo sidebar)"

# ─── STEP 7: Launch Study Servers (Jupyter Lab + MkDocs) ─────────────────────
#
# Fixed local ports so bookmarks stay stable:
#   • Jupyter Lab  → http://localhost:8888   (hands-on coding in notebooks)
#   • MkDocs site  → http://localhost:8000   (read notes/ in a web browser)
#
# Both run as detached background processes so this script can exit.
# PIDs saved to .jupyter.pid / .mkdocs.pid for a later stop command.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 7/7"
echo "  Launch Study Servers (Jupyter + MkDocs)"
echo "══════════════════════════════════════════════"

JUPYTER_PORT=8888
MKDOCS_PORT=8000
JUPYTER_PID_FILE="$REPO_ROOT/.jupyter.pid"
MKDOCS_PID_FILE="$REPO_ROOT/.mkdocs.pid"
JUPYTER_LOG="$REPO_ROOT/.jupyter.log"
MKDOCS_LOG="$REPO_ROOT/.mkdocs.log"

port_in_use() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null
    elif command -v ss &>/dev/null; then
        ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port"
    else
        # Last resort: try to bind
        (echo > "/dev/tcp/127.0.0.1/$port") &>/dev/null
    fi
}

# ── 7a. Jupyter Lab ──────────────────────────────────────────────────────────

step "Starting Jupyter Lab on port $JUPYTER_PORT"

if port_in_use "$JUPYTER_PORT"; then
    ok "Port $JUPYTER_PORT already in use — assuming Jupyter Lab is running"
else
    nohup python -m jupyter lab \
        --no-browser \
        --ServerApp.ip=127.0.0.1 \
        --ServerApp.port="$JUPYTER_PORT" \
        --ServerApp.port_retries=0 \
        --ServerApp.root_dir="$REPO_ROOT" \
        --ServerApp.open_browser=False \
        > "$JUPYTER_LOG" 2>&1 &
    echo $! > "$JUPYTER_PID_FILE"
    ok "Jupyter Lab started (PID $(cat "$JUPYTER_PID_FILE")) — log: .jupyter.log"
    echo "    Check .jupyter.log for the one-time login token/URL."
fi

# ── 7b. MkDocs site ──────────────────────────────────────────────────────────

step "Starting MkDocs site on port $MKDOCS_PORT"

if port_in_use "$MKDOCS_PORT"; then
    ok "Port $MKDOCS_PORT already in use — assuming MkDocs is running"
else
    nohup python -m mkdocs serve \
        -f "$REPO_ROOT/mkdocs.yml" \
        -a "127.0.0.1:$MKDOCS_PORT" \
        > "$MKDOCS_LOG" 2>&1 &
    echo $! > "$MKDOCS_PID_FILE"
    ok "MkDocs started (PID $(cat "$MKDOCS_PID_FILE")) — log: .mkdocs.log"
fi

# ─── ALL DONE ─────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete (all 7 steps)"
echo ""
echo "  Python env  : $VENV_PATH"
echo "  Activate    : source .venv/bin/activate"
echo "  VS Code     : ${CODE_CMD}"
echo "  Kilo Code   : $KILO_EXT_ID"
echo "  Ollama      : $OLLAMA_BASE_URL  (single-model mode, ctx=${CTX_TOKENS})"
echo "  Reasoning   : $CHOSEN_MODEL (DeepSeek-R1, ${CTX_TOKENS}-token ctx)"
echo ""
echo "  Study servers (running in background):"
echo "    Hands-on notebooks  → http://localhost:$JUPYTER_PORT"
echo "    Reading (MkDocs)    → http://localhost:$MKDOCS_PORT"
echo ""
echo "  To stop them:"
echo "    kill \$(cat .jupyter.pid .mkdocs.pid 2>/dev/null)"
echo ""
echo "  Next: open VS Code in this folder — Ollama will start automatically."
echo "  If prompted, click 'Allow Automatic Tasks' to enable the watcher."
echo "══════════════════════════════════════════════"
echo ""
