#!/usr/bin/env bash
# setup.sh — AI/ML Dev Environment Setup (macOS / Linux)
# Provisions Python, a full AI/ML library stack, VS Code, Ollama, and a local SLM.
# Run from anywhere:
#   bash scripts/setup.sh
#
# Steps implemented so far:
#   1. Python + AI/ML libraries  ✔
#   2. VS Code install            ✔
#   3. Twinny (Ollama Copilot) extension  ✔
#   4. Ollama server install & first launch  ✔
#   5. Lifecycle wiring (Ollama runs with VS Code)  ✔
#   6. Pull best SLM for coding/reasoning  ✔

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

normalize_pkg_key() {
    echo "$1" | sed 's/\[.*\]//' | sed 's/[>=<!].*//' | tr '[:upper:]' '[:lower:]' | xargs
}

INSTALLED_PIP_KEYS=$(python -m pip list --format=columns 2>/dev/null | tail -n +3 | awk '{print tolower($1)}')

pip_has_key() {
    local key="$1"
    echo "$INSTALLED_PIP_KEYS" | grep -qx "$key"
}

mark_installed_key() {
    local key="$1"
    if ! pip_has_key "$key"; then
        INSTALLED_PIP_KEYS="${INSTALLED_PIP_KEYS}"$'\n'"$key"
    fi
}

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
    for pkg in "${packages[@]}"; do
        # Normalise: strip extras/version specifiers for lookup
        local key
        key=$(normalize_pkg_key "$pkg")
        if pip_has_key "$key"; then
            ok "$pkg already installed"
        else
            warn "$pkg missing — installing ..."
            if [ ${#extra_args[@]} -gt 0 ]; then
                python -m pip install "$pkg" "${extra_args[@]}" --quiet
            else
                python -m pip install "$pkg" --quiet
            fi
            mark_installed_key "$key"
            ok "$pkg installed"
        fi
    done
}

CORE_SCIENTIFIC=(numpy pandas scipy matplotlib seaborn)
MACHINE_LEARNING=(scikit-learn xgboost lightgbm)
DEEP_LEARNING_TF=(tensorflow tensorboard keras)
PYTORCH_CPU=(torch torchvision torchaudio)
NOTEBOOK_TOOLING=(notebook ipykernel ipywidgets jupyterlab)
GENERATIVE_AI=(
    transformers diffusers accelerate datasets tokenizers
    huggingface-hub openai langchain langchain-community
    sentence-transformers faiss-cpu chromadb
)
UTILITIES=(python-dotenv tqdm pillow requests httpx pydantic)
DOCS_SITE=(mkdocs-material pymdown-extensions mkdocs-jupyter)
NOTEBOOK_EXTRAS=(
    mlflow tiktoken mcp fastapi "uvicorn[standard]" anyio redis
    langgraph langchain-core langchain-openai
    autogen-agentchat semantic-kernel ollama
)

ALL_REQUIRED_PACKAGES=(
    "${CORE_SCIENTIFIC[@]}"
    "${MACHINE_LEARNING[@]}"
    "${DEEP_LEARNING_TF[@]}"
    "${PYTORCH_CPU[@]}"
    "${NOTEBOOK_TOOLING[@]}"
    "${GENERATIVE_AI[@]}"
    "${UTILITIES[@]}"
    "${DOCS_SITE[@]}"
    "${NOTEBOOK_EXTRAS[@]}"
)

ALL_DEPENDENCIES_MET=true
for pkg in "${ALL_REQUIRED_PACKAGES[@]}"; do
    key=$(normalize_pkg_key "$pkg")
    if ! pip_has_key "$key"; then
        ALL_DEPENDENCIES_MET=false
        break
    fi
done

if [ "$ALL_DEPENDENCIES_MET" = true ]; then
    ok "All Python package dependencies already satisfied — skipping package installation step"
else
    # Core scientific stack
    install_group "Core scientific stack" "${CORE_SCIENTIFIC[@]}"

    # Machine learning
    install_group "Machine learning" "${MACHINE_LEARNING[@]}"

    # Deep learning — TensorFlow
    install_group "Deep learning / TensorFlow" "${DEEP_LEARNING_TF[@]}"

    # PyTorch — CPU-safe build (no CUDA required on stock machines)
    install_group "PyTorch (CPU build)" "${PYTORCH_CPU[@]}" \
        --extra-args --index-url https://download.pytorch.org/whl/cpu

    # Notebook tooling
    install_group "Notebook tooling" "${NOTEBOOK_TOOLING[@]}"

    # Generative AI / LLM utilities
    install_group "Generative AI / LLM utilities" "${GENERATIVE_AI[@]}"

    # General utilities
    install_group "Utilities" "${UTILITIES[@]}"

    # Docs / study site (MkDocs Material — browse notes/ in a web browser)
    # mkdocs-jupyter renders every notebook.ipynb as a page alongside the .md files.
    install_group "Docs site (MkDocs Material)" "${DOCS_SITE[@]}"

    # Notebook extras — dependencies pulled in by per-notes setup scripts
    install_group "Notebook extras (AIInfrastructure + MultiAgentAI)" "${NOTEBOOK_EXTRAS[@]}"
fi

# ─── 1e. Register Jupyter kernels ─────────────────────────────────────────────

step "Registering Jupyter kernels"

kernel_exists() {
    local kernel_name="$1"
    python3 - "$kernel_name" <<'PYEOF'
import json, subprocess, sys
name = sys.argv[1]
try:
    out = subprocess.check_output([sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"], text=True)
    data = json.loads(out)
    sys.exit(0 if name in (data.get("kernelspecs") or {}) else 1)
except Exception:
    sys.exit(2)
PYEOF
    return $?
}

ALL_KERNELS=(ai-ml-dev ml-notes ai-infrastructure multi-agent-ai)
ALL_KERNELS_PRESENT=true
for k in "${ALL_KERNELS[@]}"; do
    if ! kernel_exists "$k"; then
        ALL_KERNELS_PRESENT=false
        break
    fi
done

if [ "$ALL_KERNELS_PRESENT" = true ]; then
    ok "All required Jupyter kernels already registered — skipping kernel registration"
else
    if kernel_exists "ai-ml-dev"; then
        ok "Kernel 'ai-ml-dev' already registered"
    else
        python -m ipykernel install --user --name "ai-ml-dev" --display-name "AI/ML Dev (venv)" &>/dev/null
        ok "Kernel 'ai-ml-dev' registered"
    fi

    if kernel_exists "ml-notes"; then
        ok "Kernel 'ml-notes' already registered"
    else
        python -m ipykernel install --user --name "ml-notes" --display-name "ML Notes (venv)" &>/dev/null
        ok "Kernel 'ml-notes' registered"
    fi

    if kernel_exists "ai-infrastructure"; then
        ok "Kernel 'ai-infrastructure' already registered"
    else
        python -m ipykernel install --user --name "ai-infrastructure" --display-name "Python (AI Infrastructure)" &>/dev/null
        ok "Kernel 'ai-infrastructure' registered"
    fi

    if kernel_exists "multi-agent-ai"; then
        ok "Kernel 'multi-agent-ai' already registered"
    else
        python -m ipykernel install --user --name "multi-agent-ai" --display-name "Multi-Agent AI" &>/dev/null
        ok "Kernel 'multi-agent-ai' registered"
    fi
fi

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

# ─── STEP 3: Twinny (Ollama AI Copilot) Extension ───────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 3/6"
echo "  Twinny — Ollama AI Copilot Extension"
echo "══════════════════════════════════════════════"

TWINNY_EXT_ID="rjmacarthy.twinny"
CONTINUE_EXT_ID="Continue.continue"

step "Checking Twinny extension ($TWINNY_EXT_ID)"

EXTENSION_INSTALLED=false
if command -v "${CODE_CMD}" &>/dev/null; then
    if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$TWINNY_EXT_ID"; then
        ok "Twinny already installed"
        EXTENSION_INSTALLED=true
    fi
else
    warn "'${CODE_CMD}' not on PATH — skipping extension check"
fi

if [ "$EXTENSION_INSTALLED" = false ]; then
    warn "Twinny not found — installing ..."
    if command -v "${CODE_CMD}" &>/dev/null; then
        "${CODE_CMD}" --install-extension "$TWINNY_EXT_ID" --force &>/dev/null || true
        # Verify
        if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$TWINNY_EXT_ID"; then
            ok "Twinny installed successfully"
        else
            warn "Install ran but extension not detected yet — it may appear after VS Code restarts"
        fi
    else
        warn "Cannot install Twinny: 'code' not on PATH."
        warn "Install manually: open VS Code → Extensions → search 'Twinny' → Install"
    fi
fi

step "Twinny post-install configuration note"
echo ""
echo "  After launching VS Code:"
echo "    1. Open the Twinny sidebar (robot icon on the Activity Bar)"
echo "    2. Setup auto-writes provider/model settings in Step 6"
echo "    3. If needed, verify provider: Ollama, host: localhost, port: 11434"
echo "    4. Confirm chat/FIM model matches the model printed at the end"
echo ""

step "Checking optional Continue extension ($CONTINUE_EXT_ID)"

CONTINUE_INSTALLED=false
if command -v "${CODE_CMD}" &>/dev/null; then
    if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$CONTINUE_EXT_ID"; then
        ok "Continue already installed"
        CONTINUE_INSTALLED=true
    fi
else
    warn "'${CODE_CMD}' not on PATH — skipping Continue extension check"
fi

if [ "$CONTINUE_INSTALLED" = false ]; then
    warn "Continue not found — installing ..."
    if command -v "${CODE_CMD}" &>/dev/null; then
        "${CODE_CMD}" --install-extension "$CONTINUE_EXT_ID" --force &>/dev/null || true
        if "${CODE_CMD}" --list-extensions 2>/dev/null | grep -qi "$CONTINUE_EXT_ID"; then
            ok "Continue installed successfully"
        else
            warn "Install ran but extension not detected yet — it may appear after VS Code restarts"
        fi
    else
        warn "Cannot install Continue: 'code' not on PATH."
        warn "Install manually: open VS Code → Extensions → search 'Continue' → Install"
    fi
fi

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

    # Start server detached, redirect output to a log file
    nohup ollama serve > "$REPO_ROOT/.ollama.log" 2>&1 &
    OLLAMA_BG_PID=$!
    echo "$OLLAMA_BG_PID" > "$PID_FILE"
    ok "Ollama server started (PID $OLLAMA_BG_PID, log: .ollama.log)"
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

if [ -f "$WATCHER_SCRIPT" ]; then
    ok "ollama-watcher.sh already exists — skipping"
else
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
    nohup ollama serve >> "$REPO_ROOT/.ollama.log" 2>&1 &
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
fi

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

# ─── STEP 6: Pull Best SLM for AI/ML Coding ──────────────────────────────────
#
# Primary:  qwen2.5-coder:3b   (~2 GB class, CPU-friendly)
# Fallback: qwen2.5-coder:1.5b (~1 GB class, very lightweight)
# Selection is automatic based on detected system RAM.

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Setup — Step 6/6"
echo "  Pull Best Local SLM"
echo "══════════════════════════════════════════════"

PRIMARY_MODEL="qwen2.5-coder:3b"
FALLBACK_MODEL="qwen2.5-coder:1.5b"

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

if [ "$TOTAL_RAM_GB" -ge 8 ]; then
    CHOSEN_MODEL="$PRIMARY_MODEL"
    ok "RAM ≥ 8 GB — selecting primary model: $CHOSEN_MODEL"
else
    CHOSEN_MODEL="$FALLBACK_MODEL"
    warn "RAM < 8 GB — selecting fallback model: $CHOSEN_MODEL"
fi

# ── 6b. Check if model already pulled ───────────────────────────────────────────

step "Checking if $CHOSEN_MODEL is already available"

MODEL_PRESENT=false
if ollama list 2>/dev/null | grep -qi "$(echo "$CHOSEN_MODEL" | cut -d: -f1)"; then
    ok "$CHOSEN_MODEL already present in Ollama"
    MODEL_PRESENT=true
fi

# ── 6c. Pull the model ────────────────────────────────────────────────────────

if [ "$MODEL_PRESENT" = false ]; then
    step "Pulling $CHOSEN_MODEL (this may take a few minutes on first run)"
    echo "  Downloading model — progress shown below:"
    echo ""
    if ollama pull "$CHOSEN_MODEL"; then
        ok "$CHOSEN_MODEL pulled successfully"
    else
        warn "Pull failed — check your internet connection and retry: ollama pull $CHOSEN_MODEL"
    fi
else
    step "Skipping pull — $CHOSEN_MODEL already present"
fi

# ── 6d. Smoke test local inference via Ollama API ─────────────────────────────

step "Running local inference smoke test against $CHOSEN_MODEL"

SMOKE_PAYLOAD=$(cat <<SMOKEEOF
{
  "model": "$CHOSEN_MODEL",
  "prompt": "Reply with exactly: OLLAMA_SMOKE_OK",
  "stream": false
}
SMOKEEOF
)

SMOKE_RESPONSE=$(curl -sS --max-time 90 \
    -H "Content-Type: application/json" \
    -d "$SMOKE_PAYLOAD" \
    "$OLLAMA_BASE_URL/api/generate" || true)

if [ -n "$SMOKE_RESPONSE" ] && echo "$SMOKE_RESPONSE" | grep -q '"response"'; then
    SMOKE_TEXT=$(python3 -c 'import json,sys
raw = sys.stdin.read()
try:
    obj = json.loads(raw)
    print((obj.get("response") or "").strip())
except Exception:
    print("")' <<< "$SMOKE_RESPONSE")
    if [ -n "$SMOKE_TEXT" ]; then
        ok "Inference smoke test passed"
        echo "    Model response: $SMOKE_TEXT"
    else
        warn "Smoke test returned an empty response"
    fi
else
    warn "Smoke test failed — run manually: ollama run $CHOSEN_MODEL"
fi

# ── 6e. Configure Twinny to use the model ────────────────────────────────────────

step "Writing Twinny model settings to VS Code user settings"

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

# Build the settings block to inject
TWINNY_BLOCK=$(cat << TWINNYEOF
{
  "twinny.ollamaApiHostname": "localhost",
  "twinny.ollamaApiPort": 11434,
  "twinny.chatModelName": "${CHOSEN_MODEL}",
  "twinny.fimModelName": "${CHOSEN_MODEL}",
  "twinny.apiProvider": "ollama"
}
TWINNYEOF
)

if [ -f "$VSCODE_SETTINGS_FILE" ]; then
    # Merge: use Python (available from Step 1) to safely merge JSON
    python3 - << PYEOF
import json, sys
with open('$VSCODE_SETTINGS_FILE', 'r') as f:
    try:
        existing = json.load(f)
    except Exception:
        existing = {}
patch = {
    'twinny.ollamaApiHostname': 'localhost',
    'twinny.ollamaApiPort': 11434,
    'twinny.chatModelName': '${CHOSEN_MODEL}',
    'twinny.fimModelName': '${CHOSEN_MODEL}',
    'twinny.apiProvider': 'ollama',
}
existing.update(patch)
with open('$VSCODE_SETTINGS_FILE', 'w') as f:
    json.dump(existing, f, indent=4)
print('  ✓ Twinny settings merged into existing settings.json')
PYEOF
else
    echo "$TWINNY_BLOCK" > "$VSCODE_SETTINGS_FILE"
    ok "settings.json created with Twinny model settings"
fi

# ── 6f. Configure Continue profile for Ollama ─────────────────────────────────

step "Writing Continue Ollama profile"

CONTINUE_DIR="$HOME/.continue"
CONTINUE_CONFIG="$CONTINUE_DIR/config.yaml"
mkdir -p "$CONTINUE_DIR"

if [ -f "$CONTINUE_CONFIG" ]; then
        warn "Continue config already exists at $CONTINUE_CONFIG — leaving existing file unchanged"
        warn "If you want this profile, merge model settings manually into your existing Continue config"
else
        cat > "$CONTINUE_CONFIG" << CONTINUEYAML
name: Local Ollama
version: 1.0.0

models:
    - title: Local Chat (${CHOSEN_MODEL})
        provider: ollama
        model: ${CHOSEN_MODEL}
        apiBase: http://localhost:11434

tabAutocompleteModel:
    title: Local FIM (${CHOSEN_MODEL})
    provider: ollama
    model: ${CHOSEN_MODEL}
    apiBase: http://localhost:11434

context:
    - provider: code
    - provider: docs
    - provider: diff
CONTINUEYAML
        ok "Continue profile written: $CONTINUE_CONFIG"
fi

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
echo "  Twinny ext  : $TWINNY_EXT_ID"
echo "  Continue ext: $CONTINUE_EXT_ID"
echo "  Ollama      : $OLLAMA_BASE_URL"
echo "  SLM model   : $CHOSEN_MODEL"
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
