#!/usr/bin/env bash
# setup.sh — AI Infrastructure Notes: Dependency Setup (macOS / Linux)
# Installs every package needed to run the Jupyter notebooks across all 10
# chapters under notes/AIInfrastructure/.
#
# Run from the repo root or from this scripts/ folder:
#   bash notes/AIInfrastructure/scripts/setup.sh
#
# What it does:
#   1. Verifies Python 3.9+ (3.11+ recommended)
#   2. Creates or reuses the repo-level .venv
#   3. Installs the AIInfrastructure notebook dependency stack
#   4. Registers a Jupyter kernel named "ai-infrastructure"
#   5. Verifies all core imports work

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# notes/AIInfrastructure/scripts/ → repo root (three levels up)
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
VENV_PATH="$REPO_ROOT/.venv"

# ─── Helpers ───────────────────────────────────────────────────────────────────

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
        ubuntu|debian|linuxmint|pop)   OS="debian" ;;
        fedora|rhel|centos|rocky|alma) OS="fedora" ;;
        arch|manjaro)                  OS="arch"   ;;
        *)                             OS="linux"  ;;
    esac
fi

# ─── Header ────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════"
echo "  AI Infrastructure Notes — Notebook Dependency Setup"
echo "  macOS / Linux                                       "
echo "══════════════════════════════════════════════════════"

# ─── STEP 1: Python ────────────────────────────────────────────────────────────

step "Checking Python 3.9+"

PYTHON=""
for candidate in python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        major=$(echo "$ver" | sed 's/Python \([0-9]*\)\..*/\1/')
        minor=$(echo "$ver" | sed 's/Python [0-9]*\.\([0-9]*\)\..*/\1/')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            ok "$ver (meets 3.11+ recommendation)"
            break
        elif [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            warn "$ver found — Python 3.11+ recommended; continuing with $ver"
            PYTHON="$candidate"
            break
        else
            warn "$ver found — Python 3.9+ required"
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
                fail "Homebrew not found. Install from https://brew.sh/ then re-run, or install Python manually."
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
            fail "Cannot auto-install Python. Install Python 3.11+ from https://www.python.org/downloads/ and re-run."
            ;;
    esac
    ok "Python installed: $("$PYTHON" --version 2>&1)"
fi

# ─── STEP 2: pip ───────────────────────────────────────────────────────────────

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

# ─── STEP 3: Virtual environment ───────────────────────────────────────────────

step "Virtual environment (.venv at repo root)"

if [ -d "$VENV_PATH" ]; then
    ok "Existing .venv found at $VENV_PATH — reusing"
else
    warn "No .venv found — creating at $VENV_PATH ..."
    "$PYTHON" -m venv "$VENV_PATH"
    ok "Created .venv"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
ok "Activated .venv"

python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip / setuptools / wheel up to date"

# ─── STEP 4: Package installation ──────────────────────────────────────────────

step "Installing AI Infrastructure notebook dependencies"

install_group() {
    local group_name="$1"
    shift
    group "$group_name"

    for pkg in "$@"; do
        # Normalise package name for pip list lookup
        key=$(echo "$pkg" | sed 's/\[.*\]//' | sed 's/[>=<!].*//' | tr '[:upper:]' '[:lower:]' | xargs)
        if python -m pip show "$key" &>/dev/null 2>&1; then
            ok "$pkg already installed"
        else
            warn "$pkg missing — installing ..."
            python -m pip install "$pkg" --quiet
            ok "$pkg installed"
        fi
    done
}

# ── Core scientific stack (Chapters 1–10, all notebooks) ──
# numpy      : all arithmetic intensity calculations, roofline models, simulators
# pandas     : GPU spec databases, result tables, cost models
# matplotlib : all plots — roofline, batching curves, cost comparisons
# scipy      : used in Ch.3 (quantization error statistics) and Ch.12 (clustering metrics)
install_group "Core scientific stack (all chapters)" \
    "numpy" \
    "pandas" \
    "scipy" \
    "matplotlib"

# ── Jupyter tooling ──
# ipykernel      : required to run .ipynb notebooks
# ipywidgets     : interactive sliders for sensitivity analyses
# jupyter        : notebook server (skip if using VS Code's built-in Jupyter extension)
install_group "Jupyter tooling" \
    "ipykernel" \
    "ipywidgets" \
    "jupyter"

# ── MLOps chapter (Ch.9) ──
# mlflow         : local experiment tracking, model registry, artifact logging
install_group "MLOps — experiment tracking (Ch.9)" \
    "mlflow"

# ── Optional: heavier libraries ──
# These are NOT required for any notebook but unlock the optional cells in Ch.3
# (quantization) and Ch.9 (model registry) that call real model APIs.
# Uncomment to install:
#
# install_group "Optional: real model inference (Ch.3 optional cells)" \
#     "torch" \          # PyTorch CPU — optional GPTQ live demo in Ch.3
#     "transformers" \   # HuggingFace — optional perplexity measurement in Ch.3
#     "bitsandbytes"     # INT8/INT4 — optional quantization in Ch.3

# ─── STEP 5: Jupyter kernel ────────────────────────────────────────────────────

step "Registering Jupyter kernel 'ai-infrastructure'"

if python -m ipykernel install --user \
        --name "ai-infrastructure" \
        --display-name "Python (AI Infrastructure)" 2>/dev/null; then
    ok "Kernel registered: 'ai-infrastructure'"
else
    warn "Could not register Jupyter kernel — you can still open notebooks in VS Code directly"
fi

# ─── STEP 6: Smoke test ────────────────────────────────────────────────────────

step "Smoke-testing core imports"

python - <<'PYEOF'
import sys
libs = {
    "numpy":      "import numpy as np; np.zeros(3)",
    "pandas":     "import pandas as pd; pd.DataFrame()",
    "matplotlib": "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
    "scipy":      "import scipy",
    "mlflow":     "import mlflow",
    "ipykernel":  "import ipykernel",
}
errors = []
for name, code in libs.items():
    try:
        exec(code)
        print(f"  OK   {name}")
    except Exception as e:
        errors.append(name)
        print(f"  FAIL {name}: {e}")
if errors:
    print(f"\nFailed imports: {errors}")
    sys.exit(1)
else:
    print("\nAll imports OK.")
PYEOF

ok "All core imports verified"

# ─── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  To start a notebook:"
echo "    • Open the .ipynb file in VS Code and select the"
echo "      'Python (AI Infrastructure)' kernel, OR"
echo "    • Run: jupyter notebook"
echo ""
echo "  Chapters ready to run:"
echo "    Ch.1  GPU Architecture          GPUArchitecture/notebook.ipynb"
echo "    Ch.2  Memory Budgets             MemoryAndComputeBudgets/notebook.ipynb"
echo "    Ch.3  Quantization               QuantizationAndPrecision/notebook.ipynb"
echo "    Ch.4  Distributed Training       ParallelismAndDistributedTraining/notebook.ipynb"
echo "    Ch.5  Inference Optimization     InferenceOptimization/notebook.ipynb"
echo "    Ch.6  Serving Frameworks         ServingFrameworks/notebook.ipynb"
echo "    Ch.7  Networking & Clusters      NetworkingAndClusterArchitecture/notebook.ipynb"
echo "    Ch.8  Cloud AI Infrastructure    CloudAIInfrastructure/notebook.ipynb"
echo "    Ch.9  MLOps                      MLOpsAndExperimentManagement/notebook.ipynb"
echo "    Ch.10 Production AI Platform     ProductionAIPlatform/notebook.ipynb"
echo "══════════════════════════════════════════════════════"
echo ""
