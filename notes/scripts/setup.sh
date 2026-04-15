#!/usr/bin/env bash
# setup.sh — ML Notes Environment Setup (macOS / Linux)
# Verifies dependencies, installs missing ones, and launches Jupyter at notes/ML
# Run from anywhere:
#   bash notes/scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# notes/scripts/ → notes/ → repo root
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
NOTES_PATH="$REPO_ROOT/notes/ML"
VENV_PATH="$REPO_ROOT/.venv"

REQUIRED_PACKAGES=(
    "notebook"
    "numpy"
    "pandas"
    "scikit-learn"
    "matplotlib"
    "seaborn"
    "tensorflow"
    "tensorboard"
    "scipy"
    "ipykernel"
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

step()  { echo; echo "▶ $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ! $*"; }
fail()  { echo "  ✗ $*" >&2; exit 1; }

# ─── 1. Python ────────────────────────────────────────────────────────────────

step "Checking Python"

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        major=$(echo "$ver" | sed 's/Python \([0-9]*\)\..*/\1/')
        minor=$(echo "$ver" | sed 's/Python [0-9]*\.\([0-9]*\)\..*/\1/')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$candidate"
            ok "$ver"
            break
        else
            warn "$ver found but Python 3.9+ is required"
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3.9+ not found. Install from https://www.python.org/downloads/ or via your package manager and re-run."
fi

# ─── 2. pip ───────────────────────────────────────────────────────────────────

step "Checking pip"
if "$PYTHON" -m pip --version &>/dev/null; then
    ok "$("$PYTHON" -m pip --version)"
else
    fail "pip not available. Run: $PYTHON -m ensurepip --upgrade"
fi

# ─── 3. Virtual environment ───────────────────────────────────────────────────

step "Virtual environment"

if [ -d "$VENV_PATH" ]; then
    ok "Existing venv found at .venv"
else
    warn "No venv found — creating one at .venv ..."
    "$PYTHON" -m venv "$VENV_PATH"
    ok "Created .venv"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
ok "Activated .venv"

# Upgrade pip quietly
python -m pip install --upgrade pip --quiet

# ─── 4. Packages ──────────────────────────────────────────────────────────────

step "Checking / installing packages"

INSTALLED=$(python -m pip list --format=columns 2>/dev/null | tail -n +3 | awk '{print tolower($1)}')

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    pkg_lower=$(echo "$pkg" | tr '[:upper:]' '[:lower:]')
    if echo "$INSTALLED" | grep -qx "$pkg_lower"; then
        ok "$pkg already installed"
    else
        warn "$pkg missing — installing ..."
        python -m pip install "$pkg" --quiet
        ok "$pkg installed"
    fi
done

# ─── 5. Register kernel ───────────────────────────────────────────────────────

step "Registering Jupyter kernel"
python -m ipykernel install --user --name "ml-notes" --display-name "ML Notes (venv)" &>/dev/null
ok "Kernel 'ml-notes' registered"

# ─── 6. Verify notes/ML exists ────────────────────────────────────────────────

step "Verifying notes/ML path"
if [ ! -d "$NOTES_PATH" ]; then
    warn "notes/ML does not exist yet — creating directory ..."
    mkdir -p "$NOTES_PATH"
fi
ok "notes/ML is ready at $NOTES_PATH"

# ─── 7. Launch Jupyter ────────────────────────────────────────────────────────

step "Launching Jupyter Notebook"
echo
echo "  Opening browser at http://localhost:8888"
echo "  Root directory : $NOTES_PATH"
echo "  Press Ctrl+C to stop the server."
echo

jupyter notebook --notebook-dir="$NOTES_PATH"
