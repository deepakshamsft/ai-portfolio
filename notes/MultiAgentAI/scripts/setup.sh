#!/usr/bin/env bash
# setup.sh — Multi-Agent AI Notes: Dependency Setup (macOS / Linux)
# Installs every package needed to run the Jupyter notebooks across all 7
# chapters under notes/MultiAgentAI/.
#
# Run from the repo root or from this scripts/ folder:
#   bash notes/MultiAgentAI/scripts/setup.sh
#
# What it does:
#   1. Verifies Python 3.9+ (3.11+ recommended)
#   2. Creates or reuses the repo-level .venv
#   3. Installs the MultiAgentAI notebook dependency stack
#   4. Registers a Jupyter kernel named "multi-agent-ai"
#   5. Generates all 7 chapter notebooks from the generation script
#   6. Verifies all core imports work

set -euo pipefail

# ─── Locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"

# ─── Colour helpers ────────────────────────────────────────────────────────────
_cyan()  { printf "\n\033[36m▶ %s\033[0m\n" "$*"; }
_green() { printf "  \033[32m✓ %s\033[0m\n" "$*"; }
_warn()  { printf "  \033[33m! %s\033[0m\n" "$*"; }
_fail()  { printf "  \033[31m✗ %s\033[0m\n" "$*"; exit 1; }
_group() { printf "\n  \033[36m── %s\033[0m\n" "$*"; }

printf "\n"
printf "══════════════════════════════════════════════════════\n"
printf "  Multi-Agent AI Notes — Notebook Dependency Setup   \n"
printf "  macOS / Linux                                       \n"
printf "══════════════════════════════════════════════════════\n"

# ─── STEP 1: Python ────────────────────────────────────────────────────────────

_cyan "Checking Python 3.9+"

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" --version 2>&1 | grep -oE "[0-9]+\.[0-9]+")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON="$candidate"
            _green "$candidate $VER"
            break
        elif [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            _warn "$candidate $VER found — Python 3.11+ recommended; continuing"
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    _fail "Python 3.9+ not found. Install from https://www.python.org/downloads/ and re-run."
fi

# ─── STEP 2: Virtual environment ───────────────────────────────────────────────

_cyan "Virtual environment (.venv at repo root)"

if [ -d "$VENV_PATH" ]; then
    _green "Existing .venv found at $VENV_PATH — reusing"
else
    _warn "No .venv found — creating at $VENV_PATH ..."
    "$PYTHON" -m venv "$VENV_PATH"
    _green "Created .venv"
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
_green "Activated .venv"

pip install --upgrade pip setuptools wheel --quiet
_green "pip / setuptools / wheel up to date"

# ─── STEP 3: Package groups ────────────────────────────────────────────────────

_cyan "Installing Multi-Agent AI packages"

_group "Core notebook stack"
pip install jupyter ipykernel --quiet
_green "jupyter, ipykernel"

_group "Ch.1 — Message Formats (token counting)"
pip install tiktoken --quiet
_green "tiktoken"

_group "Ch.2 — MCP (Model Context Protocol SDK)"
pip install mcp --quiet
_green "mcp"

_group "Ch.3 — A2A protocol (HTTP / SSE)"
pip install fastapi "uvicorn[standard]" httpx anyio --quiet
_green "fastapi, uvicorn, httpx, anyio"

_group "Ch.4 — Event-driven agents (Redis Streams)"
pip install redis --quiet
_green "redis"

_group "Ch.5 — Shared Memory (already covered by redis)"
_green "redis (shared memory blackboard)"

_group "Ch.6 — Trust & Sandboxing (validation)"
pip install pydantic --quiet
_green "pydantic"

_group "Ch.7 — Agent Frameworks (LangGraph, AutoGen, Semantic Kernel)"
pip install langgraph langchain-core langchain-openai --quiet
_green "langgraph, langchain-core, langchain-openai"

pip install autogen-agentchat --quiet
_green "autogen-agentchat"

pip install semantic-kernel --quiet
_green "semantic-kernel"

_group "Local model inference (Ollama Python client)"
pip install ollama --quiet
_green "ollama"

# ─── STEP 4: Jupyter kernel registration ───────────────────────────────────────

_cyan "Registering 'multi-agent-ai' Jupyter kernel"

python -m ipykernel install --user --name multi-agent-ai --display-name "Multi-Agent AI"
_green "Kernel 'multi-agent-ai' registered"

# ─── STEP 5: Generate notebooks ────────────────────────────────────────────────

_cyan "Generating chapter notebooks"

GENERATOR="$SCRIPT_DIR/generate_notebooks.py"
if [ -f "$GENERATOR" ]; then
    python "$GENERATOR"
    _green "All 7 chapter notebooks generated"
else
    _warn "Generator script not found at $GENERATOR — skipping notebook generation"
fi

# ─── STEP 6: Smoke test imports ────────────────────────────────────────────────

_cyan "Verifying imports"

python - <<'EOF'
import sys, importlib

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

missing = []
print()
for pkg, label in packages:
    try:
        importlib.import_module(pkg)
        print(f"  \033[32m✓\033[0m {pkg:<25} {label}")
    except ImportError:
        print(f"  \033[33m!\033[0m {pkg:<25} {label}  ← missing")
        missing.append(pkg)

if missing:
    print(f"\n  Some packages missing: {missing}")
    print("  Re-run this script or: pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("\n  All imports verified.")
EOF

# ─── Done ──────────════════════════════════════════════════════════════════────

printf "\n"
printf "══════════════════════════════════════════════════════\n"
printf "  \033[32mSetup complete.\033[0m\n"
printf "══════════════════════════════════════════════════════\n"
printf "\n"
printf "  Open any chapter notebook in VS Code and select the 'Multi-Agent AI' kernel.\n"
printf "\n"
printf "  Chapter notebooks:\n"
printf "    Ch.1  Message Formats         MessageFormats/notebook.ipynb\n"
printf "    Ch.2  MCP                     MCP/notebook.ipynb\n"
printf "    Ch.3  A2A Protocol             A2A/notebook.ipynb\n"
printf "    Ch.4  Event-Driven Agents      EventDrivenAgents/notebook.ipynb\n"
printf "    Ch.5  Shared Memory            SharedMemory/notebook.ipynb\n"
printf "    Ch.6  Trust & Sandboxing       TrustAndSandboxing/notebook.ipynb\n"
printf "    Ch.7  Agent Frameworks         AgentFrameworks/notebook.ipynb\n"
printf "\n"
printf "  Optional: run Ollama for live model cells in Ch.7:\n"
printf "    # macOS: brew install ollama\n"
printf "    # Linux: curl -fsSL https://ollama.ai/install.sh | sh\n"
printf "    ollama pull phi3:mini\n"
printf "\n"
