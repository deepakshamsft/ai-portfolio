# scripts/ — AI/ML Dev Environment Setup

This directory contains cross-platform setup scripts that provision a complete local AI/ML development environment in a single run.

---

## What the scripts do

| Step | What happens |
|------|-------------|
| **1 — Python + libraries** | Detects Python 3.11+ (installs if missing), creates a `.venv` at the repo root, and installs the full AI/ML package stack |
| **2 — VS Code** | Installs Visual Studio Code if `code` is not already on PATH |
| **3 — Twinny extension** | Installs `rjmacarthy.twinny` — the Ollama AI Copilot for VS Code |
| **4 — Ollama server** | Installs the Ollama local inference server and starts it for the first time |
| **5 — Lifecycle wiring** | Writes `.vscode/tasks.json` and `ollama-watcher` so Ollama starts when VS Code opens and stops when it closes |
| **6 — Pull SLM** | Auto-detects RAM and pulls the best coding model (`qwen2.5-coder:7b` ≥ 10 GB RAM, `phi3.5` otherwise), then writes Twinny's settings |

---

## Usage

**Windows (PowerShell):**
```powershell
.\scripts\setup.ps1
```

**macOS / Linux (bash):**
```bash
bash scripts/setup.sh
```

Run from the repo root. No arguments required. The scripts are fully idempotent — safe to re-run; already-installed components are detected and skipped.

---

## Files

```
scripts/
├── setup.ps1              ← Main setup script — Windows / PowerShell
├── setup.sh               ← Main setup script — macOS / Linux
├── ollama-watcher.ps1     ← Written by setup.ps1 — monitors VS Code, stops Ollama on exit
└── ollama-watcher.sh      ← Written by setup.sh  — monitors VS Code, stops Ollama on exit
```

> `ollama-watcher.*` are generated during Step 5 of the setup run. Do not edit them manually — re-run the setup script to regenerate.

---

## Minimum machine requirements

| Resource | Recommended | Minimum |
|----------|-------------|---------|
| RAM | 16 GB | 8 GB |
| Free disk space | 20 GB | 10 GB |
| CPU | 4-core, 2020 or newer | 2-core |
| GPU | Optional — CUDA/Metal accelerates inference | Not required |
| OS | Windows 11, macOS 13+, Ubuntu 22.04+ | Windows 10, macOS 12, Ubuntu 20.04 |

> **Note on GPU:** The scripts install PyTorch with the **CPU build** by default so the setup works on any stock machine. If you have an NVIDIA GPU and want CUDA acceleration, replace the torch line in `.venv` after setup: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

---

## Python library stack installed

| Group | Packages |
|-------|---------|
| Core scientific | numpy, pandas, scipy, matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost, lightgbm |
| Deep learning | tensorflow, tensorboard, keras |
| PyTorch (CPU) | torch, torchvision, torchaudio |
| Notebooks | notebook, ipykernel, ipywidgets, jupyterlab |
| Generative AI / LLM | transformers, diffusers, accelerate, datasets, tokenizers, huggingface-hub, openai, langchain, langchain-community, sentence-transformers, faiss-cpu, chromadb |
| Utilities | python-dotenv, tqdm, pillow, requests, httpx, pydantic |

---

## Ollama + Twinny — one-time manual step

After running the setup script, open VS Code and:

1. Click **Terminal → Run Task** and choose **"Allow Automatic Tasks"** — this enables the `folderOpen` watcher that keeps Ollama running whenever the workspace is open.
2. Open the **Twinny sidebar** (robot icon on the Activity Bar).
3. Click the settings cog → set provider to **Ollama**, hostname `localhost`, port `11434`.
4. The chat and FIM model fields should already be populated (written by Step 6). If not, set them to the model printed at the end of the setup run.

---

## Lifecycle behaviour

| Event | What happens |
|-------|-------------|
| Open workspace in VS Code | `ollama-watcher` starts automatically, launches `ollama serve` if not running |
| VS Code is open | Ollama server keeps running silently at `http://localhost:11434` |
| Close all VS Code windows | Watcher detects no `code` process, kills Ollama and removes `.ollama.pid` |
| Re-open workspace | Watcher starts Ollama again |

To manually stop Ollama without closing VS Code: **Terminal → Run Task → `ollama-stop`**.

---

## Troubleshooting

**`code` not found after install**
Restart your terminal. On Windows, `winget` may install VS Code without immediately updating the current session's PATH.

**Ollama server not responding**
```bash
# Check if it's running
curl http://localhost:11434

# Start manually
ollama serve

# Check the log written by the watcher
cat .ollama.log    # macOS / Linux
Get-Content .ollama.log    # Windows
```

**pip install fails on a specific package**
Activate the venv and install manually:
```bash
source .venv/bin/activate          # macOS / Linux
.\.venv\Scripts\Activate.ps1       # Windows
pip install <package-name>
```

**Model pull is slow or fails**
The `qwen2.5-coder:7b` model is ~4.7 GB. A slow connection may time out. Resume with:
```bash
ollama pull qwen2.5-coder:7b
```
Ollama resumes partial downloads automatically.

**Want to switch models later**
```bash
ollama pull <model-name>
# Then update VS Code settings:
# twinny.chatModelName and twinny.fimModelName
```
