# ai-portfolio

A personal portfolio combining structured AI/ML study notes and hands-on engineering projects.

---

## Structure

```
ai-portfolio/
├── notes/          ← Study material: all ML and AI learning content
├── projects/       ← Hands-on projects and experiments
└── pdf-gen/        ← PDF generation tooling for the reference books
```

---

## `notes/` — Study Material

All learning content lives here: 17-chapter ML curriculum, agentic AI notes, reference books, scripts, and runnable Jupyter notebooks.

→ See [notes/README.md](notes/README.md) for the full index.

---

## `projects/` — Engineering Projects

Standalone, runnable projects built with the concepts covered in the notes.

| Path | What it is |
|---|---|
| `projects/ml/linear-regression/` | End-to-end linear regression project on real data |
| `projects/ai/rag-pipeline/` | RAG pipeline implementation |

More projects added as the curriculum progresses.

---

## Quick start

**Set up the full dev environment (Windows):**
```powershell
.\scripts\setup.ps1
```

**Set up the full dev environment (macOS / Linux):**
```bash
bash scripts/setup.sh
```

One script installs Python, the full AI/ML package stack (covering `notes/ML`, `notes/AIInfrastructure`, and `notes/MultiAgentAI`), VS Code + Twinny, Ollama, registers all Jupyter kernels, and launches both study servers (Jupyter Lab + MkDocs) at fixed local ports so you can start studying immediately.

**Install git hooks (secret scanning pre-commit):**
```bash
bash scripts/install-hooks.sh     # macOS / Linux
.\scripts\install-hooks.ps1       # Windows
```

---

## Study workflow — two servers, two purposes

The setup scripts start both servers on **fixed ports** so you can bookmark the URLs.

| Purpose | URL | What it serves |
|---|---|---|
| **Hands-on coding** | http://localhost:8888 | Jupyter Lab — run and edit every `notebook.ipynb` under `notes/` |
| **Reading the material** | http://localhost:8000 | MkDocs Material — browse every `README.md`, `.md`, **and `notebook.ipynb`** under `notes/` in a web browser (rendered, read-only), with search, dark mode, Mermaid diagrams, and math rendering |

### Launching the servers manually

If the servers are not already running (e.g. after a reboot), activate the venv and start them:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1

# Jupyter Lab — hands-on notebooks
Start-Process python -ArgumentList '-m','jupyter','lab','--no-browser','--ServerApp.port=8888','--ServerApp.ip=127.0.0.1' -WindowStyle Hidden -PassThru | Select-Object -Expand Id | Set-Content .jupyter.pid

# MkDocs — reading site
Start-Process python -ArgumentList '-m','mkdocs','serve','-a','127.0.0.1:8000' -WindowStyle Hidden -PassThru | Select-Object -Expand Id | Set-Content .mkdocs.pid
```

**macOS / Linux:**
```bash
source .venv/bin/activate

# Jupyter Lab — hands-on notebooks
nohup python -m jupyter lab --no-browser --ServerApp.port=8888 --ServerApp.ip=127.0.0.1 > .jupyter.log 2>&1 &
echo $! > .jupyter.pid

# MkDocs — reading site
nohup python -m mkdocs serve -a 127.0.0.1:8000 > .mkdocs.log 2>&1 &
echo $! > .mkdocs.pid
```

### Stopping the servers

**Windows:**
```powershell
Get-Content .jupyter.pid,.mkdocs.pid | ForEach-Object { Stop-Process -Id ([int]$_) -Force }
```

**macOS / Linux:**
```bash
kill $(cat .jupyter.pid .mkdocs.pid 2>/dev/null)
```

> Jupyter Lab's first-run login token is printed to `.jupyter.log` — open the log and copy the `?token=…` URL into your browser.

