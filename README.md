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

## Accredited courses this material best supports

This repo is **not accredited** — it's self-authored study material. It is, however, a rigorous companion to paid certifications. Courses below are sorted by how tightly the `notes/` tracks align with the certification's syllabus (tightest fit first).

| Rank | Certification | Tracks that support it | Gap this repo closes |
|---|---|---|---|
| 1 | **DeepLearning.AI — *Deep Learning Specialization* (Coursera, Andrew Ng)** | [PreRequisites](notes/PreRequisites/), [ML](notes/ML/) Ch.1–8 · Ch.15 · Ch.17–18 | Derives the math Coursera states as given; adds production depth Coursera skips. |
| 2 | **DeepLearning.AI — *Machine Learning Specialization* (Coursera)** | [ML](notes/ML/) Ch.1–6 · Ch.9–14 · Ch.19 | California Housing continuity forces real code understanding vs scaffolded notebooks. |
| 3 | **HuggingFace — *NLP Course* + *LLM Course*** | [AI](notes/AI/) (all), [ML](notes/ML/) Ch.17–18, [MultiAgentAI](notes/MultiAgentAI/) | Supplies the "why" behind every HF snippet — tokenisation, RAG, fine-tuning, evaluation. |
| 4 | **NVIDIA Deep Learning Institute — *LLMs* / *Inference Optimization* / *Fundamentals of Deep Learning*** | [AIInfrastructure](notes/AIInfrastructure/) (all), [ML](notes/ML/) Ch.4–8 | Vendor-neutral grounding for GPU architecture, quantisation, vLLM, serving. |
| 5 | **Azure AI Engineer Associate (AI-102)** / **AWS Certified Machine Learning – Specialty (MLS-C01)** | [AI](notes/AI/), [AIInfrastructure](notes/AIInfrastructure/), [MultiAgentAI](notes/MultiAgentAI/) | Cloud exams test services; this repo teaches what the services actually do underneath. |
| 6 | **Stanford Online — *XCS229 / CS229 Machine Learning* (paid professional track)** | [PreRequisites](notes/PreRequisites/) Ch.5–7, [ML](notes/ML/) Ch.5 · Ch.6 · Ch.9 · Ch.11 · Ch.15 | Practical production framing alongside CS229's academic rigor. |
| 7 | **DeepLearning.AI — *Generative AI with LLMs* (Coursera)** | [AI](notes/AI/), [ML](notes/ML/) Ch.18, [MultimodalAI](notes/MultimodalAI/) | Adds agent orchestration, multi-agent protocols, and local-inference economics. |
| 8 | **MIT / edX — *MicroMasters in Statistics and Data Science*** | [PreRequisites](notes/PreRequisites/) Ch.7, [ML](notes/ML/) Ch.9 · Ch.14 · Ch.15 | Use as warm-up, not substitute — MicroMasters is heavier on pure statistics. |

**Recommended primary pairing:** *DeepLearning.AI Deep Learning Specialization + HuggingFace LLM Course*, using [notes/PreRequisites/](notes/PreRequisites/) as the remedial math layer and [notes/AI/](notes/AI/) + [notes/AIInfrastructure/](notes/AIInfrastructure/) as the production layer those courses intentionally skip.

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

One script installs Python, the full AI/ML package stack (covering every track under `notes/`), VS Code + the **Kilo Code** extension wired to a local Ollama-served DeepSeek-R1 model, registers all Jupyter kernels, and launches both study servers (Jupyter Lab + MkDocs) at fixed local ports so you can start studying immediately.
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

