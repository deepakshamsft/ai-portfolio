# ai-portfolio

A personal portfolio combining structured AI/ML study notes and hands-on engineering projects.

---

## Structure

```
ai-portfolio/
├── notes/      ← Study material: every track, every chapter, every notebook
├── projects/   ← Hands-on engineering projects built on the notes
└── scripts/    ← One-shot setup scripts (Python, kernels, MkDocs, Ollama, Kilo Code)
```

---

## `notes/` — Study Material

All learning content lives here: a **19-chapter ML curriculum**, a math foundations track, four AI tracks (Agentic, Multi-Agent, Multimodal, AI Infrastructure), a consolidated interview guide, and runnable Jupyter notebooks throughout.

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
| 1 | **DeepLearning.AI — *Deep Learning Specialization* (Coursera, Andrew Ng)** | [00-MathUnderTheHood](notes/00-math_under_the_hood), [01-ML](notes/01-ml) Ch.1–8 · Ch.15 · Ch.17–18 | Derives the math Coursera states as given; adds production depth Coursera skips. |
| 2 | **DeepLearning.AI — *Machine Learning Specialization* (Coursera)** | [01-ML](notes/01-ml) Ch.1–6 · Ch.9–14 · Ch.19 | California Housing continuity forces real code understanding vs scaffolded notebooks. |
| 3 | **HuggingFace — *NLP Course* + *LLM Course*** | [AI](notes/03-ai) (all), [01-ML](notes/01-ml) Ch.17–18, [MultiAgentAI](notes/04-multi_agent_ai) | Supplies the "why" behind every HF snippet — tokenisation, RAG, fine-tuning, evaluation. |
| 4 | **NVIDIA Deep Learning Institute — *LLMs* / *Inference Optimization* / *Fundamentals of Deep Learning*** | [AIInfrastructure](notes/06-ai_infrastructure) (all), [01-ML](notes/01-ml) Ch.4–8 | Vendor-neutral grounding for GPU architecture, quantisation, vLLM, serving. |
| 5 | **Azure AI Engineer Associate (AI-102)** / **AWS Certified Machine Learning – Specialty (MLS-C01)** | [AI](notes/03-ai), [AIInfrastructure](notes/06-ai_infrastructure), [MultiAgentAI](notes/04-multi_agent_ai) | Cloud exams test services; this repo teaches what the services actually do underneath. |
| 6 | **Stanford Online — *XCS229 / CS229 Machine Learning* (paid professional track)** | [00-MathUnderTheHood](notes/00-math_under_the_hood) Ch.5–7, [01-ML](notes/01-ml) Ch.5 · Ch.6 · Ch.9 · Ch.11 · Ch.15 | Practical production framing alongside CS229's academic rigor. |
| 7 | **DeepLearning.AI — *Generative AI with LLMs* (Coursera)** | [AI](notes/03-ai), [01-ML](notes/01-ml) Ch.18, [MultimodalAI](notes/05-multimodal_ai) | Adds agent orchestration, multi-agent protocols, and local-inference economics. |
| 8 | **MIT / edX — *MicroMasters in Statistics and Data Science*** | [00-MathUnderTheHood](notes/00-math_under_the_hood) Ch.7, [01-ML](notes/01-ml) Ch.9 · Ch.14 · Ch.15 | Use as warm-up, not substitute — MicroMasters is heavier on pure statistics. |

**Recommended primary pairing:** *DeepLearning.AI Deep Learning Specialization + HuggingFace LLM Course*, using [notes/00-MathUnderTheHood/](notes/00-math_under_the_hood) to build mathematical intuition and [notes/03-ai/](notes/03-ai) + [notes/06-ai_infrastructure/](notes/06-ai_infrastructure) as the production layer those courses intentionally skip.

---

## How the tracks fit together — the historical arc

Every track in this repo is the response to a specific historical bottleneck. Reading them in roughly the order the field discovered them makes the curriculum feel inevitable instead of arbitrary. This is a one-paragraph teaser; each track has its own deep timeline (linked at the end of each row).

| Era | The bottleneck that defined it | Where it shows up in this repo |
|---|---|---|
| **Pre-1900s** — *math foundations* | Curves, gradients, and probability had to be invented before "fitting a model" was a coherent idea (Newton/Leibniz → Gauss → Pearson). | [notes/00-MathUnderTheHood/](notes/00-math_under_the_hood#historical-and-chronological-evolution) — Euclid through Rumelhart, mapped to chapters |
| **1805 → 2017** — *classical & deep ML* | Least squares → MLE → perceptrons → AI winter → backprop → CNNs → LSTMs → attention → Transformer. Every chapter exists because an earlier model failed at a specific problem. | [notes/README.md (ML history)](notes/README.md#how-we-got-here--a-short-history-of-machine-learning) — full 30-row timeline aligned to ML Ch.1–19 |
| **2017 → today** — *agentic AI* | Once Transformers existed, the next bottleneck moved up the stack: prompting → CoT reasoning → retrieval → tool use → ReAct → multi-agent orchestration. | [notes/03-ai/AIPrimer.md](notes/03-ai/ai-primer.md#how-we-got-here--a-short-history-of-agentic-ai) |
| **2020 → today** — *multi-agent protocols* | Single agents hit context-window and trust ceilings. The fix was protocol-level: MCP, A2A, event buses, sandboxing. | [notes/04-MultiAgentAI/README.md](notes/04-multi_agent_ai/README.md#how-we-got-here--a-short-history-of-multi-agent-ai) |
| **2014 → today** — *multimodal & generative* | GANs → VAEs → CLIP → DDPM → Latent Diffusion → ControlNet → multimodal LLMs. Each step solved a stability or controllability gap in the previous one. | [notes/05-MultimodalAI/README.md](notes/05-multimodal_ai/README.md#how-we-got-here--a-short-history-of-multimodal--generative-ai) |
| **1999 → today** — *AI infrastructure* | GPU as graphics card → CUDA → tensor cores → HBM → ZeRO → Flash Attention → PagedAttention → 4-bit quantisation. Every chapter exists because the previous bottleneck moved (compute → memory → throughput → cost). | [notes/06-ai_infrastructure/README.md](notes/06-ai_infrastructure/README.md#how-we-got-here--a-short-history-of-ai-infrastructure) |

**The through-line:** math made fitting models possible → classical ML made fitting useful → deep learning made fitting scalable → infrastructure made deep learning affordable → agents made deep learning *act* → multi-agent protocols made agents compose → multimodal made everything see and generate. Read the per-track histories above whenever a chapter feels like it appeared from nowhere.

---

## Quick start

**Set up the full dev environment (Windows):**
```powershell
.\scripts\setup.ps1
# Optional: add --enable-slm-assistant to install Kilo Code + Ollama wiring
# Optional: add --enable-mkdocs-server to launch the local MkDocs docs server
# Optional: add --enable-gpu-notebook-stack to install the GPU notebook deps
```

**Set up the full dev environment (macOS / Linux):**
```bash
bash scripts/setup.sh
# Optional: add --enable-slm-assistant to install Kilo Code + Ollama wiring
# Optional: add --enable-mkdocs-server to launch the local MkDocs docs server
# Optional: add --enable-gpu-notebook-stack to install the GPU notebook deps
```

One script installs Python, the full AI/ML package stack (covering every track under `notes/`), registers all Jupyter kernels, and launches Jupyter Lab at a fixed local port so you can start studying immediately. Pass `--enable-slm-assistant` only if you also want VS Code + the **Kilo Code** extension, Ollama, and the local DeepSeek-R1 wiring. Pass `--enable-mkdocs-server` only if you want the local MkDocs docs server too. Pass `--enable-gpu-notebook-stack` only if you want the CUDA PyTorch and fine-tuning dependencies used by the GPU supplement notebooks.

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

---

## Custom VS Code Agents (No Paid Subscription Required)

This repository includes a suite of **8 local AI agents** designed for repository maintenance, validation, and documentation work. All agents run entirely locally in VS Code with GitHub Copilot — **no external APIs, no paid subscriptions, no authentication required**.

### User-Invocable Agents

| Agent | What it does | Invocation |
|-------|------------|-----------|
| **Content Auditor** | Audit `notes/` for broken links, orphaned notebooks, missing READMEs | `@Content Auditor audit all` |
| **Explore** | Fast discovery of codebase structure and patterns | `@Explore find what I'm looking for (quick\|medium\|thorough)` |
| **Notebook Supplement Guardian** | Validate GPU detection guards in MultimodalAI supplements | `@Notebook Supplement Guardian validate all` |
| **ML Animation Auditor** | Audit ML chapters for animation opportunities and consistency | `@ML Animation Auditor audit 01-Regression` |

### Internal Agents (Batch/Workflow)

- **ML Animation Coordinator** — orchestrate animation rollout across chapters
- **ML Animation Needle Builder** — build metric-movement animations
- **ML Animation Doc Sync** — sync animation documentation
- **Multimodal Animation Builder** — generate MultimodalAI flow animations

### Quick Start

1. Open **Copilot Chat** in VS Code (`Ctrl+Shift+I`)
2. Type `@` to see available agents
3. Type `@Content Auditor audit all` and press Enter

→ **Full documentation**: [AGENTS.md](AGENTS.md)

---

