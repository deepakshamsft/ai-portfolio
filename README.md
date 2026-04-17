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

One script installs Python, the full AI/ML package stack (covering `notes/ML`, `notes/AIInfrastructure`, and `notes/MultiAgentAI`), VS Code + Twinny, Ollama, and registers all Jupyter kernels.

**Install git hooks (secret scanning pre-commit):**
```bash
bash scripts/install-hooks.sh     # macOS / Linux
.\scripts\install-hooks.ps1       # Windows
```

