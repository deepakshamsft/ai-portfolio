# Reference Books — Expansion Plan

> Parked plan. Do not execute until explicitly asked. When resuming, read this top-to-bottom and also re-read `AUTHORING_GUIDE.md` for tone/structure rules before touching any HTML.

---

## Current state (as of this plan)

| Book | File | Tone | Coverage |
|---|---|---|---|
| Neural Chronicles | `Neural_Chronicles.html` / `.pdf` | Technical reference — diagram-first interview prep, no story arc | ML Ch.1–14 |
| ML Chronicles | `ML_Chronicles.html` / `.pdf` | **Manga story-arc** (Oracle, Elder, Knight, Alchemist, Gate Keeper…) | 10 curated classical-ML chapters that overlap Neural Chronicles topics |
| AI Chronicles | *(does not exist yet)* | — | — |

Gaps vs `notes/` library:
- Neural Chronicles is missing ML Ch.15–19: MLE & Loss Functions, TensorBoard, Sequences→Attention, Transformers & Attention, Hyperparameter Tuning.
- ML Chronicles covers only 10 of the 19 ML chapters; no chapter for Linear/Logistic Regression, XOR, NNs, Backprop, Regularisation, CNNs, RNNs, Metrics, MLE, TensorBoard, Transformers, HP Tuning.
- `AI/`, `MultiAgentAI/`, `MultimodalAI/`, `AIInfrastructure/` tracks have no PDF reference book.

The `AUTHORING_GUIDE.md` references `pdf-gen/ && node generate-pdf.mjs neural` — **that folder does not exist in the repo**. PDF generation pipeline must be built before any PDF can be regenerated.

---

## User-confirmed decisions

1. **Neural Chronicles:** add Ch.15–19 only. Do not fold in AIInfrastructure.
2. **ML Chronicles:** expand from 10 → 19 chapters, preserving the manga story-arc tone. New chapters must fit the existing character/world framework (Oracle, Elder, Knight, Alchemist, Gate Keeper, Cartographer, etc.), not the Neural Chronicles reference tone.
3. **AI Chronicles (new book):** technical-reference tone (same style as Neural Chronicles). Single book covering all four tracks: `AI/` + `MultiAgentAI/` + `MultimodalAI/` + `AIInfrastructure/`.
4. **PDF pipeline:** Node + Puppeteer generator under `scripts/pdf-gen/`, regenerate all PDFs from HTML.

---

## Phased execution plan

### Phase 1 — Pipeline + Neural Chronicles completion

**Deliverables**
- `scripts/pdf-gen/` with `package.json`, `generate-pdf.mjs`, README.
  - CLI: `node generate-pdf.mjs <neural|ml|ai|all>`.
  - Uses Puppeteer with `printBackground:true`, A4, `preferCSSPageSize:true`.
  - Waits for KaTeX auto-render to complete before `page.pdf()`.
- Neural Chronicles Ch.15–19 authored at existing quality bar (splash + 4 content + interview = 6 pages per chapter = **30 new pages**).
- Update chapter map on page 3 of Neural Chronicles to show 19 chapters.
- Regenerate `Neural_Chronicles.pdf`.

**Per-chapter plan (already scoped in `AUTHORING_GUIDE.md`):**

| Ch | Title | Accent | Core concept |
|---|---|---|---|
| 15 | MLE & Loss Functions | TBD (suggest `--eval2` violet or new `--mle`) | Derive MSE from Gaussian NLL; derive Cross-Entropy from categorical NLL. MAP = MLE + prior → L2 from Gaussian prior, L1 from Laplace prior. |
| 16 | TensorBoard | new `--tb` orange | Scalars (loss/acc curves), histograms (weight drift), distributions, image/embedding projector, HParams dashboard. `tf.summary.*`, Keras callback. |
| 17 | From Sequences to Attention | new `--attn1` | **Bridge chapter**. Soft dictionary lookup: query · key → softmax → weighted values. Derive attention without transformers. Ties back to RNN hidden-state lookup. |
| 18 | Transformers & Attention | new `--attn2` deep indigo | Scaled dot-product: softmax(QK^T / √d_k) V. Multi-head concat-project. Positional encoding (sinusoidal + learned). Residual + LayerNorm. Encoder/decoder vs decoder-only. |
| 19 | Hyperparameter Tuning | `--gold` | Tune order: LR first, then batch size, then optimiser, then regularisation, then init. Grid vs random vs Bayesian (Optuna) vs ASHA/Hyperband. Budget math. |

**Bridges**
- Ch.14 → Ch.15: unsupervised metrics → "where do these losses come from in the first place?"
- Ch.15 → Ch.16: now that losses are principled, make training observable.
- Ch.16 → Ch.17: observability shows attention weights naturally — cue the bridge.
- Ch.17 → Ch.18: attention generalised → full transformer architecture.
- Ch.18 → Ch.19: bigger models amplify HP sensitivity → systematic tuning.

**Images needed (`notes/ML/img/`)**
- `ch15-mle-gaussian.png` — Gaussian likelihood surface over (μ,σ) with MLE peak.
- `ch15-prior-posterior.png` — MLE vs MAP with Gaussian prior (L2) vs Laplace prior (L1).
- `ch16-tensorboard-dashboard.png` — annotated scalars/histograms/projector screenshot.
- `ch17-soft-lookup.png` — dictionary lookup → attention weights visualisation.
- `ch18-multi-head-attention.png` — Q/K/V split into h heads, scaled dot, concat, project.
- `ch18-positional-encoding.png` — sinusoidal frequency bands over position.
- `ch19-tune-order.png` — funnel: LR → batch → optimiser → reg → init.

### Phase 2 — AI Chronicles (book 3)

**Structure** (target ~120 pages):
- Cover + Prologue (three layers: model / agent / platform) + Chapter map.
- **Part I — Agentic AI** (from `notes/AI/`): 12 chapters
  - LLM Fundamentals, Prompt Engineering, CoT Reasoning, RAG & Embeddings, Vector DBs, ReAct & Semantic Kernel, Evaluating AI Systems, Fine-Tuning, Safety & Hallucination, Cost & Latency, AI Interview Primer capstone.
- **Part II — Multi-Agent** (from `notes/MultiAgentAI/`): 7 chapters
  - Message Formats, MCP, A2A, Event-Driven Agents, Shared Memory, Trust & Sandboxing, Agent Frameworks.
- **Part III — Multimodal** (from `notes/MultimodalAI/`): 12 chapters
  - Multimodal Foundations, Vision Transformers, CLIP, Diffusion, Latent Diffusion, Schedulers, Guidance/Conditioning, Text-to-Image, Text-to-Video, Multimodal LLMs, Generative Evaluation, Local Diffusion Lab.
- **Part IV — AI Infrastructure** (from `notes/AIInfrastructure/`): 10 chapters
  - GPU Architecture, Memory & Compute Budgets, Quantization, Parallelism, Serving Frameworks, Inference Optimization, Networking & Clusters, MLOps, Production AI Platform, Cloud AI Infra.

**Design rules**
- Reuse Neural Chronicles CSS base (colours, `.box`, `.lock`, `.warn`, `.sepia`, `.bridge`, `.dial`, `.ch-nav`, `.ctag`, `.code-block`, `.math-block`).
- Add per-part accent palette: `--agent`, `--multiagent`, `--multimodal`, `--infra`.
- Each chapter = splash + 2–4 content pages + interview page (fewer than 4 content pages when the source note is shorter; judgement per chapter).
- Running examples already in the source notes must be preserved: **PizzaBot** (AI), **OrderFlow** (MultiAgent), **PixelSmith** (Multimodal), **InferenceBase** (Infra).
- Every chapter ends with the 3-column interview checklist and a Q&A page.
- Skip `_Supplement.md` content — keep it in `notes/`, don't bloat the PDF.

**Phase 2a — skeleton**: cover, prologue, chapter map, shared CSS, and one fully-authored chapter per Part as the style template for that Part (suggested templates: LLM Fundamentals, MCP, CLIP, Serving Frameworks).
**Phase 2b**: author remaining chapters Part by Part. Regenerate PDF after each Part to catch layout drift early.

### Phase 3 — ML Chronicles manga expansion

**Keep the story-arc.** Map new topics to existing or new characters consistent with the lore:

| New ML Ch | Existing characters to extend | Likely chapter metaphor |
|---|---|---|
| Linear Regression | Oracle (prophecy as prediction) | "The First Prophecy" — a straight-line oracle |
| Logistic Regression | Oracle → Gatekeeper precursor | "The Verdict Stone" — yes/no prophecy |
| XOR Problem | A cursed puzzle only a two-layer seer can solve | "The Twin-Mirror Riddle" |
| Neural Networks | Council of Oracles | "The Council Assembles" |
| Backprop & Optimisers | Forge / Smith character | "The Backforge" — hammering gradients backward |
| Regularisation | Monk of Restraint | "The Vow of Less" |
| CNNs | Scribe of Patterns | "The Tile Reader" |
| RNNs / LSTMs | Keeper of Memory | "The Lantern-Bearers" |
| MLE | Oracle's origin story | "The First Law Revealed" |
| TensorBoard | The Watchtower | "The All-Seeing Scope" |
| Sequences → Attention | Herald who routes messages | "The Herald's Gaze" |
| Transformers | Parliament of Heralds | "The Many-Eyed Court" |
| HP Tuning | The Tuning Monks | "The Twelve Dials" |

**Challenges**
- Tone consistency: ML Chronicles uses heavy manga-grid CSS (`.mg`, `.pnl`, `.spd`, `.tone`) — all new pages must match this visual language, not the Neural Chronicles panel style.
- Topic overlap: new ML Chronicles chapters for CNNs/RNNs/etc. will overlap Neural Chronicles. Accept this — the two books serve different readers (narrative learner vs reference reader).
- Story continuity: place new chapters in chronological order of the saga, not the numeric chapter order of the `notes/ML/` library. Provide a crosswalk table in the prologue.

### Phase 4 — Housekeeping

- Update `AUTHORING_GUIDE.md` page inventory after every phase.
- Add an `AUTHORING_GUIDE_AI.md` and `AUTHORING_GUIDE_ML_CHRONICLES.md` if per-book rules diverge meaningfully from the current (Neural-focused) guide.
- Add `scripts/pdf-gen/README.md` with regenerate commands.
- Wire pre-commit hook (`scripts/hooks/pre-commit`) to warn when an HTML changes without a matching PDF regeneration.

---

## Open questions to resolve before Phase 2 starts

1. Should AI Chronicles use per-Part colour palettes (recommended) or a single unified palette?
2. Should each chapter have its own companion interview page, or one consolidated "Interview Prep" part at the end?
3. Ship `.pdf` under version control or gitignore and rely on pipeline? (Currently tracked — 3.5 MB combined so far.)
4. Images: continue AI-generated PNGs under each track's `img/` folder, or centralise in `notes/Reference/img/`?

---

## Resume checklist

When asked to proceed:
1. Re-read this file and `AUTHORING_GUIDE.md`.
2. Confirm which Phase (1/2a/2b/3/4) to execute.
3. If Phase 1: start with `scripts/pdf-gen/` pipeline before authoring any chapter.
4. For every new chapter, follow the 6-page template, add the `.bridge` box, verify every page fits in 297 mm height, and update the chapter map on page 3.
5. After each chapter is authored, regenerate the affected PDF and eyeball pages 1, middle, and last for overflow.
