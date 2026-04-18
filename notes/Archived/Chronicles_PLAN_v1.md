# ML Chronicles — Expansion Plan

> Parked plan. Do not execute without checking in first. When resuming, read this top-to-bottom and re-read `AUTHORING_GUIDE.md` for Neural-Chronicles tone, but author in the manga tone established by the existing `ML_Chronicles.html` pages.

---

## Scope

**Only** the manga story-arc expansion of `ML_Chronicles.html`. Everything else previously parked in this plan (Neural Chronicles Ch.15–19, the standalone AI Chronicles book, the Node/Puppeteer PDF pipeline, pre-commit hooks, per-book authoring guides) is **out of scope** — that material is already covered in depth under `notes/AI/`, `notes/MultiAgentAI/`, `notes/MultimodalAI/`, `notes/AIInfrastructure/`, and `notes/ML/ch15-*` through `notes/ML/ch19-*`, and does not need a second surface.

---

## Current state

- `ML_Chronicles.html` has **10 manga chapters** covering classical ML only: KNN, Decision Trees, Ensembles, SVM, Comparison, t-SNE/UMAP, DBSCAN/HDBSCAN, Feature Engineering, Evaluation, Activation Functions.
- `notes/ML/` has **19 chapters** (Ch.1–Ch.19). The book is silent on chapters that map to deep-learning material: Linear/Logistic Regression, XOR, Neural Networks, Backprop & Optimisers, Regularisation, CNNs, RNNs/LSTMs, Transformers.
- Ch.10 (Activations) explicitly closes on Kai saying *"The gate is open. I know what's on the other side now."* — a ready-made hand-off into the deep-learning arc.

---

## Target roster — 19 chapters

Existing 10 stay as-is. Nine new chapters (Ch.11–Ch.19) extend the saga into deep learning, keeping Kai as the POV detective and introducing new characters to fit the lore.

| New Ch | Title | Character | Maps to `notes/ML/` |
|---|---|---|---|
| 11 | The First Prophecy | Oracle precursor (young Oracle) — "you are who you stand next to" generalised to "you are the line that passes through" | `ch01-linear-regression` |
| 12 | The Verdict Stone | Judge-Priest (Gatekeeper precursor) — binary prophecy via sigmoid | `ch02-logistic-regression` |
| 13 | The Twin-Mirror Riddle | The Puzzle Box — a cursed lattice no single seer can read | `ch03-xor-problem` |
| 14 | The Council Assembles | The Council of Oracles — many seers voting through weighted channels | `ch04-neural-networks` |
| 15 | The Backforge | The Forge-Smith — hammering gradients backward through the chain rule | `ch05-backprop-optimisers` |
| 16 | The Vow of Less | The Monk of Restraint — L1/L2/Dropout as ascetic disciplines | `ch06-regularisation` |
| 17 | The Tile Reader | The Scribe of Patterns — reading an image tile by tile | `ch07-cnns` |
| 18 | The Lantern-Bearers | The Keepers of Memory — carrying hidden state through time | `ch08-rnns-lstms` |
| 19 | The Many-Eyed Court | Parliament of Heralds — attention as routed gaze; one chapter covers both the sequences→attention bridge and full transformers | `ch17-sequences-to-attention` + `ch18-transformers` |

**Intentionally excluded from the manga:** MLE & Loss Functions (Ch.15), TensorBoard (Ch.16), Hyperparameter Tuning (Ch.19) — these are reference-style and don't carry a narrative arc. They remain in `notes/ML/` only.

Target length: **~55 new pages** once chapter bodies are authored (splash + 2–4 content + interview per chapter).

---

## Execution in phases

### Phase A — Foundation + skeleton (DONE in this pass)

Applied to `ML_Chronicles.html`:
- CSS colour vars for the nine new characters.
- Cover page topic keywords extended to include DL material.
- Prologue **Cast of Characters** expanded with the nine new figures.
- Nine splash pages + one placeholder content page per new chapter inserted before the Epilogue, each marked with a `TODO` comment.
- Quick Reference card extended with the new algorithm rows.
- Epilogue wording refreshed so it closes over 19 chapters, not 10.

### Phase B — Author chapter bodies (NOT DONE — resume here)

For each new chapter:
1. Replace the single placeholder page with 2–4 manga content pages using the established CSS language (`.mg`, `.pnl`, `.spd`, `.tone`, `.bbl`, `.spk`, `.sfx`, `.lock`, `.note`, `.correct`, `.check`, `.warn`, `.ava`, `.bridge`).
2. Add a 5th page: dedicated interview Q&A plus the 3-column checklist, matching the pattern used in existing chapters.
3. Preserve the recurring **"same dial"** callback from the classical-ML arc (K → depth → n_estimators → C → … → learning rate → dropout rate → attention heads). Every new chapter must land one of these callbacks.
4. Pull math, code snippets, and diagrams from the matching `notes/ML/ch*/` folder — do not re-derive.

### Phase C — Regenerate PDF (NOT DONE)

`ML_Chronicles.pdf` is stale. The repo currently has no PDF generator under version control — when a pipeline is added (or via any local Chromium print-to-PDF), regenerate `ML_Chronicles.pdf` after Phase B.

---

## Authoring rules (carry-over from existing chapters)

- One page per `<div class="page">`. Never split a page. Never exceed 297 mm content height — if it overflows, split a panel or remove a sub-figure.
- Every non-splash page begins with `<div class="ch-nav">CH.N — TOPIC</div>`.
- Speech bubbles attribute with `.spk kai|oracle|gini|knight|bag|boost|alc|gate|...`; add a new CSS class for each new speaker introduced.
- `.lock` boxes carry the recurring-dial callbacks. At least one per chapter.
- Keep Kai as the POV throughout. New characters speak to Kai, not about him.
- Avoid the reference-style voice used in Neural Chronicles — the ML Chronicles is deliberately narrative.

---

## Resume checklist

When asked to proceed:
1. Re-read this file.
2. Confirm the target chapter to author next (start Ch.11 and progress in order — the bias-variance callback chain depends on preceding chapters existing).
3. Open the matching `notes/ML/ch*/` folder for content; do not invent facts.
4. Author against the placeholder page marked `TODO: Ch.NN body` in `ML_Chronicles.html`.
5. Eyeball pages for overflow after every chapter by opening the HTML in a browser.
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
