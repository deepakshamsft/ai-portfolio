# Plan — Multimodal AI Track

**Last updated:** 2026-04-24
**Audit scope:** All 13 chapter directories under `notes/multimodal_ai/`
**Running example:** VisualForge Studio (local diffusion pipeline, <30s/image, ≥4.0/5.0 quality)

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## Track-Wide Context

The audit found a **systematic mismatch** between the universal section names (from `notes/authoring-guidelines.md`) and what the track's own `authoring-guide.md` requires. Chapters correctly follow the **track** authoring guide but diverge from the universal template. The fixes below reconcile both.

**Sections that need renaming (in every chapter):**

| Current heading | Required heading (universal template) | Track guide name |
|---|---|---|
| `## 4 · How It Works — Step by Step` | `## 4 · Visual Intuition` | `## 4 · Visual Intuition` |
| `## 5 · The Key Diagrams` | `## 5 · Production Example — VisualForge in Action` | `## 5 · Production Example — VisualForge in Action` |
| `## 6 · What Changes at Scale` | `## 6 · Common Failure Modes` | `## 6 · Common Failure Modes` |
| `## 7 · Common Misconceptions` | `## 7 · When to Use This vs Alternatives` | `## 7 · When to Use This vs Alternatives` |
| `## 8 · Interview Checklist` | `## 8 · Connection to Prior Chapters` | `## 8 · Connection to Prior Chapters` |

**Sections missing from all chapters:**
- `## 10 · Further Reading` (required by track authoring guide)
- `## 11 · Notebook` (explicit section linking to notebook file)
- `## 11.5 · Progress Check` (currently at `## 8.5` — must move to after `## 11`)
- Needle GIF (`chNN-[topic]-needle.gif`) in every `img/` folder
- `# Educational:` / `# Production:` labels on all code blocks

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: multimodal_ai
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/fix_multimodal_section_headings.py` | All 13 chapter `.md` files | Rename §4–§8 headings to track-authoritative names (Visual Intuition, Production Example, Common Failure Modes, When to Use, Connection to Prior Chapters) |
| `scripts/fix_multimodal_progress_check_position.py` | All 13 chapter `.md` files | Move `## 8.5 · Progress Check` block to after a new `## 11 · Notebook` section (renumber to `## 11.5 · Progress Check`) |
| `scripts/add_multimodal_stub_sections.py` | All 13 chapter `.md` files | Insert stub sections `## 10 · Further Reading` and `## 11 · Notebook` between the Bridge section and Progress Check |
| `scripts/add_multimodal_code_labels.py` | All 13 chapter `.md` files | Prepend `# Educational: [concept] from scratch` or `# Production: pipeline([...]) call` to first line of each code block based on surrounding context (regex-based heuristic: blocks with loop-level constructs = Educational; single `pipe(` call = Production) |
| `scripts/fix_multimodal_illustration_filenames.py` | All 13 chapter `.md` files | Rename dangling `## Illustrations` references from `img/[Title With Spaces].png` to `img/[kebab-case].png` matching actual file-system naming convention |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### Track-wide (all 13 chapters)

- [ ] **Create needle GIFs** — each chapter needs a `chNN-[topic]-needle.gif` animation
- [x] ✅ **Write `## 5 · Production Example — VisualForge in Action`** for 7/13 chapters — diffusion_models, schedulers, guidance_conditioning, latent_diffusion, generative_evaluation, multimodal_llms, text_to_image done; remaining: multimodal_foundations, vision_transformers, clip, text_to_video, audio_generation
- [ ] **Write `## 6 · Common Failure Modes`** for each chapter
- [ ] **Write `## 7 · When to Use This vs Alternatives`** for each chapter
- [ ] **Write `## 10 · Further Reading`** for each chapter
- [ ] **Add notation sentence** to each blockquote header
- [ ] **Write Mermaid arcs** for all Progress Check sections

### multimodal_foundations
- [ ] Rewrite `## 2 · Running Example` — current example "city skyline at night" violates Red Line #1; replace with a named VisualForge campaign brief
- [ ] Fix dangling `## Illustrations` reference: `img/Multimodal Foundations.png` (spaces in filename) — rename file and update reference

### vision_transformers
- [ ] Rewrite `## 2 · Running Example` — current example describes only tensor shapes with no VisualForge campaign context
- [ ] Write `## 5 · Production Example` showing VisualForge image-search using ViT embeddings

### clip
- [ ] Rewrite `## 2 · Running Example` to use a named VisualForge brief type for zero-shot classification
- [ ] Write `## 5 · Production Example`: VisualForge zero-shot classification
- [ ] Write `## 6 · Common Failure Modes`: caption noise, mode collapse, domain gap

### diffusion_models
- [x] ✅ Rewrote `## 2 · Running Example` — MNIST clearly labeled as educational proxy; VisualForge framing added
- [x] ✅ Added `## 5 · Production Example — VisualForge in Action` (DDPM → SD pipeline with product-on-white brief)

### schedulers
- [x] ✅ Rewrote `## 2 · Running Example` — VisualForge spring-collection brief with timing table
- [x] ✅ Added `## 5 · Production Example` (DDIM vs DPM-Solver comparison with benchmark code)

### latent_diffusion
- [x] ✅ Rewrote `## 2 · Running Example` — lifestyle scene brief; MNIST refs replaced with pixel-scale proxy note
- [x] ✅ Added `## 5 · Production Example` (SDXL-Turbo with constraint scorecard)
- [ ] Write `## 6 · Common Failure Modes`: VAE color shift, scaling factor forgotten (× 0.18215), OOM with large resolution

### guidance_conditioning
- [x] ✅ Rewrote `## 2 · Running Example` — CFG scale sweep on VisualForge spring-collection brief
- [x] ✅ Added `## 5 · Production Example` (brand-constrained product-on-white with negative prompts)

### text_to_image
- [x] ✅ Rewrote `## 2 · Running Example` — ControlNet product-at-angle brief
- [x] ✅ Added `## 5 · Production Example` (ControlNet Canny edge conditioning)

### text_to_video
- [ ] Rewrite `## 2 · Running Example` — replace synthetic sequences with VisualForge animated product-demo brief
- [ ] Write `## 5 · Production Example`: VisualForge animated product-demo brief (5-second loop, 8 frames at 512×512)

### audio_generation
- [ ] Verify Running Example uses VisualForge (or appropriate audio brief), not generic prompts
- [ ] Apply track-wide script fixes

### multimodal_llms
*(Audit pending)*
- [ ] Apply track-wide script fixes
- [ ] Verify VisualForge context in Running Example

### local_diffusion_lab
*(Audit pending)*
- [ ] Apply track-wide script fixes
- [ ] This chapter may serve as a capstone — verify Progress Check shows all VisualForge constraints met

### generative_evaluation
*(Audit pending)*
- [ ] Apply track-wide script fixes
- [ ] Write `## 6 · Common Failure Modes` for evaluation pitfalls (CLIP score gaming, FID sensitivity to dataset size, human rater inconsistency)

---

## Notes
- `notebook_supplement.ipynb` GPU guards (`cuda.is_available()` + `raise SystemExit`) are confirmed present in all chapters ✅ — no fixes needed there.
- `## 0 · The VisualForge Studio Challenge` sections are strong across all chapters ✅.
- Academic register and fuzzy metrics are clean ✅.
