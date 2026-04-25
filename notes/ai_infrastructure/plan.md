# Plan — AI Infrastructure Track

**Last updated:** 2026-04-24
**Audit scope:** All 5 chapters under `notes/ai_infrastructure/`
**Running example:** InferenceBase startup (Llama-3-8B self-hosting)
**Grand Challenge:** Replace $80k/mo OpenAI API with self-hosted model at <$15k/mo

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: ai_infrastructure
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/fix_ai_infra_section_headings.py` | All 5 chapter `.md` files | (1) Rename `## 2 · The InferenceBase Angle` → `## 2 · Running Example`; (2) Renumber `## 11 · What Can Go Wrong` → `## 8 · What Can Go Wrong` with downstream renumbering |
| `scripts/create_ai_infra_img_dirs.py` | `memory_and_compute_budgets/`, `quantization_and_precision/`, `parallelism_and_distributed_training/`, `inference_optimization/` | Create missing `img/` directories (Ch.1 gpu_architecture already has `img/`) |
| `scripts/insert_ai_infra_stub_sections.py` | All 5 chapter `.md` files | Insert stub sections after the correct anchor points: (1) `## Animation` after `## 0 · The Challenge` closing line; (2) `## 5 · Key Diagrams` stub with palette comment; (3) `## 6 · The Hyperparameter Dial` stub; (4) `## 7 · Code Skeleton` stub with `# Educational:` / `# Production:` headers; (5) `## Where This Reappears` stub |
| `scripts/fix_ai_infra_notation_placeholders.py` | All 5 chapter `.md` files | Append `<!-- TODO: add notation sentence here -->` placeholder inside each chapter's opening blockquote header (after the curriculum position line) |
| `scripts/remove_ai_infra_duplicate_interview_table.py` | `gpu_architecture/gpu-architecture.md` | Remove duplicate `## 12 · Interview Checklist` section (keep the one inside `## 11.5 · Progress Check`, delete the standalone heading that repeats the same content) |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### Track-wide (all 5 chapters)

- [x] ✅ **Write `## 6 · The Hyperparameter Dial`** for every chapter — done for all 5 (precision/batch/tile for GPU; batch/seq_len/precision for memory; bits/group_size/calibration for quantization; LoRA rank/ZeRO/gradient accumulation for parallelism; max_batch/page_block_size/speculative_depth for inference)
- [x] ✅ **Write `## 7 · Code Skeleton`** for every chapter — Educational (from scratch) + Production (library) pairs added to all 5
- [x] ✅ **Write `## Where This Reappears`** forward links for every chapter — added to all 5
- [ ] **Write notation sentence** for every blockquote header
- [ ] **Write Mermaid chapter arc** for every `## N · Progress Check` section
- [ ] **Add callout boxes** throughout all chapter bodies — 💡 insights, ⚠️ traps, ➡️ forward pointers, 📖 deep dives
- [ ] **Convert ASCII diagrams** to Mermaid blocks with approved color palette

### gpu_architecture/gpu-architecture.md
- [x] ✅ Write `## 6 · The Hyperparameter Dial` — precision dial (FP32→BF16→INT8→FP8), batch size, tile size with trade-off table
- [x] ✅ Write `## Where This Reappears` — Ch.2 (VRAM arithmetic), Ch.3 (bandwidth × quantization), Ch.5 (PagedAttention), Ch.6 (vLLM)
- [x] ✅ Write Code Skeleton — Educational: GPU spec comparison; Production: VRAM/roofline check
- [ ] Convert ASCII roofline chart (§10) to Mermaid with approved palette
- [ ] Add 💡/➡️/⚠️ callouts
- [ ] Write notation sentence

### memory_and_compute_budgets/memory-budgets.md
- [ ] Update blockquote curriculum position sentence to cite specific Ch.1 numbers
- [x] ✅ Write `## 6 · The Hyperparameter Dial` — batch_size/seq_len/precision triad with VRAM impact formulas
- [ ] Convert ASCII VRAM bar chart (§10) to Mermaid block diagram
- [x] ✅ Write Code Skeleton — Educational: manual VRAM budget calculator; Production: pre-flight VRAM check
- [x] ✅ Write `## Where This Reappears` — KV cache → Ch.3, Ch.5, Ch.6
- [ ] Write notation sentence
- [ ] Add 💡/⚠️ callouts

### quantization_and_precision/quantization.md
- [x] ✅ Write `## 6 · The Hyperparameter Dial` — bits dial (4→8→16), group_size, calibration sample count with quality/VRAM/speed table
- [ ] Move `## 9 · Mixed-Precision Strategies (Advanced)` content into `## 6 · The Hyperparameter Dial`
- [ ] Convert §10 ASCII chart to Mermaid
- [x] ✅ Write Code Skeleton — Educational: GPTQ quantization in 20 lines; Production: parameterized pipeline with quality gate
- [x] ✅ Write `## Where This Reappears` — INT4 → Ch.5, Ch.6, Ch.8
- [ ] Write notation sentence

### parallelism_and_distributed_training/parallelism.md
- [ ] **Fix blockquote continuity error** — remove Ch.5 metric references from Ch.4's blockquote
- [ ] Fix "Next chapter" pointer in Progress Check — change to *"Next: Ch.5 · Inference Optimization"*
- [x] ✅ Write `## 6 · The Hyperparameter Dial` — LoRA rank (4→8→16→32→64), ZeRO stage (0→1→2→3), gradient accumulation steps
- [ ] Convert §10 ASCII training memory comparison to Mermaid
- [x] ✅ Write Code Skeleton — Educational: minimal LoRA in 15 lines; Production: parameterized training with evaluation + early stopping
- [x] ✅ Write `## Where This Reappears` — LoRA → Ch.6, Ch.9, Ch.10; ZeRO → Ch.7
- [ ] Write notation sentence

### inference_optimization/inference-optimization.md
- [ ] **Add formal math to `## 3 · The Math`** — Little's Law, throughput formula, PagedAttention waste ratio
- [x] ✅ Write `## 6 · The Hyperparameter Dial` — max_batch_size, page block size, speculative draft depth
- [ ] Convert §10 ASCII timeline diagrams to Mermaid
- [x] ✅ Write Code Skeleton — Educational: 30-line queue simulation; Production: vLLM `AsyncLLMEngine` invocation
- [x] ✅ Write `## Where This Reappears` — continuous batching → Ch.6; PagedAttention → Ch.6; speculative decoding → Ch.10
- [ ] Write notation sentence

---

## Notes
- All 5 chapters pass academic-register and fuzzy-metrics checks — no violations there.
- Failure-first pedagogy is strong in Ch.1–3, Ch.5. Ch.4 is slightly diluted (challenge framing "not blocking launch") — see manual fix for Ch.4 above.
- `## 0 · The Challenge` sections are strong across all chapters with concrete failure scenarios and numbers.
