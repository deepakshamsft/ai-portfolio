# Plan — Interview Guides

**Last updated:** 2026-04-24
**Audit scope:** All 4 interview guides under `notes/interview_guides/`
**Standard template:** 4 sections per `notes/interview_guides/authoring-guide.md`

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: interview_guides
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| ~~`scripts/fix_interview_agentic_duplicate.py`~~ | ~~`agentic-ai.md`~~ | ✅ Done manually — duplicate `## Illustrations` removed when adding `## 3 · The Rapid-Fire Round` |
| `scripts/create_interview_img_dir.py` | `notes/interview_guides/` | Create `img/` directory (referenced by guides but missing) |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

> **Context:** All 4 guides have strong senior-quality Q&A content. The gap is structural — they use chapter-by-chapter or SECTION-flat formats instead of the authoritative 4-section template. The fixes below add the missing scaffold around existing content rather than rewriting from scratch.

### Required 4-section template (from `authoring-guide.md`)
1. `## 1 · Concept Map — The 10 Questions That Matter`
2. `## 2 · Section-by-Section Deep Dives` (with sub-structure: What They're Testing / Junior vs Senior Answer / Key Tradeoffs / Failure Mode Gotchas / Production Angle)
3. `## 3 · The Rapid-Fire Round` (20 Q&A pairs, ≤3 sentences each)
4. `## 4 · Signal Words That Distinguish Answers` (✅ say this / ❌ don't say this vocabulary list)

---

### agentic-ai.md
- [x] ✅ Remove duplicate `## Illustrations` section *(done manually)*
- [x] ✅ Write `## 1 · Concept Map`
- [x] ✅ Restructure each SECTION under `## 2 · Section-by-Section Deep Dives`
- [x] ✅ Write `## 3 · The Rapid-Fire Round` — 20 Q&A pairs
- [x] ✅ Write `## 4 · Signal Words`

### multi-agent-ai.md
- [x] ✅ Write `## 1 · Concept Map` — 10 question clusters (message routing, MCP N×M, A2A lifecycle, pub/sub, blackboard, prompt injection, framework tradeoffs, idempotency, auth, protocol composition)
- [x] ✅ Add `## 2 · Section-by-Section Deep Dives` wrapper
- [x] ✅ Write `## 3 · The Rapid-Fire Round` — 20 Q&A pairs
- [x] ✅ Write `## 4 · Signal Words`

### ai-infrastructure.md
- [x] ✅ Write `## 1 · Concept Map` — 10 clusters (roofline, KV cache, quantization, TP/DP, prefill/decode, PagedAttention, serving, cloud economics, MLOps, TTFT/P99)
- [x] ✅ Add `## 2 · Section-by-Section Deep Dives` wrapper
- [x] ✅ Convert `## Quick Reference Table` into `## 3 · The Rapid-Fire Round` — 20 Q&A pairs
- [x] ✅ Write `## 4 · Signal Words`

### multimodal-ai.md
- [x] ✅ Write `## 1 · Concept Map` — 10 clusters (patch tokenization, ViT, CLIP/InfoNCE, diffusion, schedulers, latent diffusion, CFG, U-Net, FID/CLIP Score, production latency)
- [x] ✅ Add `## 2 · Section-by-Section Deep Dives` wrapper
- [x] ✅ Write `## 3 · The Rapid-Fire Round` — 20 Q&A pairs
- [x] ✅ Write `## 4 · Signal Words`

---

## Notes
- All 4 guides have accurate, senior-quality technical content — the work here is restructuring and labeling, not rewriting.
- The `notebook.ipynb` file exists in `notes/interview_guides/` — unclear if it's linked from any guide; add a reference once content fixes are done.
