# Plan — ML / 03 Neural Networks

**Last updated:** 2026-04-24
**Audit source:** `notes/ml/content-audit.md` (April 2026)
**Chapters:** ch01–ch10 (XOR through Transformers)

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: ml/03_neural_networks
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/fix_nn_mission_names.py` | `ch01_xor_problem/README.md`, `ch07_mle_loss_functions/README.md` | Replace `Launch **SmartVal AI**` with `Launch **UnifiedAI**` in `## 0 · The Challenge` blockquotes |
| `scripts/fix_nn_duplicate_sections.py` | `ch03_backprop_optimisers/README.md`, `ch06_rnns_lstms/README.md`, `ch10_transformers/README.md` | (1) Remove the second `## 9 · Where This Reappears` stub at the end of ch03; (2) Move `## 9 · Where This Reappears` from top of ch06 to after § 8; (3) Move same misplaced section in ch10 to after § 8 |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### ch01_xor_problem — ⚡ HIGH PRIORITY
- [ ] `## 0 · The Challenge` blockquote: change mission name from `SmartVal AI` → `UnifiedAI` *(also covered by script)*
- [ ] Standardise Constraint #1 target: use `<$50k MAE` consistently across all NN chapters as the intermediate milestone; `$28k MAE + 95% accuracy` is the final track-level target only (stated in the track README)
- [ ] Remove `🎯` and `🚨` unapproved emoji from prose — replace `🎯` with no emoji (plain text in blockquote), replace `🚨` with `⚠️`

### ch02_neural_networks — ⚡ HIGH PRIORITY
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] **Add numeric walkthrough to Section 3 (Math):** feed a 3-row toy dataset through the network layer by layer, showing numeric activations at each layer
- [ ] Remove `🚀`, `🎯`, `🚨` unapproved emoji from prose

### ch03_backprop_optimisers — ⚡ HIGH PRIORITY
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] Fix Constraint #1 target inconsistency: `## 0 · The Challenge` states `<$40k MAE` (wrong — should be `<$50k MAE`)
- [ ] Remove the duplicate `## 9 · Where This Reappears` stub at the bottom of the file *(also covered by script)*
- [ ] Remove `🚀` unapproved emoji from prose

### ch04_regularisation
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] Fix constraint status: ch04 re-claims `Constraint #1 ACHIEVED ($48k MAE)` via `Ch.5` — update to frame ch04's contribution as `Constraint #2 GENERALIZATION`
- [ ] Clarify the Progress Check: label the $48k MAE as training MAE to avoid misleading readers
- [ ] Remove `🚀` unapproved emoji

### ch05_cnns
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] **Add a policy note** in the Running Example acknowledging the synthetic 8×8 pixel grid is an educational proxy
- [ ] Remove `🚀` unapproved emoji

### ch06_rnns_lstms — ⚡ HIGH PRIORITY
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] Move `## 9 · Where This Reappears` (currently at the very top of the file) to after `## 8 · What Can Go Wrong` *(also covered by script)*
- [ ] Add a policy note in the Running Example: the synthetic monthly price index dataset is an educational proxy
- [ ] Remove `🚀` unapproved emoji

### ch07_mle_loss_functions — ⚡ HIGH PRIORITY
- [ ] `## 0 · The Challenge` blockquote: change mission name `SmartVal AI` → `UnifiedAI` *(also covered by script)*
- [x] ✅ Fix "What we know so far" stale chapter references (wrong constraint numbering fixed)
- [ ] Remove `🚀` and `🤔` unapproved emoji

### ch08_tensorboard — ⚡ HIGH PRIORITY
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] **Add numeric walkthrough to Section 3 (Math/Diagnostics):** fill the incomplete gradient health table with a concrete worked example
- [ ] Remove `🚀` unapproved emoji

### ch09_sequences_to_attention
- [x] ✅ Fix "What we know so far" stale chapter references
- [ ] Remove `🤔` and `🚀` unapproved emoji

### ch10_transformers
- [ ] Move `## 9 · Where This Reappears` from top of file to after `## 8 · What Can Go Wrong` *(also covered by script)*
- [ ] Remove `🚀` unapproved emoji
