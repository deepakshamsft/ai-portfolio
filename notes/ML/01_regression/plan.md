# Plan — ML / 01 Regression

**Last updated:** 2026-04-24
**Audit source:** `notes/ml/content-audit.md` + `notes/ml/validation-report.md`
**Chapters:** ch01–ch07

## Legend
- 🐍 = Python script needed — listed in Scripts table below, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: ml/01_regression
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/fix_regression_crossrefs.py` | `ch01_linear_regression/README.md`, `ch02_multiple_regression/README.md` | Replace all `../../../../MathUnderTheHood/` with `../../../math_under_the_hood/` (4-level → 3-level relative path) |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### ch01_linear_regression
- [ ] Fix broken cross-reference: line 134 contains `../../../../MathUnderTheHood/ch05-matrices/` — correct to `../../../math_under_the_hood/ch05_matrices/` *(also covered by script above)*
- [ ] Verify line 1027 `../../../math_under_the_hood/` link is correct depth and path matches folder naming (`ch05_matrices` not `ch05-matrices`)

### ch02_multiple_regression
- [ ] Fix broken cross-references at lines 274 and 447: `../../../../MathUnderTheHood/` → `../../../math_under_the_hood/` *(also covered by script)*
- [ ] After fix, verify all MathUnderTheHood links in this file resolve correctly

### ch03_feature_importance
- [ ] Remove the standalone `## Notation` section that sits between the opening blockquote and `## 0 · The Challenge` — all notation belongs in the opening blockquote only, per authoring guidelines §4
- [ ] Confirm the notation symbols (ρ, VIF, etc.) are present in the blockquote header inline sentence after removing the standalone section

### ch04_polynomial_features – ch07_hyperparameter_tuning
- [ ] No structural issues flagged. Verify emoji: `🎯` / `🚀` / `🎉` in prose bullets in these chapters should be replaced with `⚡` per the approved emoji set (see validation-report.md Check 1 for exact line numbers in ch05–ch07)

---

## Notes
- 02-Classification, 04-RecommenderSystems, 05-AnomalyDetection, 06-RL, 08-EnsembleMethods all passed content audit — no plan.md needed for those tracks beyond the universal items listed here.
- ch04–ch07 in this track also passed content audit. The emoji cleanup above is a low-priority polish item.
