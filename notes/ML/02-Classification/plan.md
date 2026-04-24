# Plan — 02-Classification (remaining TODOs only)

---

## Per-chapter TODOs

- [ ] **Emoji cleanup** — Replace remaining non-approved emojis in all sub-chapters. Approved set: `{💡, ⚠️, ⚡, 📖, ➡️}`.
  - `ch03-metrics/README.md`: Remove `💀` on the `Recall $= 0/19 = 0\%$ 💀` line in the Math section. (Verified still present.)
  - All chapters: Replace standalone `✅` / `❌` used as callout prefixes **outside** of status/progress tables with approved alternatives. (Verified still present in all 5 chapters — see details below.)
    - `ch01-logistic-regression/README.md`: 4 bullet violations (`- ✅ Topic 01:…`, `- ✅ Understand MSE…`, `- ❌ **But we can only predict…`, `- ✅ **Constraint #1 PARTIAL**…`)
    - `ch02-classical-classifiers/README.md`: 1 bullet violation (`- ✅ **Constraint #4 PARTIAL**…`)
    - `ch03-metrics/README.md`: 1 bullet violation (`- ✅ **Constraint #1 VALIDATED**…`)
    - `ch04-svm/README.md`: 1 bullet violation (`- ✅ **Constraint #1 IMPROVED**…`)
    - `ch05-hyperparameter-tuning/README.md`: 2 bullet violations (`- ✅ **Constraint #1 ✅**…`, `- ✅ **Constraint #2 ✅**…`)

---

## Notebook TODOs

- [ ] **Mirror README section labels** — All 5 notebooks currently use `§0`, `§1`, … headers. Add markdown cells with the exact README section names (`Running Example`, `Math`, `Step by Step`) so the notebook structure matches the README 1-to-1.
- [ ] **Add 3-row inline synthetic demo (ch01)** — Insert a code cell in `ch01-logistic-regression/notebook.ipynb` that reproduces the README's exact 3-row BCE walkthrough (`z = [2.0, −0.5, 0.8]`, labels `[1, 0, 1]`) with a deterministic seed, printing each contribution and the mean loss = 0.325.

---

## Automated checks to add

- [ ] **Emoji audit script** — Scan all sub-chapter READMEs for characters outside the approved emoji set and report offenders.
- [ ] **Notebook mirror check** — Assert that each notebook contains markdown cells with the text `Running Example`, `Math`, and `Step by Step`.
