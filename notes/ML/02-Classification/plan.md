# Plan — 02-Classification (remaining TODOs only)

> Verified complete: emoji cleanup (April 24, 2026)

---

## Notebook TODOs

- [ ] **Mirror README section labels** — All 5 notebooks currently use `§0`, `§1`, … headers. Add markdown cells with the exact README section names (`Running Example`, `Math`, `Step by Step`) so the notebook structure matches the README 1-to-1.
- [ ] **Add 3-row inline synthetic demo (ch01)** — Insert a code cell in `ch01-logistic-regression/notebook.ipynb` that reproduces the README's exact 3-row BCE walkthrough (`z = [2.0, −0.5, 0.8]`, labels `[1, 0, 1]`) with a deterministic seed, printing each contribution and the mean loss = 0.325.

---

## Automated checks to add

- [ ] **Emoji audit script** — Scan all sub-chapter READMEs for characters outside the approved emoji set and report offenders.
- [ ] **Notebook mirror check** — Assert that each notebook contains markdown cells with the text `Running Example`, `Math`, and `Step by Step`.
