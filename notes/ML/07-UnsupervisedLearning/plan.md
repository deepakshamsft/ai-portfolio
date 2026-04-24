# Plan — 07-UnsupervisedLearning

Remaining work after audit (2026-04-24, re-verified 2026-04-24). Completed: `Notation` and `0 · The Challenge` sections are present in all sub-chapters with downstream links.

---

## Per-chapter TODOs

- [ ] **Emoji standardisation** — Replace `❌` used as callout bullets with approved emojis {💡, ⚠️, ⚡, 📖, ➡️}. Affected files:
  - `ch01-clustering/README.md` line ~19 (`- ❌ **No labels!**`)
  - `ch02-dimensionality-reduction/README.md` lines ~19–20 (`- ❌ **Can't visualise…**`, `- ❌ **Silhouette only…**`)

- [ ] **Numeric PCA walkthrough** — Add a 3–5 row hand-computed example to `ch02-dimensionality-reduction/README.md` §3.1 that shows: raw data → centred matrix → covariance matrix → eigenvectors → projected coordinates. The current section has symbolic steps and an EVR table but no concrete number walkthrough.

---

## Notebook TODOs

- [ ] **Mirror README math** — Ensure each chapter notebook reproduces the key math examples from the README and adds deterministic small-data embedding demos with exporter-friendly artifacts for the embedding projector.
- [ ] **CPU-reproducible examples** — Keep all notebook examples tiny and reproducible on CPU (small fixed datasets, `random_state=42` everywhere).

---

## Automated checks to add

- [ ] Emoji audit script; Section checklist; Numeric walkthrough detector; Notebook mirror check.
