# Plan — 04-RecommenderSystems (remaining work)

**Last audited:** 2026-04-24. Three of nine todos are complete (3×3 MF example, Notation/Challenge sections, Animation GIFs). Six remain.

---

## Per-chapter TODOs

- [ ] **Standardise emoji usage to approved set `{💡, ⚠️, ⚡, 📖, ➡️}`.**
  `✅` and `❌` appear in bullet lists and constraint tables across all six chapter READMEs.
  Replace with approved emoji or plain text (`Done` / `Not started`).

---

## Notebook TODOs

- [ ] **Mirror README examples in each chapter notebook.**
  Each `notebook.ipynb` should include a tiny reproducible training cell that runs in <2 minutes on CPU and produces the same HR@k values cited in the README.

- [ ] **Seed RNGs and reduce dataset sizes for deterministic CI runs.**
  Add `np.random.seed(42)` (and framework equivalents) and cap datasets to a small reproducible subset so CI passes consistently.

---

## Sequence assessment

- [ ] **Add cross-links from embedding-heavy chapters (ch03, ch04) to `07-UnsupervisedLearning`.**
  No cross-links found in any chapter README. Linking to the unsupervised track for embedding background would strengthen the narrative.

---

## Automated checks to add

- [ ] **Create scripts for:** emoji audit, section checklist, numeric-walkthrough detector, notebook mirror check.
  None of these exist in `scripts/` yet.

---

## Next steps

- [ ] **Apply notebook mirroring and RNG-seed patches, then re-run per-chapter conformance checks.**
  The 3×3 numeric examples are in the READMEs; notebooks still need matching toy-data cells.
