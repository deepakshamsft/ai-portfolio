# Plan — 04-RecommenderSystems (remaining work)

**Last audited:** 2026-04-24. Emoji standardisation and UnsupervisedLearning cross-links completed.

---

## Notebook TODOs

- [ ] **Mirror README examples in each chapter notebook.**
  Each `notebook.ipynb` should include a tiny reproducible training cell that runs in <2 minutes on CPU and produces the same HR@k values cited in the README.

- [ ] **Seed RNGs and reduce dataset sizes for deterministic CI runs.**
  Add `np.random.seed(42)` (and framework equivalents) and cap datasets to a small reproducible subset so CI passes consistently.

---

## Automated checks to add

- [ ] **Create scripts for:** emoji audit, section checklist, numeric-walkthrough detector, notebook mirror check.
  None of these exist in `scripts/` yet.

---

## Next steps

- [ ] **Apply notebook mirroring and RNG-seed patches, then re-run per-chapter conformance checks.**
  The 3×3 numeric examples are in the READMEs; notebooks still need matching toy-data cells.
