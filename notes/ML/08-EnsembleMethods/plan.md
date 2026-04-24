# Plan — 08-EnsembleMethods

**Audit (2026-04-24):** Per-chapter README work is complete. Remaining work is notebooks, cross-links, and automated tooling.

---

## Remaining TODOs

- [ ] **Notebooks**: Mirror README worked comparisons in each chapter notebook; add deterministic tiny datasets (`random_state=42`) and a short `# ~N sec on CPU` runtime budget comment so examples run quickly without a GPU.

- [ ] **Cross-links**: Add "See also" links to the `Metrics` and `Hyperparameter Tuning` chapters at the bottom of the relevant chapter READMEs (at minimum ch01, ch02, ch03).

- [ ] **Automated checks**: Add scripts for emoji audit, section-presence checklist, numeric-walkthrough detector, and notebook-mirror check under `scripts/` (or extend existing `check_notebooks.py`).
