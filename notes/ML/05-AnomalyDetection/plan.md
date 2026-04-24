# Plan — 05-AnomalyDetection

## Remaining TODOs

### Per-chapter
- [ ] Normalize callout emoji usage to the approved set {💡, ⚠️, ⚡, 📖, ➡️}.
  - `❌` is used as a status indicator in all 6 chapter challenge sections — replace with ⚠️ or remove.
  - `🚨` and `✅` appear in mermaid node labels throughout all chapters — replace with text labels or approved emoji.

### Notebooks
- [ ] Document evaluation metrics (precision@k, ROC AUC) with small numeric examples in READMEs and/or notebooks.
  - `precision@k` is absent everywhere — no worked numeric table exists for it.
  - ROC AUC is computed in code cells but no prose-level small numeric example has been written.

### Cross-links
- [ ] Add cross-links to `../07-UnsupervisedLearning/` where relevant (no Time Series track exists in this repo).
  - Clustering-based anomaly detection (DBSCAN, GMM) is covered in ch07 — add ➡️ see-also links in ch01 and/or ch05.

### Automated checks
- [ ] Add automated check scripts: emoji audit, section checklist, numeric walkthrough detector, notebook mirror check.
  - No such scripts exist in `scripts/` for Anomaly Detection chapter validation.
