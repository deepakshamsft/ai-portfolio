# Plan — 05-AnomalyDetection

**Audit summary:** Anomaly detection chapters are present but inconsistent in runnable examples; add concise numeric examples (small time-series or feature vectors) and enforce notebook mirroring.

---

## Per-chapter TODOs

- Normalize callout emoji usage to the approved set {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a 3–5 row example demonstrating an anomaly score computation (e.g., z-score, Isolation Forest decision path) and a single update/decision step.
- Ensure `Notation` and `0 · The Challenge` sections are present and aligned to learning objectives.

---

## Notebook TODOs

- Mirror README math and examples with deterministic seeds; provide a tiny, runnable anomaly detection demo.
- Document evaluation metrics (precision@k, ROC AUC) with small numeric examples.

---

## Sequence assessment

- Placement is flexible; cross-link to `Time Series` or `UnsupervisedLearning` chapters where relevant.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Add numeric worked examples, update notebooks, then run the automated checks and surface unresolved items.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/05-AnomalyDetection/`.
- Common findings:
	- Non-approved emojis found (examples: 🚨, 🎯). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Several `Math` sections (Mahalanobis, Z-score) would benefit from a 3–5 row numeric worked example demonstrating scoring and threshold selection.
	- Notebook mirroring: add deterministic tiny datasets and ensure the ROC/recall numbers in README match notebook outputs.
- Recommended quick fixes: replace emojis, add numeric verification examples, and update notebooks to include deterministic scoring cells.
