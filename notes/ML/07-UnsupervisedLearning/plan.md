# Plan — 07-UnsupervisedLearning

**Audit summary:** Unsupervised chapters (PCA, clustering, dimensionality reduction) are present; add numeric PCA/projector examples and ensure notebooks reproduce embeddings used in READMEs.

---

## Per-chapter TODOs

- Standardise callout emoji usage to {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a 3–5 row numeric PCA example that shows the covariance, eigenvectors, and projected coordinates by hand or in a tiny cell.
- Ensure `Notation` and `0 · The Challenge` appear and link to downstream topics (e.g., `ML/AI/Multimodal` where embeddings are reused).

---

## Notebook TODOs

- Mirror README math and add deterministic small-data embedding demos and exporter-friendly artifacts for the embedding projector.
- Keep examples tiny and reproducible on CPU.

---

## Sequence assessment

- Unsupervised modules serve as background for many later chapters; sequence is appropriate.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Add numeric PCA/clustering examples, harmonise notebooks, and run the automated checks.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/07-UnsupervisedLearning/`.
- Common findings:
	- Non-approved emojis present in some chapters (examples: 🎯). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- PCA and clustering math sections would benefit from a 3–5 row numeric PCA covariance/eigenvector example and a tiny clustering distance computation example.
	- Ensure notebooks reproduce embedding/projector outputs deterministically with small datasets and fixed RNG seeds.
- Recommended quick fixes: replace emojis, add numeric PCA/clustering examples, and update notebooks for deterministic runs.
