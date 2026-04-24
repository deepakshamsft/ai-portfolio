# Plan — 04-RecommenderSystems

**Audit summary:** Recommender Systems material is present but varies in depth and runnable examples; add compact matrix-factorisation numeric examples and ensure notebooks are small and deterministic.

---

## Per-chapter TODOs

- Standardise callout emoji usage to {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a 3×3 toy user×item rating matrix example showing one update step for an SVD or alternating-least-squares method.
- Verify `Notation` and `0 · The Challenge` existence and alignment to the grand challenge where applicable.
- Ensure `Animation` needle GIFs exist and match README claims.

---

## Notebook TODOs

- Mirror README examples; include a tiny reproducible training cell that runs in <2 minutes on CPU.
- Seed RNGs and reduce dataset sizes for deterministic CI runs.

---

## Sequence assessment

- Recommender content can follow core supervised topics; consider cross-linking to `UnsupervisedLearning` for embedding techniques.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Apply the small numeric examples and notebook mirroring patches, then re-run the per-chapter conformance checks.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/04-RecommenderSystems/` (fundamentals and chapter READMEs).
- Common findings:
	- Non-approved emoji usage observed (examples: 🎯, ✅). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Several chapters lack a tiny 3×3 numeric MF example in `Math` or `Step by Step`. Add a minimal SVD/ALS worked example.
	- Notebook mirroring: ensure toy datasets + seeds reproduce README examples and associated HR@k numbers.
- Recommended quick fixes: replace emojis, add small MF numeric examples, update notebooks for deterministic runs.
