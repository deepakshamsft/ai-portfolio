# Plan — 08-EnsembleMethods

**Audit summary:** Ensemble chapters exist but often lack tiny worked examples comparing bagging vs boosting; add compact numeric examples and ensure notebooks reproduce comparisons deterministically.

---

## Per-chapter TODOs

- Replace non-approved emojis with {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a 3–5 row numeric example contrasting bagging and boosting updates (simple decision stump votes and one boosting weight update).
- Ensure `Notation`, `0 · The Challenge`, and `Animation` are present and consistent.

---

## Notebook TODOs

- Mirror README worked comparisons; add deterministic tiny datasets and a short runtime budget comment so examples run quickly on CPU.

---

## Sequence assessment

- Ensemble methods appropriately follow core supervised material; cross-link to `Metrics` and `Hyperparameter Tuning` chapters.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Add numeric ensemble examples, update notebooks, and run the automated conformance checks.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/08-EnsembleMethods/`.
- Common findings:
	- Non-approved emojis observed (examples: 🎯). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Lack of compact numeric examples contrasting bagging vs boosting; add a 3–5 row worked example showing stump votes and one boosting weight update.
	- Ensure notebooks include tiny deterministic datasets so comparisons (bagging vs boosting) reproduce README numbers.
- Recommended quick fixes: replace emojis, add the numeric ensemble example, and update notebooks for deterministic runs.
