# Plan — 02-Classification

**Audit summary:** Classification chapters use the canonical structure but need compact numeric derivations (e.g., small logistic-regression example), stricter notation blocks, and notebook mirroring to ensure runnable examples reproduce README numbers.

---

## Per-chapter TODOs

- Replace non-approved callout emojis with the approved set {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a 3–5 row numeric verification example for logistic loss, decision boundary, and calibration checks.
- Ensure `Notation`, `0 · The Challenge`, and `Running Example` sections are present and consistent across chapters.
- Confirm `Animation` assets exist and are correctly named.

---

## Notebook TODOs

- Mirror the README `Running Example`, `Math`, and `Step by Step` cells exactly and add deterministic seeds.
- Provide a tiny synthetic dataset demonstration (3–5 rows) that reproduces the decision boundary and logits in the README.

---

## Sequence assessment

- Classification should remain after Regression in the track; no structural reorder required.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Apply README and notebook edits for each chapter, run automated checks, and escalate ambiguous items for human review.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/02-Classification/`.
- Common findings:
	- Non-approved emojis present (examples: 🎯, ✅). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Some `Math` sections (logistic loss, ROC/PR derivations) lack a compact 3–5 row numeric worked example; add a short numeric verification block.
	- Ensure `Notation` block appears at chapter top for consistent symbol definitions.
- Recommended quick fixes: replace emojis, add numeric verification blocks, and ensure notebooks reproduce README numbers.
