# Plan — 01-Regression

**Audit summary:** Regression chapters mostly follow the canonical template but require small editorial fixes: replace non-approved emojis, add compact 3–5 row numeric worked examples in `Math` sections, and ensure notebooks mirror README examples exactly.

---

## Per-chapter TODOs

- Convert callouts to approved set {💡, ⚠️, ⚡, 📖, ➡️} and remove decorative emoji where they interfere with parsing.
- Add a 3–5 row numeric verification example in every `Math` subsection (OLS, closed-form updates, gradient steps, etc.).
- Ensure each README contains a `Notation` block at the top and a `0 · The Challenge` section aligned to the grand challenge.
- Verify `Animation` GIF exists and filename matches the chapter slug.

---

## Notebook TODOs

- Mirror the README `Running Example`, `Math`, and `Step by Step` cells exactly.
- Add deterministic small dataset cells (3–5 rows) reproducing numeric examples; set RNG seeds so CI is deterministic.
- Add optional TensorBoard logging where training loops appear.

---

## Sequence assessment

- Regression chapter order is pedagogically correct (linear → multiple → feature importance → polynomial → regularisation → metrics → tuning). No reordering recommended; prefer editorial harmonization.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Apply the README edits and notebook mirroring patches, then run automated checks and surface remaining ambiguous items for human review.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/01-Regression/` (chapter-level READMEs and chapter subfolders).
- Common findings:
	- Non-approved emojis found (examples: 🎯, ✅, 🚨). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Several `Math` sections lack a compact 3–5 row numeric worked example; add a short numeric verification block to each Math subsection.
	- Animation image references exist for most chapters; verify filenames match chapter slugs and `img/` paths.
- Recommended quick fixes: replace disallowed emojis, add tiny numeric examples in `Math`, and confirm notebook cells reproduce the README numeric examples.
