# Plan — 03-NeuralNetworks

Audit performed 2026-04-24. Re-audited 2026-04-24. 12 README todos completed; 5 README todos from the re-audit now also complete (Fixes 1–5 applied 2026-04-24). Notebook todos and automated-check scripts are all still pending.

---

## Completed README TODOs (verified 2026-04-24)

- [x] **ch03-backprop-optimisers/README.md**: Replaced `✅`/`❌` bullet callout prefixes throughout (lines ~17–22, 658–685) and removed `🎯` from LR-range Mermaid diagram node label.
- [x] **ch04-regularisation/README.md**: Replaced `✅`/`❌` bullet callout prefixes throughout (lines ~21–24, 341–432).
- [x] **ch06-rnns-lstms/README.md**: Moved misplaced `## 9 · Where This Reappears` stub from before `## 1 · Core Idea` to after `## 8 · What Can Go Wrong` (canonical position). Cross-link stub retained as-is — concrete link harmonisation is a follow-up.
- [x] **ch07-mle-loss-functions/README.md**: Replaced `✅` bullet callout prefixes (lines ~21–22).
- [x] **ch10-transformers/README.md**: Moved misplaced `## 9 · Where This Reappears` stub from before `## 1 · Core Idea` to after `## 8 · What Can Go Wrong` (canonical position). Minimal training recipe (d_model=16, H=2) is still pending.

---

## Remaining README TODOs

- [ ] **ch10-transformers/README.md**: Add ultra-minimal training recipe with `d_model=16, H=2` for reproducibility — the current `build_tabular_transformer()` default uses `d_model=32, num_heads=4`; add a commented minimal-config call or a dedicated "Quick reproducibility recipe" code block.
- [ ] **ch06-rnns-lstms/README.md**: Harmonize concrete cross-links in `Where This Reappears` (§9) — replace stub text with specific links to ch09-sequences-to-attention, ch10-transformers, and MultimodalAI/AIInfrastructure notes.

---

## Notebook TODOs (applies to each chapter notebook, ch03–ch10)

- [ ] **Mirror README**: Ensure the notebook mirrors README sections `Running Example`, `Math`, and `Step by Step` exactly (same equations, same numeric examples).
- [ ] **Numeric verification cells**: Add cells that compute the 3–5 row numeric verification examples from the README so readers can run and confirm the math.
- [ ] **Deterministic & fast**: Use a small dataset / fixed RNG seed so examples run in <2 minutes on CPU.
- [ ] **Instrumentation**: Where training occurs, add optional TensorBoard logging and a short note on how to view logs locally.
- [ ] **Needle animation**: Display the chapter needle GIF inline and assert the notebook plots reproduce README numbers (or explain variance if stochastic).

---

## Automated checks to add (follow-up scripts)

- [ ] **Emoji audit**: script to detect emojis outside approved set {💡, ⚠️, ⚡, 📖, ➡️} and propose replacements.
- [ ] **Section checklist**: script to confirm presence of `Notation`, `0 · The Challenge`, `Math`, `Running Example`, and `Animation` for every README.
- [ ] **Numeric walk-through detector**: simple heuristic to flag Math sections without a 3–5 row numeric table or explicit small-sample computation.
- [ ] **Notebook mirror check**: compare top-level code blocks in README vs first runnable cells in notebook.ipynb; flag differences.
