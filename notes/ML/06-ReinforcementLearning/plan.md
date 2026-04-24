# Plan — 06-ReinforcementLearning

**Audit summary (2026-04-24):** Notation blocks and `0 · The Challenge` sections are present and correct in all 6 chapters. Compact 3-state MDP examples added to ch01 and ch02. Remaining work: emoji cleanup, 3-state examples for ch03–ch06, notebook determinism, and TensorBoard cross-links.

---

## Per-chapter TODOs

- [ ] **All chapters (ch01–ch06):** Replace non-approved emojis with {💡, ⚠️, ⚡, 📖, ➡️}. Currently ❌ and ✅ appear in constraint-status tables and bullet points across all chapters — neither is in the approved set.
- [ ] **ch03–ch06:** Add a compact 3-state, 2-action worked example in the `Math` section (ch01 and ch02 already have one). Show a single TD(0) step (ch03), a DQN target computation (ch04), a policy-gradient update (ch05), or a PPO clip calculation (ch06) on the toy {s0, s1, s2} MDP.

---

## Notebook TODOs

- [ ] Provide deterministic toy environments (small gridworld or 3-state chain) with fixed RNG seeds and shallow episode counts so examples run locally in <2 minutes.
- [ ] Mirror README math: show the same numerical step-by-step outputs used in each chapter's `Math` section inside the corresponding notebook.

---

## Sequence assessment

- [ ] Add explicit cross-links to `TensorBoard` for instrumentation guidance (no chapter currently references it).

---

## Automated checks to add

- [ ] Emoji audit script; Section checklist; Numeric walkthrough detector; Notebook mirror check.
