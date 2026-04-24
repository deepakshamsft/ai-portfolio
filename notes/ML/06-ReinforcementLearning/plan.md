# Plan — 06-ReinforcementLearning

**Audit summary:** Reinforcement Learning chapters introduce key algorithms but need compact MDP examples and deterministic toy environments for notebooks so readers can run experiments quickly.

---

## Per-chapter TODOs

- Replace non-approved emojis with {💡, ⚠️, ⚡, 📖, ➡️}.
- Add a tiny MDP worked example (3 states, 2 actions) showing value update / TD(0) step or a single policy-gradient update computed by hand.
- Ensure `Notation` block and `0 · The Challenge` are present and map to learning objectives.

---

## Notebook TODOs

- Provide deterministic toy environments (small gridworld) with fixed RNG and shallow episode counts so examples run locally in <2 minutes.
- Mirror README math and show the same numerical step-by-step outputs.

---

## Sequence assessment

- RL is properly placed after supervised and unsupervised primitives; consider explicit cross-links to `TensorBoard` for instrumentation guidance.

---

## Automated checks to add

- Emoji audit; Section checklist; Numeric walkthrough detector; Notebook mirror check.

---

## Next steps

- Add the MDP numeric examples, adjust notebooks for deterministic runs, and run the conformance checks.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/06-ReinforcementLearning/`.
- Common findings:
	- Non-approved emoji usage (examples: 🎯, ✅). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Many chapters contain good conceptual math but lack a compact 3-state, 2-action numeric MDP example to illustrate updates; add a short worked example in `Math`.
	- Ensure notebooks include deterministic tiny gridworlds or Gym wrappers with fixed seeds so readers can reproduce step-by-step updates.
- Recommended quick fixes: replace emojis, add MDP numeric examples, and ensure notebooks run deterministically with small environments.
