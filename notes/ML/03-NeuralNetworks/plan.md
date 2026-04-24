# Plan — 03-NeuralNetworks

**Audit summary:** All chapter READMEs (Ch.3–Ch.10) follow the canonical template (`Story`, `Where you are`, `Notation`, `0 · The Challenge`, `Running Example`, `Math`, `Step by Step`, `Key Diagrams`, `Animation`). Main gaps are: (1) non-approved callout emoji usage, (2) most chapters lack a compact 3–5 row numeric verification example in the Math section, and (3) some notebooks need to be updated so they faithfully mirror README examples and numbers.

---

## Per-chapter TODOs

- **ch03-backprop-optimisers/README.md**: Replace non-approved callouts (map `🎯`→`💡`, `🚨`→`⚠️`, `✅/❌`→`⚡` or inline text); add 3–5 row numeric backward-pass worked example under `Math`; ensure notation block follows canonical list format.
- **ch04-regularisation/README.md**: Add a tiny numeric dataset showing L2 vs L1 effects (3 rows); convert callouts to approved set; ensure `Animation` file naming matches pattern `ch04-regularisation-needle.gif`.
- **ch05-cnns/README.md**: Add a runnable 8×8 synthetic-grid numeric verification cell (3 rows table + outputs); ensure `Running Example` code block is runnable and referenced by the notebook.
- **ch06-rnns-lstms/README.md**: Add short numeric BPTT example that demonstrates vanishing vs gated flow (3 rows); harmonize cross-links in `Where This Reappears` to canonical chapter paths.
- **ch07-mle-loss-functions/README.md**: Insert worked numeric derivation (Gaussian → MSE) with small sample and one-step gradient compute (3 rows); replace non-approved callouts.
- **ch08-tensorboard/README.md**: Add minimal code snippet showing `tf.keras.callbacks.TensorBoard` config for histogram/gradients; mark `write_graph`/`histogram_freq` defaults and show command to launch tensorboard.
- **ch09-sequences-to-attention/README.md**: Add a tiny numeric Q/K/V example (T=3) that computes attention weights by hand and shows the resulting context vector.
- **ch10-transformers/README.md**: Add explicit small numeric scaled-dot-product example (T=3, d_k=4) and positional-encoding matrix snippet; include an ultra-minimal training recipe (d_model=16, H=2) for reproducibility.

---

## Notebook TODOs (applies to each chapter notebook)

- **Mirror README**: Ensure the notebook mirrors README sections `Running Example`, `Math`, and `Step by Step` exactly (same equations, same numeric examples).
- **Numeric verification cells**: Add cells that compute the 3–5 row numeric verification examples from the README so readers can run and confirm the math.
- **Deterministic & fast**: Use a small dataset / fixed RNG seed so examples run in <2 minutes on CPU.
- **Instrumentation**: Where training occurs, add optional TensorBoard logging and a short note on how to view logs locally.
- **Needle animation**: Display the chapter needle GIF inline and assert the notebook plots reproduce README numbers (or explain variance if stochastic).

---

## Sequence assessment

- **Overall:** The sequence (Backprop → Regularisation → CNNs → RNNs → MLE/Loss → TensorBoard → Attention → Transformers) is pedagogically coherent for the ML track. No reordering required. Recommend a single editorial pass after the above fixes to harmonize cross-links and example numbers.

---

## Automated checks to add (follow-up work)

- **Emoji audit**: script to detect emojis outside approved set {💡, ⚠️, ⚡, 📖, ➡️} and propose replacements.
- **Section checklist**: script to confirm presence of `Notation`, `0 · The Challenge`, `Math`, `Running Example`, and `Animation` for every README.
- **Numeric walk-through detector**: simple heuristic to flag Math sections without a 3–5 row numeric table or explicit small-sample computation.
- **Notebook mirror check**: compare top-level code blocks in README vs first runnable cells in notebook.ipynb; flag differences.

---

## Next steps

- Apply the per-chapter README edits (replace emojis, add numeric examples).  
- Update each chapter notebook to mirror README examples and add TensorBoard hooks where applicable.  
- Run the automated checks and iterate on any remaining non-deterministic examples.

If you want, I can (A) apply these README edits and notebook TODOs now, or (B) produce a PR-ready patch per chapter for review. Which do you prefer?

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/03-NeuralNetworks/` (Ch.3–Ch.10).
- Common findings:
	- Non-approved emojis present across multiple chapters (examples: 🎯, ✅). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
	- Most `Math` sections lack a compact 3–5 row numeric verification example; add short numeric verification blocks so readers can run the math in the notebook.
	- Some chapter notebooks do not mirror README numeric examples exactly; add deterministic small-sample cells to each notebook.
- Recommended quick fixes: replace emojis, add numeric checks in Math subsections, and update notebooks to reproduce README numbers deterministically.
