# ML Track Content Quality Audit

**Date:** 2026-04-24  
**Auditor:** Automated content review  
**Files reviewed:** 49 sub-chapter READMEs + 9 track-level READMEs  
**Reference documents:** `AUTHORING_GUIDE.md`, `01-Regression/GRAND_CHALLENGE.md`, `validation_report.md` (prior structural audit, 2025-07-22)

> **Note on prior audit.** The `validation_report.md` (2025-07-22) identified: (a) `## §N ·` prefix in all 5 Classification chapters, and (b) unapproved `🚀` / `🤔` emoji in 15 files. Both issues are **resolved** in the current working tree — Classification headings now use `## N ·` and emoji were replaced with `⚡`. Those issues are not re-raised here.

---

## Summary

| Issue Type | Count | Severity |
|---|---|---|
| Grand Challenge / Story Arc Coherence | 9 | High |
| Pedagogical Ordering / Structure | 4 | Medium |
| Content Correctness | 5 | Medium–High |
| Content Completeness (missing numeric walkthroughs) | 2 | Medium |
| Voice & Tone (dataset policy) | 5 | Low–Medium |
| **Total** | **25** | |

---

## Issues by Track

### 03-NeuralNetworks

#### ch01-xor-problem

- **[Grand Challenge — wrong mission name]** The `## 0 · The Challenge` blockquote says `Launch **SmartVal AI**`; the Neural Networks track uses **UnifiedAI**. Every other NN chapter (ch02–ch06, ch08–ch10) correctly uses UnifiedAI.
- **[Content Correctness — Constraint #1 target]** The challenge lists `ACCURACY: <$50k MAE`, but ch03 (Backprop) — where the constraint is actually unlocked — uses `<$40k MAE` as its target in the same blockquote. The NN track's final AUTHORING_GUIDE target is `$28k MAE + 95% accuracy`. An intermediate milestone target should be stated explicitly and held consistently across all NN chapters.

#### ch02-neural-networks

- **[Grand Challenge — stale chapter references]** `What we know so far` references `Ch.1: Linear regression` and `Ch.2: Logistic regression` — these are old single-track numbers (Regression=old Ch.1; Classification=old Ch.2). In the current NN track, Ch.1 is the XOR Problem and Ch.2 is Neural Networks (this chapter). The reader following the NN-track path will not have seen those old chapters.
- **[Content Completeness — missing numeric walkthrough]** Section 3 (Math) presents only symbolic forward-pass equations (§3.1 single neuron, §3.2 two-hidden-layer pass) with no concrete digit example (e.g., feed a 3-row sample through the network and show the numeric activations at each layer). The `validation_report.md` Check 3 flagged this; it remains unresolved.

#### ch03-backprop-optimisers

- **[Grand Challenge — stale chapter references]** `What we know so far` still uses old single-track chapter numbers: `Ch.3: XOR problem` (currently NN Ch.1), `Ch.4: Neural network architecture` (currently NN Ch.2). A reader who arrived via the NN track will be confused.
- **[Content Correctness — internal target inconsistency]** The `## 0 · The Challenge` blockquote states Constraint #1 target as `<$40k MAE` (the Regression track target), while the `## 9 · Progress Check` inside the same chapter states `$48k MAE (target: <$50k)` and declares it achieved. Both the target and the achieved value are contradictory within one file.
- **[Structural — duplicate out-of-place section]** A second `## 9 · Where This Reappears` stub appears at the end of the file (after `## 10 · Bridge to Chapter 4`), repeating the same section heading that also appears earlier. One copy should be removed.

#### ch04-regularisation

- **[Grand Challenge — stale chapter reference + wrong constraint status]** `What we know so far` states `✅ Ch.5: Backprop + Adam optimizer → ✅ Constraint #1 ACHIEVED ($48k MAE)`. In the current NN track, Backprop is Ch.3 (not Ch.5). Additionally, the Progress Check in ch03 already claims Constraint #1 at `$48k MAE test` — ch04 re-claiming it via `Ch.5` is both a stale reference and a cross-chapter duplication of achievement.
- **[Content Correctness — MAE above constraint target]** The Progress Check table in this chapter records `Constraint #1 ACHIEVED — $48k MAE on training`. The challenge blockquote's stated target is `<$50k MAE`, so $48k is within target, but noting it is a *training* MAE (not test MAE) misleads about actual generalization performance.

#### ch05-cnns

- **[Grand Challenge — stale chapter references]** `What we know so far` says `Ch.1-6: Dense neural networks achieving $48k MAE`. In the current NN track there are only 10 chapters total, CNNs is Ch.5, and "dense networks" covers Ch.1–4. The description `Ch.1-6` does not map to the current NN-track numbering.
- **[Voice & Tone — dataset policy]** The Running Example uses a **synthetic 8×8 pixel grid** rather than California Housing or CelebA. The `AUTHORING_GUIDE` fingerprint specifies `dataset: california_housing_only_no_synthetic_data_except_toy_subsets`. A CNN chapter is a reasonable place to introduce image data, but the choice should either be explicitly noted as a policy exception or grounded in a real auxiliary dataset (e.g., the guide's own mention of "property condition from aerial/street-view photo grids").

#### ch06-rnns-lstms

- **[Structural — section out of order]** `## 9 · Where This Reappears` (a stub cross-reference section) appears at the very top of the file — before `## 1 · Core Idea`. The canonical template places "Where This Reappears" between the Hyperparameter Dial and the Code Skeleton (or as section 9 after "What Can Go Wrong"), not before the chapter's content sections. Move it to follow section 8.
- **[Grand Challenge — stale chapter references]** `What we know so far` says `Ch.7: CNNs for spatial data (images)`. CNNs is NN Ch.5 in the current track, not Ch.7.
- **[Voice & Tone — dataset policy]** Running Example uses a **synthetic monthly price index** dataset rather than California Housing. Same policy concern as ch05-cnns.

#### ch07-mle-loss-functions

- **[Grand Challenge — wrong mission name]** `## 0 · The Challenge` blockquote says `Launch **SmartVal AI**`. This chapter is in the Neural Networks track (mission: **UnifiedAI**). SmartVal AI is the Regression track identity.
- **[Grand Challenge — wrong constraint status claims]** States `Ch.1-14: Achieved Constraints #1, #2, #3, #4`. This is old single-track numbering (Ch.14 = Unsupervised Metrics). In the current NN track at Ch.7, Constraints #3 (MULTI-TASK) and #4 (INTERPRETABILITY) have not yet been demonstrated; those are addressed in later chapters (Ensemble track for #4, Clustering for #3). Claiming both are already achieved is factually incorrect for a reader following the NN track.

#### ch08-tensorboard

- **[Grand Challenge — stale chapter references]** States `Ch.1-15: Achieved Constraints #1-4`. TensorBoard is NN Ch.8, not Ch.16 (old single-track); the preceding chapter (MLE) is NN Ch.7. Old numbering inherited from prior single-track layout.
- **[Content Completeness — missing numeric walkthrough]** Section 3 (Math/Diagnostics) describes gradient health qualitatively and presents an incomplete table stub (column headers only, no filled rows). No concrete numeric example demonstrates the gradient diagnostic interpretation. `validation_report.md` Check 3 flagged this; it remains unresolved.

#### ch09-sequences-to-attention

- **[Grand Challenge — stale chapter references]** States `Ch.1-16: Achieved Constraints #1-4, have training instrumentation (TensorBoard)`. TensorBoard is NN Ch.8 (not Ch.16). Old single-track numbering.

#### ch10-transformers

- **[Structural — section out of order]** `## 9 · Where This Reappears` stub appears near the top of the file before `## 1 · Core Idea`, identical structural problem as ch06-rnns-lstms.

---

### 01-Regression

#### ch01-linear-regression

- **[Content Correctness — broken cross-reference]** Line 134 contains a link `../../../../MathUnderTheHood/ch05-matrices/` which resolves 4 directory levels up to the workspace root, not `notes/`. The correct depth is `../../../MathUnderTheHood/ch05-matrices/`. (Line 1027 in the same file correctly uses 3 levels, confirming the intended depth.)

#### ch02-multiple-regression

- **[Content Correctness — broken cross-references]** Lines 274 and 447 both contain `../../../../MathUnderTheHood/` links with the same extra-level error as ch01. Three total broken links across these two files (see `validation_report.md` Check 4 for exact lines).

#### ch03-feature-importance

- **[Structural — duplicate Notation block]** A standalone `## Notation` section (a full symbol table) appears between the opening blockquote and `## 0 · The Challenge`. The opening blockquote already declares notation (`ρ(xⱼ,y)`, `VIFⱼ`, etc.). The authoring guide specifies notation belongs in the opening blockquote only; the standalone section duplicates content and adds a stray heading.

---

### 02-Classification

*(All 5 chapters passed structural, story arc, and content checks. The prior § prefix and emoji issues are resolved.)*

---

### 04-RecommenderSystems

*(All 6 chapters passed all checks.)*

---

### 05-AnomalyDetection

*(All 6 chapters passed all checks.)*

---

### 06-ReinforcementLearning

*(All 6 chapters passed all checks.)*

---

### 07-UnsupervisedLearning

#### ch01-clustering, ch02-dimensionality-reduction, ch03-unsupervised-metrics

- **[Voice & Tone — dataset]** All three chapters use the **UCI Wholesale Customers dataset** (440 customers, 6 spending features) as their running example. The authoring guide fingerprint requires California Housing as the base dataset. This is a track-level dataset choice — if intentional, it should be noted in the `07-UnsupervisedLearning/README.md` as a deliberate exception (which it currently is not). The grand challenge ("SegmentAI") and dataset choice are internally consistent across all three chapters, but diverge from the track-wide California Housing convention.

---

### 08-EnsembleMethods

*(All 6 chapters passed all checks.)*

---

## Aggregate Recommendations

The following five issues affect the most chapters or carry the highest impact on reader comprehension:

### 1. Systematic stale chapter numbering in 03-NeuralNetworks `## 0 · The Challenge` (6 chapters)

Chapters ch02–ch09 in the NN track all contain "What we know so far" bullet lists that reference old single-track chapter numbers (e.g., "Ch.4: Neural network architecture", "Ch.5: Backprop", "Ch.16: TensorBoard"). These were inherited from a prior single-track layout and were never updated after reorganisation into topic-based tracks. The fix is to remap every reference in those bullets to the current NN-track chapter number (XOR=Ch.1, NN=Ch.2, Backprop=Ch.3, Regularisation=Ch.4, CNNs=Ch.5, RNNs=Ch.6, MLE=Ch.7, TensorBoard=Ch.8, Sequences=Ch.9).

### 2. Wrong grand challenge identity in two NN chapters (ch01-xor-problem, ch07-mle-loss-functions)

Both files emit "Launch **SmartVal AI**" in their challenge blockquote — the Regression track identity — when they belong to the Neural Networks track ("UnifiedAI"). For a reader following only the NN track, seeing "SmartVal AI" breaks the narrative continuity. One-line fix in each file.

### 3. Constraint #1 accuracy target inconsistency within the NN track

The NN track has two different Constraint #1 targets in circulation: `<$40k MAE` (from the Regression track, incorrectly copied into ch03), `<$50k MAE` (used by ch01–ch02 and ch04–ch10), and the AUTHORING_GUIDE track-final target of `$28k MAE + 95% accuracy`. The track should adopt one consistent intermediate milestone (e.g., `<$50k MAE`) across all NN chapters' challenge blockquotes, reserving `$28k MAE` as the final track-level goal stated in the `## 0 · The Challenge` preamble.

### 4. Out-of-order `## 9 · Where This Reappears` stubs (ch06-rnns-lstms, ch10-transformers, ch03-backprop-optimisers)

Three chapters have the "Where This Reappears" cross-link stub placed outside its canonical position. In ch06 and ch10 it appears at the top of the file before the chapter's main content; in ch03 a duplicate copy appears after the Bridge section. These stubs appear to be incomplete scaffolding left at the wrong location. They should be consolidated into the correct position (between section 8 "What Can Go Wrong" and the Progress Check) and filled in with actual forward-pointer links.

### 5. Broken MathUnderTheHood cross-references in 01-Regression ch01–ch02 (3 broken links)

Two canonical chapters — the ones the authoring guide designates as the reference standard for new chapters — contain broken links using `../../../../MathUnderTheHood/` (4 levels up from sub-chapter directory, which overshoots to the workspace root). The correct prefix is `../../../`. Because the authoring guide instructs all new chapters to be compared against ch01 and ch02 before publishing, broken links in these two files propagate risk to any author following that checklist.

---

## Clean Files (No Issues Found)

The following 28 sub-chapter READMEs passed all pedagogical ordering, grand challenge coherence, structural consistency, content correctness, and voice/tone checks:

| Track | Chapters |
|---|---|
| **01-Regression** | ch04-polynomial-features, ch05-regularization, ch06-metrics, ch07-hyperparameter-tuning |
| **02-Classification** | ch01-logistic-regression, ch02-classical-classifiers, ch03-metrics, ch04-svm, ch05-hyperparameter-tuning |
| **03-NeuralNetworks** | *(none — all 10 have at least one issue)* |
| **04-RecommenderSystems** | ch01-fundamentals, ch02-collaborative-filtering, ch03-matrix-factorization, ch04-neural-cf, ch05-hybrid-systems, ch06-cold-start-production |
| **05-AnomalyDetection** | ch01-statistical-methods, ch02-isolation-forest, ch03-autoencoders, ch04-one-class-svm, ch05-ensemble-anomaly, ch06-production |
| **06-ReinforcementLearning** | ch01-mdps, ch02-dynamic-programming, ch03-q-learning, ch04-dqn, ch05-policy-gradients, ch06-modern-rl |
| **07-UnsupervisedLearning** | *(dataset deviation noted for all 3; otherwise structurally clean)* |
| **08-EnsembleMethods** | ch01-ensembles, ch02-boosting, ch03-xgboost-lightgbm, ch04-shap, ch05-stacking, ch06-production |
