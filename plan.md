# ML Track Content Enhancement — Phased Execution Plan

> **Status:** Ready for execution  
> **Last updated:** 2026-04-24  
> **Owner:** Repository maintainer  
> **Execution mode:** Subagent-driven with human review gates

---

## Executive Summary

This plan coordinates the execution of **9 staged `plan.md` files** across the ML track that will:
- Standardize emoji/callout usage to approved set {💡, ⚠️, ⚡, 📖, ➡️}
- Add 3–5 row numeric verification examples to Math sections
- Ensure notebooks mirror README examples deterministically
- Expand feature engineering chapter (Ch.03) with new content sections

**Total scope:** 8 chapter-level audits + 1 detailed architectural build  
**Estimated effort:** 12–18 hours total (3–5 hours per phase)  
**Completion criteria:** All TODOs in plan.md files resolved; automated conformance checks pass

---

## Prerequisites & Permissions Verification

### ✅ Repository Access
- [x] Write access to `c:\repos\ai-portfolio` confirmed (git status shows untracked plan.md files)
- [x] Working directory is clean except for staged plan.md files
- [x] PowerShell terminal access available

### ✅ Required Tools
- [x] Python environment with sklearn, numpy, pandas, matplotlib
- [x] Jupyter notebook kernel configured
- [x] Git version control active
- [ ] Optional: SHAP library (`pip install shap`) for Phase 0 only

### ✅ File Structure
- [x] All target chapter READMEs exist under `notes/ML/`
- [x] All plan.md files present and readable
- [x] `gen_scripts/` folder exists for animation generators

### ⚠️ Human Review Gates
Each phase requires **manual review and approval** before proceeding to the next phase. This prevents cascading errors and maintains content quality.

---

## Phase Architecture

```
Phase 0 (Solo)        — Feature Engineering Deep Build (ch03-feature-importance)
         ↓
Phase 1 (Batch A)     — Regression + Classification + Neural Networks
         ↓
Phase 2 (Batch B)     — Recommender + Anomaly Detection + Reinforcement Learning
         ↓
Phase 3 (Batch C)     — Unsupervised Learning + Ensemble Methods
         ↓
Phase 4 (Validation)  — Automated conformance checks + final QA
```

**Critical constraint:** Phases must execute sequentially (no parallel phase execution) because Phase 0 establishes patterns that inform later phases.

---

## Phase 0 — Feature Engineering Deep Build

**Duration:** 4–6 hours  
**Complexity:** High (architectural + content generation)  
**Recommended LLM:** Claude Sonnet 4.5 (reasoning-heavy, long-context edits)  
**Alternative:** GPT-4 Turbo (if Claude unavailable)

### Scope

Execute the detailed architectural plan in:
- `notes/ML/01-Regression/ch03-feature-importance/plan.md`

This phase is **qualitatively different** from Phases 1–3 because it:
- Adds entirely new README sections (Encoding, Missing Data, Interactions, Selection, PCA)
- Requires alignment audit to scope out-of-range content
- Generates 7 new visualizations via gen_scripts
- Adds 11 new numeric walkthroughs
- Extends notebook with 8+ new cells

### Key Deliverables

1. **README edits:**
   - Expand Section 2 (Scaling) with log/Box-Cox transforms + weight standardization
   - Add Section 6 variance threshold precondition before multicollinearity
   - Add Section 7 filter selection methods (Pearson, Spearman, MI)
   - Add Section 8 SHAP brief treatment (waterfall + beeswarm)
   - Bridge sections for interactions (→Ch.4) and Lasso (→Ch.5)

2. **Visualizations (gen_scripts):**
   - `gen_ch03_log_transform.py` → `img/ch03-log-transform.png`
   - `gen_ch03_pearson_vs_mi.py` → `img/ch03-pearson-vs-mi.png`
   - `gen_ch03_lasso_path.py` → `img/ch03-lasso-path.png`
   - `gen_ch03_shap.py` → waterfall + beeswarm plots
   - 3 additional pipeline/PCA diagrams

3. **Notebook extensions:**
   - Log transform cells (after existing scaling section)
   - Variance threshold filter cell
   - Pearson/MI comparison cells
   - SHAP waterfall/beeswarm cells

4. **Numeric walkthroughs (11 new):**
   - Log transform by hand (5 values)
   - Variance threshold computation
   - Pearson correlation by hand (5 points)
   - Mutual information (2×2 contingency table)
   - 2-feature Shapley value derivation
   - PCA projection (4 points in 2D)
   - 5 additional walkthroughs per plan.md TODO list

### Execution Strategy

**Step 1:** Read the alignment audit section (lines 1–150 of plan.md)  
**Step 2:** Implement only ✅ approved TODOs; skip ❌ out-of-scope items  
**Step 3:** Generate visualizations in sequence (each gen script depends on previous)  
**Step 4:** Update README with new content sections  
**Step 5:** Add notebook cells mirroring new README sections  
**Step 6:** Run notebook end-to-end to verify deterministic outputs

### Review Criteria

- [ ] All ✅ approved TODOs from plan.md completed
- [ ] All ❌ out-of-scope items (categoricals, missing data for CA Housing, full PCA derivation) **not** implemented
- [ ] 7 new images generated and display correctly with dark background
- [ ] Notebook executes in <5 minutes on CPU without errors
- [ ] All numeric walkthroughs match notebook cell outputs exactly

### Why This Phase is Solo

The architectural complexity and scope alignment decisions require sustained context and judgment. Batch processing would risk implementing out-of-scope content.

---

## Phase 1 — Batch A: Core Supervised Learning

**Duration:** 3–4 hours  
**Complexity:** Low-Medium (uniform editorial changes)  
**Recommended LLM:** GPT-4o or Claude Sonnet 3.5 (fast, parallel-capable)  
**Alternative:** Claude Sonnet 4.5 (if you want consistency with Phase 0)

### Scope

Execute editorial plans for:
1. `notes/ML/01-Regression/plan.md`
2. `notes/ML/02-Classification/plan.md`
3. `notes/ML/03-NeuralNetworks/plan.md`

All three plans follow the same pattern:
- Replace non-approved emojis (🎯→💡, ✅→⚡, 🚨→⚠️)
- Add 3–5 row numeric verification examples to Math sections lacking them
- Ensure notebooks mirror README examples with deterministic seeds
- Verify Animation GIFs exist and match README references

### Parallelization Strategy

Within Phase 1, the three chapter sets can be processed in parallel:
1. Subagent A: Regression chapters
2. Subagent B: Classification chapters
3. Subagent C: Neural Networks chapters

All three subagents execute simultaneously; results merge after completion.

### Key Deliverables (per chapter set)

**Regression:**
- Emoji replacements in ch01, ch02, ch03 (ch03 already done in Phase 0)
- Numeric examples for gradient descent steps, closed-form OLS
- Notebook seed verification

**Classification:**
- Emoji replacements in classification chapters
- Numeric logistic loss example (3–5 rows)
- ROC/PR curve numeric verification

**Neural Networks (Ch.3–Ch.10):**
- Emoji replacements across 8 sub-chapters
- Numeric backprop example (3 rows, one backward pass)
- L2 vs L1 regularization numeric demo (3 rows)
- 8×8 CNN synthetic grid numeric verification
- BPTT example (3 timesteps)
- Q/K/V attention numeric example (T=3)
- Scaled dot-product attention (T=3, d_k=4)

### Review Criteria

- [ ] All emojis standardized to approved set {💡, ⚠️, ⚡, 📖, ➡️}
- [ ] Every Math section contains at least one 3–5 row numeric example
- [ ] All notebooks execute deterministically (same outputs on repeated runs)
- [ ] Cross-references to other chapters use correct canonical paths
- [ ] Animation GIFs load correctly in README previews

---

## Phase 2 — Batch B: Specialized Methods

**Duration:** 2–3 hours  
**Complexity:** Low-Medium  
**Recommended LLM:** GPT-4o (fastest for straightforward edits)  
**Alternative:** Claude Sonnet 3.5

### Scope

Execute editorial plans for:
1. `notes/ML/04-RecommenderSystems/plan.md`
2. `notes/ML/05-AnomalyDetection/plan.md`
3. `notes/ML/06-ReinforcementLearning/plan.md`

Same editorial pattern as Phase 1, with domain-specific numeric examples:

**Recommender Systems:**
- 3×3 user×item rating matrix example
- SVD or ALS one-step update by hand

**Anomaly Detection:**
- Z-score anomaly computation (5 values)
- Isolation Forest decision path (3 rows)

**Reinforcement Learning:**
- Tiny MDP example (3 states, 2 actions)
- TD(0) value update by hand
- Single policy-gradient step

### Parallelization Strategy

Same as Phase 1 — three subagents process chapters in parallel.

### Review Criteria

- [ ] Domain-specific numeric examples added (MF update, anomaly score, MDP step)
- [ ] Notebooks use toy environments (small gridworld for RL, tiny rating matrix for RecSys)
- [ ] All notebooks run in <2 minutes on CPU
- [ ] Emoji standardization complete

---

## Phase 3 — Batch C: Unsupervised & Ensembles

**Duration:** 2–3 hours  
**Complexity:** Low  
**Recommended LLM:** GPT-4o or GPT-4o-mini (simplest edits)  
**Alternative:** Claude Sonnet 3.5

### Scope

Execute editorial plans for:
1. `notes/ML/07-UnsupervisedLearning/plan.md`
2. `notes/ML/08-EnsembleMethods/plan.md`

**Unsupervised Learning:**
- 3–5 row PCA covariance/eigenvector example
- K-means distance computation numeric walkthrough
- Embedding projector output artifacts

**Ensemble Methods:**
- 3–5 row bagging vs boosting comparison
- Decision stump vote aggregation
- One boosting weight update by hand

### Parallelization Strategy

Two subagents process chapters in parallel.

### Review Criteria

- [ ] PCA numeric example added (covariance matrix by hand)
- [ ] Bagging vs boosting comparison numeric walkthrough
- [ ] Embedding projector outputs reproducible
- [ ] All edits maintain voice consistency with Ch.01/Ch.02 canonical examples

---

## Phase 4 — Validation & Conformance

**Duration:** 1–2 hours  
**Complexity:** Low (automated checks + manual spot-checks)  
**Recommended LLM:** Any (or run scripts directly without LLM)

### Scope

Run automated conformance checks mentioned in all plan.md files:

1. **Emoji audit script**
   - Scan all README files for emojis outside approved set
   - Flag violations with file path + line number

2. **Section checklist validator**
   - Verify presence of required sections: `Notation`, `0 · The Challenge`, `Math`, `Running Example`, `Animation`
   - Flag missing sections

3. **Numeric walkthrough detector**
   - Heuristic: flag Math sections without a table or explicit small-sample computation
   - Manual review flagged sections

4. **Notebook mirror check**
   - Compare top-level code blocks in README vs first runnable cells in notebook
   - Flag differences exceeding 20% edit distance

### Key Deliverables

1. **Validation report:** `validation_report.md` listing:
   - Pass/fail per chapter per check
   - List of remaining violations (if any)
   - Manual review items

2. **Final QA checklist:**
   - [ ] All notebooks execute without errors
   - [ ] All images load correctly
   - [ ] All cross-references resolve
   - [ ] Math notation consistent across chapters
   - [ ] Voice and tone match canonical Ch.01/Ch.02 examples

3. **Git commit strategy:**
   - One commit per phase with descriptive message
   - Tag final state as `ml-track-editorial-v1.0`

---

## LLM Recommendations by Phase

Based on task characteristics:

| Phase | Primary Recommendation | Rationale | Alternative |
|-------|------------------------|-----------|-------------|
| **Phase 0** | **Claude Sonnet 4.5** | Long-context architectural decisions; 400+ line plan.md; complex scope alignment audit | GPT-4 Turbo (if Claude rate-limited) |
| **Phase 1** | **GPT-4o** | Fast parallel edits; uniform pattern across 3 chapter sets; lower token cost | Claude Sonnet 3.5 (for consistency) |
| **Phase 2** | **GPT-4o** | Straightforward domain-specific examples; fast turnaround | GPT-4o-mini (cost-optimized) |
| **Phase 3** | **GPT-4o-mini** | Simplest edits (2 chapter sets only); lowest complexity | GPT-4o (if budget not a concern) |
| **Phase 4** | **Any / No LLM** | Automated scripts; minimal reasoning required | Run validation scripts directly in terminal |

### Model Selection Guidance

**Use Claude Sonnet 4.5 if:**
- You need sustained architectural reasoning (Phase 0)
- Content quality and consistency are top priority
- You have access and rate limits permit

**Use GPT-4o if:**
- You want fastest execution for editorial changes (Phases 1–3)
- You're processing multiple chapters in parallel
- Cost efficiency matters

**Use GPT-4o-mini if:**
- Edits are purely mechanical (emoji replacement, seed setting)
- You want to minimize API costs
- Tasks are well-defined with no ambiguity

---

## Execution Checklist

### Before Starting
- [ ] Read this entire plan.md document
- [ ] Verify all prerequisites in "Prerequisites & Permissions Verification" section
- [ ] Create a branch: `git checkout -b ml-track-editorial-phase-0`
- [ ] Back up current state: `git stash push -u -m "pre-phase-0-backup"`

### Phase 0
- [ ] Read `notes/ML/01-Regression/ch03-feature-importance/plan.md` fully
- [ ] Execute alignment audit (lines 1–150)
- [ ] Implement ✅ approved TODOs only
- [ ] Generate all visualizations (7 new images)
- [ ] Add notebook cells
- [ ] Test notebook end-to-end
- [ ] Commit: `git commit -am "Phase 0: Feature engineering deep build complete"`
- [ ] **HUMAN REVIEW GATE** — verify outputs before proceeding

### Phase 1
- [ ] Launch 3 parallel subagents for Regression/Classification/NeuralNetworks
- [ ] Merge results
- [ ] Spot-check 2 chapters per set for quality
- [ ] Commit: `git commit -am "Phase 1: Batch A editorial complete (Regression/Classification/NN)"`
- [ ] **HUMAN REVIEW GATE**

### Phase 2
- [ ] Launch 3 parallel subagents for RecSys/Anomaly/RL
- [ ] Merge results
- [ ] Verify domain-specific examples work correctly
- [ ] Commit: `git commit -am "Phase 2: Batch B editorial complete (RecSys/Anomaly/RL)"`
- [ ] **HUMAN REVIEW GATE**

### Phase 3
- [ ] Launch 2 parallel subagents for Unsupervised/Ensembles
- [ ] Merge results
- [ ] Commit: `git commit -am "Phase 3: Batch C editorial complete (Unsupervised/Ensembles)"`
- [ ] **HUMAN REVIEW GATE**

### Phase 4
- [ ] Run automated emoji audit script
- [ ] Run section checklist validator
- [ ] Run numeric walkthrough detector
- [ ] Run notebook mirror check
- [ ] Generate validation report
- [ ] Fix any remaining violations
- [ ] Final commit: `git commit -am "Phase 4: Validation complete; all checks pass"`
- [ ] Tag: `git tag ml-track-editorial-v1.0`
- [ ] Push: `git push origin ml-track-editorial-phase-0 --tags`

---

## ⚠️ Post-Mortem: Phase 1–4 Execution Review (2026-04-24)

> **Reviewed by:** Human + Copilot audit  
> **Status:** Multiple critical issues found. Corrective pass required before proceeding to Phase 3.  
> **Remediation applied:** 5 track-level READMEs restored from git (`git checkout --`).

---

### What Was Done Correctly

1. `notes/ML/01-Regression/ch01-linear-regression/README.md` — correct approach: content **inserted into** the existing file without overwriting. Adds the Normal Equation section, OLS vs gradient descent comparison table, and learning-rate failure modes. **Keep this change.**
2. `notes/ML/AUTHORING_GUIDE.md` — adds a "Style Ground Truth" section and LLM-STYLE-FINGERPRINT comment derived from canonical Ch.01/Ch.02. **Keep this change.**
3. All `plan.md` files for ch01–ch08 and root `plan.md` — correctly created as new untracked files. No existing content was damaged.

---

### Critical Issues Found

#### Issue 1 — Wrong Target Files (HIGH SEVERITY)

The executing LLM edited **track-level** READMEs (e.g., `03-NeuralNetworks/README.md`) instead of the **sub-chapter** READMEs (e.g., `ch03-backprop-optimisers/README.md`). Track-level READMEs are **navigation/overview** documents. The `plan.md` files for each chapter explicitly list **sub-chapter files** as targets.

**Files that should have been edited but were not touched:**

```
# Phase 1 — Regression (ch01–ch07 sub-chapters)
notes/ML/01-Regression/ch01-linear-regression/README.md  ← DONE (1 of 7)
notes/ML/01-Regression/ch02-multiple-regression/README.md
notes/ML/01-Regression/ch04-polynomial-features/README.md
notes/ML/01-Regression/ch05-regularization/README.md
notes/ML/01-Regression/ch06-metrics/README.md
notes/ML/01-Regression/ch07-hyperparameter-tuning/README.md

# Phase 1 — Classification (all sub-chapters)
notes/ML/02-Classification/ch01-logistic-regression/README.md
notes/ML/02-Classification/ch02-classical-classifiers/README.md
notes/ML/02-Classification/ch03-metrics/README.md
notes/ML/02-Classification/ch04-svm/README.md
notes/ML/02-Classification/ch05-hyperparameter-tuning/README.md

# Phase 1 — Neural Networks (ch03–ch10 sub-chapters)
notes/ML/03-NeuralNetworks/ch03-backprop-optimisers/README.md
notes/ML/03-NeuralNetworks/ch04-regularisation/README.md
notes/ML/03-NeuralNetworks/ch05-cnns/README.md
notes/ML/03-NeuralNetworks/ch06-rnns-lstms/README.md
notes/ML/03-NeuralNetworks/ch07-mle-loss-functions/README.md
notes/ML/03-NeuralNetworks/ch08-tensorboard/README.md
notes/ML/03-NeuralNetworks/ch09-sequences-to-attention/README.md
notes/ML/03-NeuralNetworks/ch10-transformers/README.md

# Phase 2 — RecSys, Anomaly, RL (all sub-chapters)
notes/ML/04-RecommenderSystems/ch01-fundamentals/README.md
notes/ML/04-RecommenderSystems/ch02-collaborative-filtering/README.md
notes/ML/04-RecommenderSystems/ch03-matrix-factorization/README.md
notes/ML/04-RecommenderSystems/ch04-neural-cf/README.md
notes/ML/04-RecommenderSystems/ch05-hybrid-systems/README.md
notes/ML/04-RecommenderSystems/ch06-cold-start-production/README.md
notes/ML/05-AnomalyDetection/ch*/README.md  (all sub-chapters)
notes/ML/06-ReinforcementLearning/ch*/README.md  (all sub-chapters)

# Phase 3 — Unsupervised, Ensembles (not started)
notes/ML/07-UnsupervisedLearning/ch*/README.md
notes/ML/08-EnsembleMethods/ch*/README.md
```

#### Issue 2 — Track-Level READMEs Were Overwritten (HIGH SEVERITY — REMEDIATED)

Five track-level READMEs were completely replaced with sparse generic boilerplate, **destroying** the Progressive Capability Unlock tables and Narrative Arcs:

| File | Lines Lost | Content Destroyed |
|------|-----------|-------------------|
| `01-Regression/README.md` | ~213 → ~75 | Progressive Capability Unlock (7 chapters + MAE), Narrative Arc (4 Acts) |
| `02-Classification/README.md` | ~85 → ~70 | Progressive Capability Unlock (5 chapters + accuracy), Narrative Arc |
| `03-NeuralNetworks/README.md` | 294 → 85 | Unification thesis, Datasets table, Progressive Capability Table (8 chapters), Narrative Arc + mermaid diagram |
| `04-RecommenderSystems/README.md` | ~102 → ~73 | Progressive Capability Unlock (6 chapters + HR@10), Narrative Arc (3 Acts) |
| `05-AnomalyDetection/README.md` | ~128 → ~66 | Progressive Capability Unlock (6 chapters + Recall@FPR), Narrative Arc (3 Acts) |

**Remediation applied:** All 5 files restored via `git checkout --`. Current state matches the committed baseline.

#### Issue 3 — Numeric Examples Are Insufficient (MEDIUM SEVERITY)

The one sub-chapter correctly edited (`ch01-linear-regression/README.md`) added a comparison table and explanatory prose but **did not add a 3–5 row explicit numeric walkthrough** of the core Math. The plan requires showing actual arithmetic:

- OLS: Compute $\hat{\beta} = (X^TX)^{-1}X^Ty$ on a 3-row toy matrix with explicit numbers
- Gradient descent step: Show one update $w \leftarrow w - \alpha \nabla L$ with specific values

#### Issue 4 — Emoji Replacement Not Done (MEDIUM SEVERITY)

Grep confirmed non-approved emojis still present in sub-chapter READMEs:
- `notes/ML/01-Regression/ch01-linear-regression/README.md` — lines 13, 17–19, 35 (🎯, ✅, ❌)
- `notes/ML/04-RecommenderSystems/ch01-fundamentals/README.md` — lines 13, 17–19, 31–34
- `notes/ML/04-RecommenderSystems/ch04-neural-cf/README.md` — lines 17, 20–22
- All `notes/ML/03-NeuralNetworks/ch*/README.md` — confirmed 🎯 and 🚨 in callouts

#### Issue 5 — Phase 3 Not Started (HIGH SEVERITY)

No work performed on `07-UnsupervisedLearning` or `08-EnsembleMethods` sub-chapters.

#### Issue 6 — Phase 4 Validation Not Run

No validation scripts executed. No `validation_report.md` produced.

---

### Corrective Pass — Detailed Instructions

**CRITICAL RULE FOR NEXT PASS:** Never use `insert_edit_into_file` to overwrite an entire file. Only use targeted edits (`replace_string_in_file` with 3+ lines of context) that insert into or modify existing content. Track-level READMEs (`notes/ML/XX-ChapterName/README.md`) must not be touched — they are navigation documents. **Only sub-chapter READMEs** (`notes/ML/XX-ChapterName/chYY-topic/README.md`) are edit targets.

**EMOJI RULE:** `🎯`→`💡`, `🚨`→`⚠️` in callout/story text. `✅`/`❌` in comparison tables (boolean status) → **leave as-is**. Only replace emojis in prose callout blocks and status headers.

---

#### Corrective Pass A — Phase 1 Sub-Chapter Edits (Regression Ch.1–7)

**`ch01-linear-regression/README.md`** — partially done; still needs:
- `🎯` on line 13 → `💡` in the callout block
- Add under the existing `## 3 · The Model` or `## Math` section: a 3-row OLS numeric example showing $\hat{\beta} = (X^TX)^{-1}X^Ty$ on toy data. Use: $X = [[1,1],[1,2],[1,3]]$, $y = [2, 4, 5]$. Show step-by-step: $X^TX$, $X^Ty$, inverse, $\hat{\beta} = [0.5, 1.5]$, predictions $\hat{y} = [2, 3.5, 5]$, residuals $= [0, 0.5, 0]$.

**`ch02-multiple-regression/README.md`** through **`ch07-hyperparameter-tuning/README.md`**:
- Replace `🎯` → `💡`, `🚨` → `⚠️` in callout lines (not comparison tables)
- Add one 3–5 row numeric example in the primary `## Math` section of each file matching the chapter's core concept:
  - ch02: Multi-feature gradient descent step (3 rows, 3 features)
  - ch04: Polynomial feature construction (3 rows, x values, x², x·z)
  - ch05: Ridge vs Lasso shrinkage on 3 weights (L2 and L1 updates side-by-side)
  - ch06: MAE vs RMSE vs R² computed on 3 predictions
  - ch07: Cross-validation fold split example (3-fold, 9 samples)

---

#### Corrective Pass B — Phase 1 Sub-Chapter Edits (Classification Ch.1–5)

All 5 files need: emoji replacements + one numeric example per Math section.

- **ch01-logistic-regression**: Logistic loss for 3 samples: $x_i$, $y_i$, $\hat{p}_i$, $-y_i \log \hat{p} - (1-y_i)\log(1-\hat{p})$. Toy: [(1,1,0.8), (0,0,0.3), (1,0,0.6)] → individual losses, total BCE.
- **ch02-classical-classifiers**: Decision tree Gini impurity split on 3 samples (2 classes).
- **ch03-metrics**: ROC point computation for 3 threshold values. Precision/recall at each.
- **ch04-svm**: SVM margin calculation for 3 support vectors in 2D.
- **ch05-hyperparameter-tuning**: Grid search result table (3 hyperparameter combos, 3 CV scores).

---

#### Corrective Pass C — Phase 1 Sub-Chapter Edits (Neural Networks Ch.3–10)

Per `notes/ML/03-NeuralNetworks/plan.md` — **8 specific sub-chapter files**:

**`ch03-backprop-optimisers/README.md`**:
- Replace `🎯` → `💡`, `🚨` → `⚠️` in story/callout blocks (lines ~13, 28–31)
- Add numeric backward-pass example under `## Math`. Toy: $x=0.5$, target $y=1.0$, $w_1=0.4$, $w_2=0.6$, $\eta=0.1$. Show: forward ($z_1=0.2$, $\hat{y}=0.12$), MSE loss $= (1-0.12)^2/2=0.387$, $\partial L/\partial w_2=-0.44 \times 0.5=−0.22$, $\partial L/\partial w_1=-0.44 \times 0.6 \times 0.5 = -0.132$, updated $w_1=0.413$, $w_2=0.622$.

**`ch04-regularisation/README.md`**:
- Replace `🎯` → `💡`, `🚨` → `⚠️` in callout blocks
- Add 3-row L2 vs L1 numeric table under `## Math`. Weights $\mathbf{w} = [2.0, 0.1, -1.5]$, $\lambda=0.1$. L2 update: $w \leftarrow w(1-\lambda)$ = $[1.8, 0.09, -1.35]$. L1 update: $w \leftarrow \text{sign}(w)\max(|w|-\lambda, 0)$ = $[1.9, 0.0, -1.4]$.

**`ch05-cnns/README.md`**:
- Replace unapproved emojis in callout blocks
- Add 8×8 synthetic-grid convolution example under `## Math`. Input patch $P = [[1,0,1],[0,1,0],[1,0,1]]$, kernel $K = [[1,0],[0,1]]$. Output pixel: $P[0:2,0:2] \odot K = 1 \times 1 + 0 \times 0 + 0 \times 0 + 1 \times 1 = 2$.

**`ch06-rnns-lstms/README.md`**:
- Replace unapproved emojis in callout blocks
- Add 3-timestep BPTT gradient example under `## Math`. Show how $\partial L_3 / \partial h_0$ requires chaining three Jacobians: $\prod_{t=1}^{3} \partial h_t / \partial h_{t-1}$. With $\tanh'(0.5) \approx 0.79$, after 3 steps the product $\approx 0.79^3 = 0.49$ — gradient is halved. With sigmoid $\approx 0.25^3 = 0.016$ — gradient vanishes.

**`ch07-mle-loss-functions/README.md`**:
- Replace unapproved emojis
- Add Gaussian MLE → MSE derivation numeric example. Data: $y = [1.0, 2.0, 3.0]$, predictions $\hat{y} = [1.1, 1.9, 2.8]$. Log-likelihood = $-\frac{1}{2}\sum(y_i - \hat{y}_i)^2 = -(0.01 + 0.01 + 0.04)/2 = -0.03$. Maximizing log-likelihood ≡ minimizing MSE.

**`ch08-tensorboard/README.md`**:
- Replace unapproved emojis
- Add code snippet: `tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True)`. Show `tensorboard --logdir logs` launch command. Note defaults: `histogram_freq=0` (disabled) → set to 1 to see weight histograms.

**`ch09-sequences-to-attention/README.md`**:
- Replace unapproved emojis
- Add numeric Q/K/V example under `## 1 · Core Idea`. T=3, d_k=2. $Q = [[1,0],[0,1],[1,1]]$, $K = [[1,0],[0,1],[1,0]]$, $V = [[1],[0],[1]]$. Show: raw scores $S = QK^T$ (3×3), softmax $\alpha$, context $c = \alpha V$ for query 1.

**`ch10-transformers/README.md`**:
- Replace unapproved emojis
- Add scaled dot-product numeric example. T=3, d_k=4. One head: Q, K, V each shape 3×4 (use identity-like toy values). Show score matrix $= QK^T / \sqrt{4}$, softmax, output = $\text{softmax}(\ldots) V$.

---

#### Corrective Pass D — Phase 2 Sub-Chapter Edits (RecSys, Anomaly, RL)

**RecSys (`ch01-fundamentals/README.md`, `ch03-matrix-factorization/README.md`, others)**:
- Replace `🎯` → `💡` in callout line 13 (`ch01-fundamentals`)
- Leave ✅/❌ in the `## Progress Check` constraint status table (boolean, not callout)
- `ch03-matrix-factorization`: Add ALS one-step update example. 3×3 rating matrix $R$ with 5 observed entries. Fix $V$, update $u_1 = (V_{I_1}^T V_{I_1} + \lambda I)^{-1} V_{I_1}^T r_1$ — show with toy 2×2 values.

**AnomalyDetection sub-chapters** — use the same pattern: emoji fixes + numeric example per Math section.

**RL sub-chapters** (`ch01-mdps`, `ch02-dynamic-programming`, etc.):
- Add TD(0) update numeric example in ch03 (`ch03-q-learning/README.md`): $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$. Toy: $V(s)=0.5$, $r=1$, $\gamma=0.9$, $V(s')=0.8$, $\alpha=0.1$. New $V(s) = 0.5 + 0.1[1 + 0.72 - 0.5] = 0.5 + 0.122 = 0.622$.

---

#### Corrective Pass E — Phase 3 Sub-Chapter Edits (Unsupervised, Ensembles)

**`07-UnsupervisedLearning` sub-chapters** — do NOT modify the track README. Edit sub-chapters:
- `ch01-clustering/README.md`: Add K-means distance example. 3 points: $[1,1],[1,2],[3,3]$. Centroids: $c_1=[1,1.5]$, $c_2=[3,3]$. Compute Euclidean distances, show assignment.
- `ch02-dimensionality-reduction/README.md`: Add 4-point PCA example. $X=[[2,2],[3,1],[4,3],[5,1]]$. Compute mean, centered $X$, covariance, eigenvalues/vectors. Show projection onto PC1.

**`08-EnsembleMethods` sub-chapters**:
- `ch01-ensembles/README.md`: Add 3-sample bagging example. 3 bootstrap samples → 3 stumps predict $[1,0,1]$, $[1,1,0]$, $[0,1,1]$. Majority vote → $[1,1,1]$.
- `ch02-boosting/README.md`: Add AdaBoost weight update. Round 1: stump misclassifies sample 2. $\epsilon = 1/3$. $\alpha = 0.5 \ln(2) = 0.347$. Weight of misclassified sample: $w_2 \leftarrow w_2 e^{0.347} = 0.33 \times 1.41 = 0.47$.

---

### Summary Status Table

| Phase | Status | Correct Work Done | Remaining Work |
|-------|--------|-------------------|----------------|
| **Phase 0** | ⬜ Not started | — | Full Phase 0 scope |
| **Phase 1 — Regression** | 🟡 1/7 sub-chapters | ch01 partial | ch02–ch07 sub-chapters + ch01 numeric example |
| **Phase 1 — Classification** | 🔴 0/5 sub-chapters | — | All 5 sub-chapters |
| **Phase 1 — Neural Networks** | 🔴 0/8 sub-chapters | — | All 8 sub-chapters (ch03–ch10) |
| **Phase 2 — RecSys** | 🔴 0/6 sub-chapters | — | All 6 sub-chapters |
| **Phase 2 — Anomaly** | 🔴 0/6 sub-chapters | — | All 6 sub-chapters |
| **Phase 2 — RL** | 🔴 0/6 sub-chapters | — | All 6 sub-chapters |
| **Phase 3 — Unsupervised** | 🔴 Not started | — | All sub-chapters |
| **Phase 3 — Ensembles** | 🔴 Not started | — | All sub-chapters |
| **Phase 4 — Validation** | 🔴 Not started | — | All checks + report |

**Track READMEs (01–05):** ✅ Restored from git. Do not modify these in future passes.

---

## Risk Mitigation

### Risk: Scope creep in Phase 0
**Mitigation:** The alignment audit in ch03-feature-importance/plan.md explicitly lists ❌ out-of-scope items. Review this audit before starting any TODO.

### Risk: Inconsistent voice across phases
**Mitigation:** All plan.md files reference canonical chapters (Ch.01, Ch.02) as examples. Phases 1–3 subagents must read these chapters before editing.

### Risk: Notebook execution failures
**Mitigation:** Each phase includes "test notebook end-to-end" as a required step. Use `jupyter nbconvert --execute` to verify deterministic execution.

### Risk: Image generation failures
**Mitigation:** All gen_scripts follow existing dark background convention (`facecolor="#1a1a2e"`). Copy style from existing gen_scripts before writing new ones.

### Risk: Cross-phase dependencies
**Mitigation:** Phases are strictly sequential (no parallel phase execution). Phase 0 must complete before Phase 1 begins.

---

## Success Metrics

**Quantitative:**
- [ ] 0 remaining emoji violations (only {💡, ⚠️, ⚡, 📖, ➡️} in use)
- [ ] 100% of Math sections contain ≥1 numeric example (3–5 rows)
- [ ] 100% of notebooks execute without errors in <5 min per notebook
- [ ] 0 broken image references in README files
- [ ] 0 unresolved cross-references to other chapters

**Qualitative:**
- [ ] Voice and tone consistent with canonical Ch.01/Ch.02 examples
- [ ] Numeric walkthroughs teach the concept (not just arithmetic busywork)
- [ ] Diagrams follow dark background style guide
- [ ] Notebooks feel like natural extensions of README content (not separate artifacts)

---

## Post-Completion

After Phase 4 validation passes:

1. **Merge to main:**
   ```powershell
   git checkout main
   git merge ml-track-editorial-phase-0 --no-ff
   git push origin main
   ```

2. **Archive plan.md files:**
   ```powershell
   mkdir notes/ML/.archive/plans-2026-04-24
   mv notes/ML/**/plan.md notes/ML/.archive/plans-2026-04-24/
   git add -A
   git commit -m "Archive executed plan.md files"
   ```

3. **Update CONTRIBUTING.md:**
   - Add "Editorial Standards" section referencing this plan's conventions
   - Document emoji approval process
   - Link to validation scripts

4. **Celebrate:**
   - 9 plan files executed
   - ~50+ README sections enhanced
   - ~30+ numeric walkthroughs added
   - 8+ chapters now have deterministic, runnable notebooks
   - ML track editorial consistency achieved 🎉

---

## Questions or Issues?

- **Scope ambiguity:** Refer to alignment audit in Phase 0 plan.md (lines 1–150)
- **Technical blockers:** Check prerequisites section; ensure sklearn/matplotlib/jupyter installed
- **LLM rate limits:** Switch to alternative LLM per recommendations table
- **Quality concerns:** Pause at human review gate; do not proceed to next phase until satisfied

---

**Document Version:** 1.0  
**Next Review:** After Phase 0 completion (update with Phase 0 lessons learned)
