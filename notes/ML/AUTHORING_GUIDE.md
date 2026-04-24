# ML Track — Authoring Guide

> **This document tracks the chapter-by-chapter build of the ML notes library.**  
> Each chapter lives under `notes/ML/` in its own folder, containing a README and a Jupyter notebook.  
> Read this before starting any chapter to keep tone, structure, and the running example consistent.

<!-- LLM-STYLE-FINGERPRINT-V1
canonical_chapters: ["notes/ML/01-Regression/ch01-linear-regression/README.md", "notes/ML/01-Regression/ch02-multiple-regression/README.md"]
voice: second_person_practitioner
register: technical_but_conversational
formula_motivation: required_before_each_formula
numerical_walkthroughs: california_housing_3_to_5_rows_with_explicit_arithmetic
dataset: california_housing_only_no_synthetic_data_except_toy_subsets
failure_first_pedagogy: true
callout_system: {insight:"💡", warning:"⚠️", constraint:"⚡", optional_depth:"📖", forward_pointer:"➡️"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_plots
section_template: [story_header, challenge_0, animation, core_idea_1, running_example_2, math_3, step_by_step_4, key_diagrams_5, hyperparameter_dial_6, code_skeleton_7, what_can_go_wrong_8, progress_check_N, bridge_N1]
math_style: scalar_first_then_vector_generalization
ascii_matrix_diagrams: required_for_matrix_operations
forward_backward_links: every_concept_links_to_where_it_was_introduced_and_where_it_reappears
conformance_check: compare_new_chapter_against_ch01_and_ch02_before_publishing
red_lines: [no_formula_without_verbal_explanation, no_concept_without_california_housing_grounding, no_section_without_forward_backward_context, no_math_derivation_without_toy_numerical_example, no_callout_box_without_actionable_content]
-->

---

## The Plan

The notes library is currently 19 chapters. Ch.1–Ch.14 cover the classical / neural foundations; Ch.15 (MLE & Loss Functions), Ch.16 (TensorBoard), Ch.17 (From Sequences to Attention — bridge chapter), Ch.18 (Transformers & Attention), and Ch.19 (Hyperparameter Tuning) extend the curriculum into modern architectures. We're converting each into a standalone, runnable learning module:

```
notes/ML/
├── ch01-linear-regression/
│   ├── README.md          ← Technical deep-dive + diagrams
│   └── notebook.ipynb     ← Runnable code that mirrors the README
├── ch02-logistic-regression/
│   ├── README.md
│   └── notebook.ipynb
│ ... (17 chapters total)
```

Each module is self-contained. Read the README to understand the concept, run the notebook to see it in action. The README and notebook teach exactly the same things in the same order.

---

## The Running Example — California Housing

Every chapter uses a **single consistent dataset**: the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (`sklearn.datasets.fetch_california_housing`).

The scenario: *you're a data scientist building a home valuation and market intelligence tool for a real estate platform.*

This one dataset threads naturally through all 17 chapters:

| Chapter | What we do with housing data |
|---|---|
| Ch.1 — Linear Regression | Predict `median_house_value` from `median_income` (one feature) |
| Ch.2 — Logistic Regression | Classify: will a district be "high-value" (above median price)? |
| Ch.3 — XOR Problem | Show why a linear boundary fails to separate coastal vs inland expensive districts |
| Ch.4 — Neural Networks | Build a multi-feature neural network for house value prediction |
| Ch.5 — Backprop & Optimisers | Train Ch.4's network — watch loss curves with SGD vs Adam |
| Ch.6 — Regularisation | Prevent the Ch.4 model from memorising training districts |
| Ch.7 — CNNs | Classify property condition from aerial/street-view photo grids |
| Ch.8 — RNNs / LSTMs | Predict monthly housing price index as a time series |
| Ch.9 — Metrics Deep Dive | Deeply evaluate the Ch.2 high-value classifier (precision, recall, AUC) |
| Ch.10 — Classical Classifiers | Use Decision Trees and KNN to classify neighbourhood price tiers |
| Ch.11 — SVM & Ensembles | XGBoost house value regression — compare with Ch.1's linear baseline |
| Ch.12 — Clustering | K-Means / DBSCAN to discover natural neighbourhood clusters |
| Ch.13 — Dimensionality Reduction | PCA / t-SNE / UMAP on the full housing feature space |
| Ch.14 — Unsupervised Metrics | Evaluate the Ch.12 clusters (Silhouette, Davies-Bouldin, ARI) |
| Ch.15 — MLE & Loss Functions | Derive MSE and Cross Entropy from MLE — when to use which loss |
| Ch.16 — TensorBoard | Instrument the Ch.5 training loop with TensorBoard scalars, histograms, and projector |
| Ch.17 — From Sequences to Attention | **Bridge chapter.** Treat each district's 8 features as a sequence of tokens; implement attention as a soft dictionary lookup with nothing beyond `numpy` dot product + softmax |
| Ch.18 — Transformers & Attention | Build a minimal transformer encoder on the housing feature set; observe how attention weights reflect feature correlations (income ↔ value) |
| Ch.19 — Hyperparameter Tuning | Sweep every major dial (learning rate, optimiser, batch size, init, regularisation, depth, width, data size) on the Ch.4 housing network |

> **Why this works:** The dataset is built into sklearn (no download required), has both regression and classification targets, has continuous and categorical features, and 20,000 rows — large enough to show real training dynamics without being slow.

---

## The Grand Challenge — SmartVal AI Production System

> **NEW**: Every chapter now threads through a unified production-system challenge. This framework mirrors the "knuckleball free kick" arc from the Math prerequisites track.

### The Scenario

You're the **Lead ML Engineer** at a major real estate platform (Zillow/Redfin scale). The CEO wants to launch **"SmartVal AI"** — a flagship intelligent home valuation and market intelligence system for production use.

This isn't a Kaggle competition. It's a **production system** that real estate agents, lenders, and homebuyers will rely on for multi-million-dollar decisions. It must satisfy strict business and regulatory requirements.

### The 5 Core Constraints

Every chapter explicitly tracks which constraints it helps solve:

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **ACCURACY** | <$40k Mean Absolute Error on median house values | Appraisal regulations require estimates within 20% of true value. Miss this → lose lending partnerships |
| **#2** | **GENERALIZATION** | Work on unseen districts + future expansion (CA → nationwide) | Can't just memorize training ZIP codes. Must learn true patterns, not artifacts |
| **#3** | **MULTI-TASK** | Predict BOTH median value (regression) AND market segment (classification) | Investors need "High-value coastal" vs "Affordable inland" classifications alongside prices |
| **#4** | **INTERPRETABILITY** | Predictions must be explainable to non-technical stakeholders | Lending decisions require justifiable valuations (regulatory compliance). "The neural net said so" doesn't work |
| **#5** | **PRODUCTION-READY** | Handle missing data, scale to millions, <100ms inference, monitoring | Research notebooks ≠ production systems. Must bridge the gap |

### Progressive Capability Unlock (19 Chapters)

| Ch | What Unlocks | Constraints Addressed | Status |
|----|--------------|----------------------|--------|
| 1 | Single-feature baseline ($70k MAE) | #1 Partial | Foundation |
| 2 | Binary classification (high/low value) | #3 Partial | Classification unlocked |
| 3 | Diagnose linear limits | None | Problem revealed |
| 4 | Non-linear modeling ($55k MAE) | #1 Major step | But no training yet |
| 5 | Backprop + optimizers | **#1 ✅ <$40k MAE achieved!** | Accuracy unlocked! |
| 6 | Regularization (L1/L2/Dropout) | **#2 ✅ Generalization** | No memorization |
| 7 | CNNs for aerial photos | #5 Partial | Image features |
| 8 | RNNs for price trends | #5 Partial | Time series |
| 9 | Metrics deep dive | Validation for #1 #2 #3 | Measurement |
| 10 | Interpretable trees | #4 Partial | Accuracy vs interpretability tradeoff |
| 11 | XGBoost + SHAP | **#4 ✅ Accuracy + explainability** | Best of both worlds |
| 12 | Clustering (K-Means) | **#3 ✅ Market segmentation** | Unsupervised segments |
| 13 | PCA/t-SNE | #5 Partial | Faster inference |
| 14 | Unsupervised metrics | Validate #3 clusters | Cluster quality |
| 15 | MLE + loss theory | Foundation | Understand all losses |
| 16 | TensorBoard | **#5 Partial — Monitoring** | Production tooling |
| 17 | Attention mechanics | #1 #4 | Interpretable weights |
| 18 | Transformers | #1 #2 #3 all optimized | SOTA architecture |
| 19 | Hyperparameter tuning | **#5 ✅ Production-ready!** | 🎉 **COMPLETE!** |

---

## Chapter README Template

Every chapter README now follows this **extended structure** (adds §0 Challenge and §N Progress Check):

```
# Ch.N — [Topic Name]

> **The story.** (Historical context — who invented this, when, why)
>
> **Where you are in the curriculum.** (Links to previous chapters, what this adds)
>
> **Notation in this chapter.** (Declare all symbols upfront)

---

## 0 · The Challenge — Where We Are

> 🎯 **The goal**: Launch **[Grand Challenge Name]** — [one-sentence mission] satisfying 5 constraints:
> 1. ACCURACY: [target metric and threshold]
> 2. GENERALIZATION: [unseen-data target]
> 3. MULTI-TASK / MULTI-LABEL: [multi-output target]
> 4. INTERPRETABILITY: [explainability requirement]
> 5. PRODUCTION: [latency / scale / monitoring target]

**What we know so far:**
- ✅ [Summary of previous chapters' achievements]
- ❌ **But we still can't [X]!**

**What's blocking us:**
[Concrete description of the gap this chapter addresses]

**What this chapter unlocks:**
[Specific capability that advances one or more constraints]

---

## 1 · The Core Idea (2–3 sentences, plain English)

## 2 · Running Example: What We're Solving
(one paragraph: plug the track's running scenario and dataset into this chapter's concept — see the track README for the dataset and grand challenge)

## The Math
(key equations, annotated — no wall-of-symbols, every term explained inline)

## How It Works — Step by Step
(numbered list or flow diagram in Mermaid/ASCII)

## The Key Diagrams
(Mermaid diagrams or ASCII art — minimum 1)

## The Hyperparameter Dial
(the main tunable, its effect, typical starting value)

## Code Skeleton
(minimal Python — illustrative, not copy-paste production)

## What Can Go Wrong
(3–5 bullet traps, each one sentence)

## N-1 · Where This Reappears
(Forward links to later chapters that build on this concept)

## N · Progress Check — What We Can Solve Now

![Progress visualization](./img/chNN-progress-check.png) ← **Optional**: Visual dashboard showing constraint progress

✅ **Unlocked capabilities:**
- [Specific things you can now do]
- [Constraint achievements: "Constraint #1 ✅ Achieved! <$40k MAE"]

❌ **Still can't solve:**
- ❌ [What's blocked — explicitly preview next chapter's unlock]
- ❌ [Other remaining challenges]

**Real-world status**: [One-sentence summary: "We can now X, but we can't yet Y"]

**Next up:** Ch.X gives us **[concept]** — [what it unlocks]

---

## N+1 · Bridge to the Next Chapter
(one clause what this established + one clause what next chapter adds)
```

**Note:** Interview checklists are maintained in the centralized [Interview_guide.md](Interview_guide.md) file, not in individual chapters.

---

## Jupyter Notebook Template

Each notebook mirrors the README exactly — same sections, same order. The notebook adds:
- **Runnable cells**: every code block in the README is a cell in the notebook
- **Visual outputs**: `matplotlib` / `seaborn` plots that generate the diagrams described in the README
- **Exercises**: 2–3 cells at the end where the reader changes a hyperparameter and re-runs

Cell structure per notebook:

```
[markdown] Chapter title + one-liner
[markdown] ## The Core Idea
[markdown] ## Running Example
[code]     Load the track dataset (see track README for the running example dataset and scenario)
[markdown] ## The Math
[code]     Implement the math (numpy where practical, sklearn/tf for full models)
[markdown] ## Step by Step
[code]     The step-by-step walkthrough as runnable code
[code]     Plotting the key diagram
[markdown] ## The Hyperparameter Dial
[code]     Sweep the dial, plot before/after
[markdown] ## What Can Go Wrong
[code]     Demonstrate one of the traps
[markdown] ## Exercises
[code]     Exercise scaffolds (partially filled)
```

---

## Build Tracker

The ML track is organised into 8 independent topic-based folders, each with its own dataset, grand challenge, and chapter sequence. See each topic's README for chapter-level build status (README ✅, notebook ✅, done/in-progress).

| # | Topic | Folder | Grand Challenge | Chapters |
|---|-------|--------|----------------|----------|
| 1 | Regression | `01-Regression/` | <$40k MAE on housing values | 7 chapters + `GRAND_CHALLENGE.md` |
| 2 | Classification | `02-Classification/` | >90% avg accuracy across 40 facial attributes | 5 chapters |
| 3 | Neural Networks | `03-NeuralNetworks/` | $28k MAE + 95% accuracy with shared architecture | 10 chapters |
| 4 | Recommender Systems | `04-RecommenderSystems/` | >85% hit rate @ top-10 | 6 chapters |
| 5 | Anomaly Detection | `05-AnomalyDetection/` | 80% recall @ 0.5% FPR | 6 chapters |
| 6 | Reinforcement Learning | `06-ReinforcementLearning/` | Conceptual mastery (theory-only, no notebooks) | 6 chapters |
| 7 | Unsupervised Learning | `07-UnsupervisedLearning/` | Silhouette >0.5, 5 actionable segments | 3 chapters |
| 8 | Ensemble Methods | `08-EnsembleMethods/` | Beat single models by 5%+ | 6 chapters |

> For chapter-level status, see the individual topic README linked in the table above.

---

## Chapter Summaries (Quick Reference)

> ⚠️ **Legacy reference** — these summaries describe chapters from the old single-track layout (`notes/ML/ch01-linear-regression` … `ch19-hyperparameter-tuning`), which has been reorganised into 8 independent topic-based tracks. For current chapter-level summaries, see each topic's own README.

Brief bullet on what each chapter covers — so you can pick up any chapter without re-reading the HTML book.

### Ch.1 — Linear Regression
- Model: `ŷ = wx + b` → extend to `ŷ = Wᵀx + b`
- Loss: MSE, MAE, RMSE; Metric: R², Adjusted R²
- Training: Gradient Descent (batch, SGD, mini-batch)
- Dial: learning rate α
- Trap: R² always increases with more features — use Adjusted R²

### Ch.2 — Logistic Regression
- Model: sigmoid squashes the linear output to [0,1] → probability
- Loss: Binary Cross-Entropy (log loss)
- Metric: Accuracy, Precision, Recall, F1, AUC-ROC
- Dial: decision threshold (default 0.5 — rarely optimal)
- Trap: high accuracy on imbalanced datasets is meaningless

### Ch.3 — The XOR Problem
- Why a single perceptron can't solve XOR (not linearly separable)
- Universal Approximation Theorem: one hidden layer can approximate any function
- Introduces the need for hidden layers and non-linear activations
- Dial: number of hidden units
- Trap: more units ≠ better generalisation without regularisation

### Ch.4 — Neural Networks
- Architecture: input → [Dense + activation] × N → output
- Activations: ReLU, Sigmoid, Tanh, Softmax — when to use each
- Weight initialisation: Xavier, He
- Dial: depth (layers) and width (units per layer)
- Trap: wrong activation on the output layer (sigmoid vs softmax vs linear)

### Ch.5 — Backprop & Optimisers
- Backpropagation: chain rule applied layer by layer
- Optimisers: SGD → Momentum → RMSProp → Adam
- Learning rate schedules: step decay, cosine annealing, warmup
- Dial: learning rate + optimiser choice
- Trap: Adam's adaptive rate can mask bad architectures

### Ch.6 — Regularisation
- L1 (Lasso): pushes weights to zero → feature selection
- L2 (Ridge / weight decay): shrinks weights → smooth model
- Dropout: randomly zeros units during training
- Early stopping: halt on validation loss plateau
- Dial: dropout rate, λ (L2), patience (early stopping)
- Trap: applying dropout before the output layer

### Ch.7 — CNNs
- Convolution: sliding filter extracts local features
- Pooling: max/avg pooling reduces spatial size
- Feature hierarchy: edges → textures → parts → objects
- Architecture progression: LeNet → AlexNet → VGG → ResNet idea
- Dial: filter count (32→64→128), kernel size
- Trap: not using BatchNorm after deep conv stacks

### Ch.8 — RNNs / LSTMs / GRUs
- RNN: hidden state carries context forward
- Vanishing gradient: gradients shrink exponentially through time steps
- LSTM: gates (input, forget, output) control what to remember
- GRU: lighter alternative — reset and update gates only
- Dial: LSTM units, sequence length, number of LSTM layers
- Trap: feeding the full sequence at once instead of step-by-step in custom loops

### Ch.9 — Metrics Deep Dive
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
- Regression: MSE, RMSE, MAE, MAPE, R², Adjusted R²
- Confusion matrix anatomy: TP, TN, FP, FN
- When to prefer recall over precision (and why)
- Trap: optimising for accuracy on class-imbalanced data

### Ch.10 — Classical Classifiers
- Decision Trees: split on information gain / Gini impurity
- KNN: classify by the k nearest neighbours in feature space
- Comparison: DT is interpretable but prone to overfit; KNN is lazy but scale-sensitive
- Dial: max_depth (DT), k (KNN)
- Trap: KNN on un-normalised features

### Ch.11 — SVM & Ensembles
- SVM: find the maximum-margin hyperplane
- Kernel trick: RBF maps data to higher dimensions implicitly
- Bagging: train many models on bootstrapped data → variance reduction (Random Forest)
- Boosting: train models sequentially on residuals → bias reduction (XGBoost, LightGBM)
- Dial: C and γ (SVM), n_estimators and max_depth (XGBoost)
- Trap: boosting on noisy labels → overfits fast

### Ch.12 — Clustering
- K-Means: assign to nearest centroid, recompute centroids, repeat
- DBSCAN: density-reachable clustering — handles arbitrary shapes, marks noise
- HDBSCAN: hierarchical DBSCAN with variable density tolerance
- Dial: k (K-Means), ε and min_samples (DBSCAN)
- Trap: K-Means on non-spherical clusters

### Ch.13 — Dimensionality Reduction
- PCA: find orthogonal directions of maximum variance
- t-SNE: preserve local neighbourhood structure (non-linear, non-invertible)
- UMAP: faster, topology-preserving, can be used for downstream tasks
- Dial: n_components (PCA), perplexity (t-SNE), n_neighbors (UMAP)
- Trap: t-SNE cluster distances are not meaningful — only topology is

### Ch.14 — Unsupervised Metrics
- Silhouette score: cohesion vs separation [-1, 1] — higher is better
- Davies-Bouldin index: ratio of within-cluster to between-cluster distance — lower is better
- Adjusted Rand Index (ARI): compare cluster labels to ground truth (when available)
- Explained Variance Ratio (PCA): how much variance each component captures
- Trap: picking k based only on silhouette without plotting the elbow curve

### Ch.15 — MLE & Loss Functions
- MLE: choose the model parameters that maximise the probability of observing the training data
- MSE derived from MLE under Gaussian noise assumption
- Cross Entropy derived from MLE under Bernoulli (binary) / Categorical (multi-class) assumption
- Decision rule: use the loss that matches the noise model of your target variable
- Dial: none — loss choice is determined by the output type, not tuned
- Trap: using MSE for classification (gradients vanish near 0/1; no probability calibration)

### Ch.16 — TensorBoard
- Scalars: log training/validation loss and metrics per epoch
- Histograms: track weight and gradient distributions across layers over time
- Projector (Embedding Visualiser): visualise high-dimensional representations via PCA / t-SNE
- Images: log sample predictions or feature maps
- Dial: log_dir, update_freq, histogram_freq
- Trap: logging every batch (use epoch-level to avoid disk bloat)

### Ch.17 — From Sequences to Attention (bridge chapter)
- Mental model: **attention is a soft dictionary lookup** — dot product + softmax + weighted sum of values
- Pre-teaches the three building blocks for Ch.18: dot product as similarity, softmax with temperature, soft-vs-hard lookup
- Introduces $Q, K, V$ as three **roles** of the same input (question / label / payload) without learned projections
- Shows self-attention is permutation-equivariant — motivates positional encoding as the price paid
- Shorter than a full chapter by design; exists so Ch.18 lands softly for learners without prior attention exposure
- Trap: treating attention weights as an *explanation* of model behaviour rather than a *diagnostic* of where the model looked

### Ch.18 — Transformers & Attention
- Scaled dot-product attention: `softmax(QKᵀ/√d_k)V` — every position attends to every other in parallel
- Positional encoding: sinusoidal PE injected additively so the model knows token order
- Multi-head attention: H parallel attention heads, each learning different relationship patterns
- Encoder block: Pre-LN → Multi-Head Attention → Residual → Pre-LN → FFN → Residual
- Encoder vs Decoder: one causal mask difference — upper-triangle of −∞ makes it autoregressive
- Dial: `d_model` (most impactful), `num_heads` (must divide `d_model`), `num_layers`, LR warmup
- Trap: forgetting LR warmup (transformers diverge at initialisation without it); `num_heads` not dividing `d_model` silently corrupts projections

### Ch.19 — Hyperparameter Tuning
- Parameters (`W, b`) are learned; hyperparameters are chosen before training
- Tuning order: learning rate → batch size → optimiser → initialiser → architecture (depth/width/layer type) → regularisation (dropout, weight decay, early stopping) → loss choice → more data
- Dials covered: learning rate & schedules, optimisers (SGD/Momentum/Adam/AdamW), batch size, weight init (He/Xavier), dropout, loss choice, layer types, depth, width, activation functions, weight decay, gradient clipping, epochs/early stopping
- Bias vs variance decision: learning curves tell you whether more data will help
- Search strategy: random search ≫ grid in high dimensions; Bayesian (Optuna) for expensive training runs
- Trap: tuning multiple dials at once, tuning on the test set, picking depth/width before learning rate

---

## Conventions

**Diagrams:** Use Mermaid flowcharts (`flowchart TD`) or `flowchart LR` for pipelines. Use ASCII art for weight matrices and mathematical structures where Mermaid is overkill.

**Code style:** Python, sklearn + TensorFlow/Keras. Keep cells short — one idea per cell. Import only what's needed for that cell.

**Tone:** Direct and time-efficient. Assume the reader is smart and preparing for an interview. No "Let's explore together!" — every sentence earns its place.

**Equations:** Use LaTeX inline (`$...$`) and block (`$$...$$`) where supported. Always annotate every symbol on first use.

---

## How to Use This Document

1. Open this file to check what's done and what's next.
2. Pick the next ⬜ chapter from the tracker above.
3. Use the README template and notebook template above — don't invent new structures.
4. Keep the housing scenario in focus: every example should tie back to the real estate platform.
5. After completing a chapter, mark its checkboxes ✅ in the tracker.

---

## Style Ground Truth — Derived from Ch.01 and Ch.02

> **LLM instruction:** Before authoring or reviewing any chapter in this track, treat Ch.01 (`notes/ML/01-Regression/ch01-linear-regression/README.md`) and Ch.02 (`notes/ML/01-Regression/ch02-multiple-regression/README.md`) as the canonical style reference. Every dimension below was extracted from close reading of those two chapters. When a new or existing chapter deviates from any dimension, flag it. When generating new content, verify against each dimension before outputting.

---

### Voice and Register

**The register is: technical-practitioner, second person, conversational within precision.**

The reader is treated as a capable engineer who doesn't need flattery, gets impatient with abstract theory, and wants to know what to *do* and *why it matters*. The tone is direct — every sentence earns its place. There is no "Let's explore together!", no "In this section we will discuss", no hedging language that softens a concrete fact into a vague observation.

**Second person is the default.** The reader is placed inside the scenario at all times:

> *"You're a data scientist at a real estate platform. Your first task: build a model that estimates the median house value."*  
> *"Your manager calls: the luxury coastal segment is haemorrhaging client trust."*  
> *"You just did gradient descent. Very slowly. And by feel."*

**Dry, brief humour appears exactly once per major concept.** It is never laboured. The examples above — "by feel", "haemorrhaging client trust" — illustrate the register: wry, businesslike, never cute.

**Contractions and em-dashes are used freely** when they tighten a sentence:
> *"That's it."*  
> *"MSE gives urgency — but it can panic over the wrong things."*  
> *"Full stop."*

**Academic register is forbidden.** Phrases like "In this section we demonstrate", "It can be shown that", "The reader may note", "we present", "we propose" do not appear in these chapters and must not appear in any new chapter.

---

### Story Header Pattern

Every chapter opens with three specific items, in order, in a blockquote:

1. **The story** — historical context. Who invented this concept, in what year, on what problem. Always a real person and a real date. Ch.01 opens with Legendre (1805) and Gauss (1809). Ch.02 opens with Gauss again (1808) for multiple regression, then Fisher (1922) and Frisch–Waugh–Lovell (1933). The history is brief (one paragraph), specific (named people, named papers, named years), and closes with a sentence connecting the historical moment to the practitioner's daily work.

2. **Where you are in the curriculum** — one paragraph precisely describing what the previous chapter(s) gave you and what gap this chapter fills. Must name specific MAE numbers or constraint statuses from preceding chapters.

3. **Notation in this chapter** — a one-line inline declaration of every symbol introduced in the chapter, before the first section begins. Not a table — a single sentence with $inline$ math. Example from Ch.01: *"$x$ — input feature (`MedInc`); $y$ — true target (`MedHouseVal`); $\hat{y}=wx+b$ — model prediction; $w$ — weight (slope); $b$ — bias (intercept)..."*

---

### The Challenge Section (§0)

**Required pattern — followed exactly in both chapters:**

```
> 🎯 The mission: [one line, Grand Challenge name + constraint list]

What we know so far:
  ✅ [summary of what previous chapters have established]
  ❌ But we [specific capability that is still missing]

What's blocking us:
  [2–4 sentences: the concrete, named gap. Not abstract ("we need to address X")
   but specific ("A house in Bakersfield and a house in San Jose can have the same 
   median income and differ in value by $200k or more. Location matters.")]

What this chapter unlocks:
  [Specific capability bullet points with numbers where possible]
```

**Numbers are always named.** The gap is never "our model is not accurate enough" — it is "$55k MAE" vs. "$40k target". The blocker is never "non-linearity" — it is "income–value relationship curves at high incomes (diminishing returns)."

Ch.02 goes further and adds a Mermaid diagram in §0 showing Ch.1 architecture side-by-side with Ch.2, with arrows labelled with specific MAE values. This sets the standard for any chapter introducing a structural expansion of the model.

---

### The Failure-First Pedagogical Pattern

**This is the most important stylistic rule.** Concepts are never listed and explained — they are *discovered by exposing what breaks*.

The loss function section of Ch.01 is the canonical example:
- Act 1: introduce MAE because it's intuitive → show exactly where it breaks (luxury segment haemorrhage)
- Act 2: introduce MSE as the fix → show exactly where *that* breaks (outlier hijacking, units²)
- Act 3: introduce RMSE → show it's not a new idea, just a unit converter, same flaw as MSE
- Act 4: introduce Huber → show it fixes the tension precisely

Each step in the arc: **tool → specific failure → minimal fix → that fix's failure → next tool**. The reader is never asked to memorise a taxonomy of loss functions. They experience the need for each one before seeing it.

**This pattern must appear in every subsection that covers multiple options or variants.** If a section presents three methods (e.g., filter/wrapper/embedded feature selection), the section must show *what breaks* with the simpler method before introducing the more complex one. Listing methods without demonstrating failure is the wrong pattern.

---

### Mathematical Style

**Rule 1: scalar form before vector form.** Every formula is first shown for one sample or one feature, then generalised. The generalisation is presented as a direct extension, not a separate derivation.

Ch.01 §4: shows `ŷ = wx + b` first (one feature), then `ŷ = Wᵀx + b` (multiple features). Ch.02 §3.1: "Ch.1 (single feature): `ŷ = wx + b`" → "Ch.2 (multiple features): `ŷ = Σ wⱼxⱼ + b`" → matrix form.

**Rule 2: every formula is verbally glossed immediately after it appears.** Not in a table of notation (though those also exist) — in the prose directly below the LaTeX block:

> *"The denominator is the total squared error of predicting the training-set mean ȳ for every district — the dumbest possible baseline. R² is the fraction of that baseline error your model eliminates."*

If a formula has no verbal gloss within three lines, it is incomplete.

**Rule 3: the notation table lives in the header.** All symbols are declared in the "Notation in this chapter" header blockquote before any section. Subsections add no new notation without glossing it immediately.

**Rule 4: optional depth gets a callout box.** Derivations that would break the flow of a practitioner reading for intuition go inside an indented `> 📖 **Optional:**` block. These are clearly labelled and can be skipped without losing the main thread. Ch.01 puts the full matrix chain rule derivation inside one of these blocks. Ch.02 puts the full Jacobian derivation in one. The optional block ends with a cross-reference to MathUnderTheHood for the rigorous treatment.

**Rule 5: ASCII matrix diagrams for matrix operations.** When showing a matrix multiply or a matrix structure, draw it in ASCII with aligned brackets, showing the dimensions of each operand and the result. The Ch.02 `Xᵀe` walkthrough is the canonical example:

```
Xᵀ · e                                            (2×3) · (3×1) → (2×1)

  Xᵀ                          e
  ┌  0.5   1.5   2.0  ┐      ┌  -1.5  ┐
  └  1.0   0.0  -1.0  ┘  ×   │  -2.5  │
                              └  -4.0  ┘
```

---

### Numerical Walkthrough Pattern

**Every mathematical concept must be demonstrated on actual numbers before being generalised.** The walkthrough always uses a 3–5 row subset of the California Housing dataset (never entirely synthetic data — always features like MedInc, AveRooms, Latitude, Population).

**The canonical walkthrough structure** (from Ch.02 §3.4 "Watching the Vectors Move"):
1. State the toy dataset as a markdown table with named columns
2. State initial conditions (`w = [0, 0]`, `b = 0.0`, `α = 0.1`)
3. Show forward pass in a table: column for each feature, column for ŷᵢ, column for eᵢ, column for eᵢ²
4. Show gradient computation as ASCII matrix multiply
5. Show the numerical gradient values bolded
6. Show the update step as explicit arithmetic: `w₁ = 0.0 − 0.1 × (−8.333) = 0.833`
7. State the loss before and after, and compute the % reduction
8. Repeat for epoch 2 to show the pattern changes

**Every walkthrough ends with a verification sentence** — "The match is exact." or "MSE dropped from 8.167 → 1.233: an 85% reduction in one epoch." This confirms the arithmetic was correct and closes the example cleanly.

**Walkthroughs show both the scalar (manual) path and the vectorised equivalent.** The scalar computation is worked first to make the mechanics transparent, then a single-line matrix expression is shown that computes the same result.

---

### Forward and Backward Linking

**Every new concept is linked to where it was first introduced and where it will matter again.** This is not optional — both chapters do it on virtually every paragraph.

**Backward link pattern:** *"This is the same update rule from Ch.1 — the only difference is that Xᵀ now accumulates contributions from all d features."*

**Forward link pattern:** *"This is the entire conceptual foundation of neural network backpropagation. Every time you call `loss.backward()` in PyTorch, this matrix multiply is running — one per layer."*

**The forward pointer callout box** (`> ➡️`) is used for concepts that will be formally introduced later but need to be planted early. Ch.01 plants the seed for R² at the end of the loss section with a `> ➡️` callout that says R² will be introduced in Ch.02 §1.5 where comparing two models makes it meaningful.

**Cross-track links** to MathUnderTheHood are standard for rigorous derivations. Always reference the specific chapter: `[MathUnderTheHood ch06 — Gradient & Chain Rule](../MathUnderTheHood/ch06-gradient-chain-rule/)`.

---

### Callout Box System

Used consistently across both chapters. Must be used exactly this way — no improvised emoji or callout patterns:

| Symbol | Meaning | When to use |
|---|---|---|
| `💡` | Key insight / conceptual payoff | After a result that surprises or reframes something the reader thought they understood |
| `⚠️` | Warning / common trap | Before or immediately after a pattern that is often done wrong |
| `⚡` | Grand Challenge constraint connection | When content advances or validates one of the 5 SmartVal constraints |
| `> 📖 **Optional:**` | Deeper derivation | Full proofs and matrix calculus that break the narrative flow |
| `> ➡️` | Forward pointer | When a concept needs to be planted before its full treatment |

The callout box content is always **actionable**: it ends with a Fix, a Rule, a What-to-do. No callout box that just says "this is interesting" without consequence.

---

### Image and Animation Conventions

**Every image has a purpose — none are decorative.** Both chapters contain only images that demonstrate something the prose cannot fully convey with text: how a gradient step changes the line position, how loss contours change with scaling, how MAE vs MSE gradients diverge epoch-by-epoch.

**Image naming convention:**
- `ch0N-[topic]-[type].png/.gif` for chapter-specific generated images
- `[concept]_generated.gif/.png` for algorithmically generated animations
- Descriptive alt-text is mandatory: `![MSE(w) parabola (left) and its linear derivative dL/dw (right), making the residual-to-gradient link explicit](./img/loss_parabola_generated.png)`

**Generated plots use dark background `facecolor="#1a1a2e"`** — matching the chapter's rendered dark theme. Light-background plots are not used.

**Image types observed in Ch.01/Ch.02:**

| Type | Purpose | Examples |
|---|---|---|
| GIF animation | Show a process evolving over time: training, convergence | `epoch_walk_generated.gif`, `mae_mse_convergence.gif`, `gradient_descent_steps.gif` |
| PNG comparison | Side-by-side before/after | `feature_scaling_contours.png`, `loss_curves_mae_vs_mse.png` |
| PNG breakdown | Annotated diagram explaining one concept | `xtranspose_breakdown.png`, `huber_gradient_comparison.png` |
| PNG loss surface | 2D/3D visualisation of loss landscape | `loss_parabola_generated.png`, `loss_surface_ellipse.png` |
| GIF needle | Chapter-level progress animation (needle moving toward target) | `ch01-linear-regression-needle.gif`, `ch02-multiple-regression-needle.gif` |

**Every chapter has a needle GIF** — the chapter-level animation showing which constraint needle moved. This appears immediately after §0 under the heading `## Animation`.

**Mermaid diagram colour palette** — used consistently for all flowcharts:
- Primary/data: `fill:#1e3a8a` (dark blue)
- Success/achieved: `fill:#15803d` (dark green)
- Caution/in-progress: `fill:#b45309` (amber)
- Danger/blocked: `fill:#b91c1c` (dark red)
- Info: `fill:#1d4ed8` (medium blue)

All Mermaid nodes use `stroke:#e2e8f0,stroke-width:2px,color:#ffffff` for text legibility.

---

### Code Style

**Code blocks are minimal but complete.** The standard is "enough to run end-to-end with real output, nothing extra." No scaffolding classes, no type annotations on internal code, no error handling beyond what a practitioner would actually need.

**Variable naming is consistent across all chapters:**

| Variable | Meaning |
|---|---|
| `X_train`, `X_test` | Raw feature matrices |
| `X_train_s`, `X_test_s` | Standardised feature matrices |
| `y_train`, `y_test` | Target vectors |
| `model` | Fitted sklearn estimator |
| `mae` | Mean absolute error (in $100k units unless converted with `× 100_000`) |
| `w`, `b` | Manual gradient descent weights |
| `alpha` | Learning rate |
| `n` | Number of training samples |
| `d` | Number of features |

**Comments explain *why*, not *what*.** The code line `scaler.fit(X_train)` does not need a comment saying "fit the scaler". It needs a comment like `# use TRAIN statistics only — applying to test set avoids leakage`.

**The manual gradient descent loop always appears alongside the sklearn version** in the Code Skeleton section. The manual version is labelled "Educational: gradient descent from scratch" and the sklearn version is the production reference.

---

### Progress Check Section

The Progress Check is the last substantive section before the Bridge. It has a fixed format:

```
✅ Unlocked capabilities:
  [bulleted list — specific capabilities with named metrics]
  [e.g., "MAE improved: ~$55k (down from $70k — 21% improvement!)"]

❌ Still can't solve:
  [bulleted list — named, specific gaps]
  [e.g., "❌ $55k > $40k target — Linear model with 8 features still not accurate enough"]

Progress toward constraints:
  [table: Constraint | Status | Current State]

[Mermaid LR flowchart showing all chapters from Ch.1 to Ch.5+, 
 with current chapter highlighted and MAE values annotated]
```

The progress flowchart always shows the full forward arc, not just the current chapter. It anchors the reader in the overall narrative even when deep in one chapter's detail.

---

### What Can Go Wrong Section

**Format:** 3–5 traps, each following the pattern:
- **Bold name of the trap** — one clause description in the heading
- Explanation in 2–3 sentences with concrete numbers from the chapter's dataset
- **Fix:** one actionable sentence starting with "`Fix:`"

The section always ends with a Mermaid diagnostic flowchart that walks through the traps as decision branches. The flowchart is not a summary of the traps — it is a live diagnostic tool a practitioner can follow on a real problem.

---

### Section Depth vs. Length Contract

Both chapters are long — Ch.01 in particular runs to 1,100+ lines of Markdown. This length is earned, not padded. The standard:

- **Never summarise where you can demonstrate.** A worked 2-epoch gradient descent walkthrough that shows the arithmetic explicitly teaches the concept; a prose paragraph saying "the weights update toward the minimum" does not.
- **One concept per subsection.** Ch.01's §6 "Gradient Descent" has 7 distinct subsections (Try It First, Loss Surface, Convergence, MAE vs MSE comparison, Gradient Descent Lens, Feature Engineering). Each subsection has exactly one conceptual payload. None runs into another.
- **The subsection heading is descriptive, not label-like.** Not "6.5 Comparison" but "6.5 · MAE vs MSE — Why Gradient Shape Determines Convergence". The title states the conclusion, not just the topic.
- **100-line rule for inline explanations.** If explaining a concept fully would take more than ~100 lines in a natural reading flow, split it: give the intuition inline, move the full derivation to a `> 📖 Optional` callout box, and cross-reference MathUnderTheHood for the proof.

---

### What These Chapters Are Not

Understanding what the chapters deliberately avoid is as important as the positive rules:

- **Not a textbook reference.** They do not aim to cover all variants of a concept. They cover the variants you will encounter in practice and deliberately exclude the rest, with a footnote pointing elsewhere.
- **Not a tutorial.** They do not hold the reader's hand through copying code. The notebook does that. The README teaches the why so deeply that the how is obvious.
- **Not a paper.** No passive voice, no citations (except cross-references to MathUnderTheHood and official sklearn docs), no "it has been shown that." All claims are demonstrated numerically on the chapter's data.
- **Not an abstract lecture.** Every formula is anchored to a California Housing row within 3 lines of its introduction. The district, the income value, the predicted house price — always named.

---

### Conformance Checklist for New or Revised Chapters

Before publishing any chapter, verify each item:

- [ ] Story header: real person, real year, real problem — and a bridge to the practitioner's daily work
- [ ] §0 Challenge: specific numbers (MAE, constraint status), named gap, named unlock
- [ ] Notation block in header: all symbols declared inline before §0
- [ ] Every formula: verbally glossed within 3 lines
- [ ] Every formula: scalar form shown first, vector form second
- [ ] Every non-trivial formula: demonstrated on a 3–5 row California Housing toy dataset with explicit arithmetic
- [ ] Failure-first pedagogy: new concepts introduced because the simpler one broke, not listed a priori
- [ ] Optional depth: full derivations behind `> 📖 Optional` callout boxes with MathUnderTheHood cross-reference
- [ ] Forward/backward links: every concept links to where it was introduced and where it reappears
- [ ] Callout boxes: only `💡 ⚠️ ⚡ 📖 ➡️` — no improvised emoji
- [ ] Mermaid diagrams: colour palette respected (dark blue / dark green / amber / dark red)
- [ ] Images: dark background, descriptive alt-text, purposeful (not decorative)
- [ ] Needle GIF: chapter-level progress animation present under `## Animation`
- [ ] Code: `X_train_s`/`X_test_s` naming, fit-on-train-only scaler, manual GD loop + sklearn reference
- [ ] Progress Check: ✅/❌ bullets with specific numbers + constraint table + Mermaid LR arc
- [ ] What Can Go Wrong: 3–5 traps with Fix + diagnostic Mermaid flowchart
- [ ] Bridge section: one clause what this chapter established + one clause what next chapter adds
- [ ] Voice: second person, no academic register, dry humour once per major section maximum
- [ ] Section headings: descriptive (state the conclusion, not just the topic)
- [ ] Dataset: California Housing only — no synthetic data except toy 3–5 row subsets derived from real features
