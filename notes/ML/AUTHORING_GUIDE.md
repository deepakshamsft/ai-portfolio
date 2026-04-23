# ML Track — Authoring Guide

> **This document tracks the chapter-by-chapter build of the ML notes library.**  
> Each chapter lives under `notes/ML/` in its own folder, containing a README and a Jupyter notebook.  
> Read this before starting any chapter to keep tone, structure, and the running example consistent.

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
