# Reference Books — Neural Chronicles Authoring Guide

> **Read this before starting or continuing any chapter.**
> Contains everything needed to maintain tone, structure, and consistency.

---

## CURRENT STATE — Page Inventory

| Page | Content | Status |
|---|---|---|
| 1 | Cover — title + regression scatter SVG | ✅ Done |
| 2 | Prologue — three data types, compatibility table, bridge SVGs | ✅ Done |
| 3 | Guide Overview — chapter map table (now 17 chapters), hyperparameter pattern, usage guide | ✅ Done |
| 4 | Ch.1 Splash — dark page, regression SVG, subtitle + keywords | ✅ Done |
| 5 | Ch.1 p1 — The Model (ŷ=wx+b, weighted sum, Dense(1) connection) | ✅ Done |
| 6 | Ch.1 p2 — Cost Functions & Metrics (MSE, MAE, RMSE, R², bar chart SVG, table) | ✅ Done |
| 7 | Ch.1 p3 — Gradient Descent (loss bowl SVG, learning rate curves SVG) | ✅ Done |
| 8 | Ch.1 p4 — TF Code + 3-column checklist + bridge to Ch.2 | ✅ Done |
| 9 | Ch.1 p5 — Interview Q&A + traps + assumptions + decision SVG | ✅ Done |
| 10–15 | Ch.2 — Logistic Regression | ✅ Done |
| 16–21 | Ch.3 — The XOR Problem | ✅ Done |
| 22–27 | Ch.4 — Neural Networks | ✅ Done |
| 28–33 | Ch.5 — Backprop & Optimisers | ✅ Done |
| 34–39 | Ch.6 — Regularisation | ✅ Done |
| 40–45 | Ch.7 — CNNs | ✅ Done |
| 46–51 | Ch.8 — RNNs / LSTMs / GRUs | ✅ Done |
| 52–57 | Ch.9 — Metrics Deep Dive | ✅ Done |
| 58–63 | Ch.10 — Classical Classifiers (DT, KNN, comparison table, sklearn code, interview) | ✅ Done |
| 64–69 | Ch.11 — SVM & Ensembles (margin, kernel trick, bagging vs boosting, XGBoost, interview) | ✅ Done |
| 70–75 | Ch.12 — Clustering (K-Means, DBSCAN, HDBSCAN, feature engineering) | ✅ Done |
| 76–81 | Ch.13 — Dimensionality Reduction (PCA, t-SNE, UMAP) | ✅ Done |
| 82–87 | Ch.14 — Unsupervised Metrics (Silhouette, Davies-Bouldin, ARI, EVR, reconstruction error) | ✅ Done |

**The HTML source (Neural_Chronicles.html) covers Ch.1–Ch.14. Ch.15 (MLE & Loss Functions), Ch.16 (TensorBoard), and Ch.17 (Transformers & Attention) are notes-library-only chapters written beyond the original HTML book — 17 chapters total.**

> **Actual HTML line counts (as of last edit):**
> - Ch.1 Linear Regression: ~lines 369–1107
> - Ch.2 Logistic Regression: ~lines 1107–1678
> - Ch.3 XOR Problem: ~lines 1678–2187
> - Ch.4 Neural Networks: ~lines 2187–2750
> - Ch.5 Backprop & Optimisers: ~lines 2751–3350
> - Ch.6 Regularisation: ~lines 3350–3730
> - Ch.7 CNNs: ~lines 3730–4150
> - Ch.8 RNNs / LSTMs / GRUs: ~lines 4150–4350 (not recorded precisely)
> - Ch.9 Metrics Deep Dive: not recorded
> - Ch.10 Classical Classifiers: ~lines 4150–4350
> - Ch.11 SVM & Ensembles: ~lines 4350–4550 (file ends here)
> - Ch.12–Ch.14: covered in HTML but line ranges not recorded


### File locations
- **Neural Chronicles HTML:** `notes/Reference/Neural_Chronicles.html`
- **Neural Chronicles PDF:** `notes/Reference/Neural_Chronicles.pdf`
- **ML Chronicles HTML:** `notes/Reference/ML_Chronicles.html`
- **ML Chronicles PDF:** `notes/Reference/ML_Chronicles.pdf`
- **Regenerate PDF:** `cd pdf-gen && node generate-pdf.mjs neural`

---

## BOOK IDENTITY & TONE

### What this book is
A **diagram-first technical interview reference** for deep learning. 14 chapters covering linear regression through unsupervised learning. Each chapter ends with a three-column interview checklist and a dedicated Q&A page.

### What it is NOT
- Not a narrative/story. No characters. No fictional framing. An earlier version had a manga story arc (Kai, the Neural Realm, Professor Regress, etc.) — that has been completely removed.
- Not a textbook. No exhaustive proofs. No citation chains.
- Not a tutorial. Code is illustrative, not copy-paste production.

### Writing voice
Direct, confident, occasionally dry. The voice assumes the reader is smart and time-constrained — someone reading the night before an interview. Write like explaining to a colleague, not lecturing to a class.

- ✅ "MSE squares errors — a 10× error costs 100× as much."
- ✅ "Plain R² always inflates with more features. Use Adjusted R²."
- ❌ "Let us explore the fascinating world of gradient descent together!"
- ❌ "Kai descended into the loss landscape..."

### Primary use case
Read the night before a machine learning or data science interview. Each chapter 5-page cycle covers one concept group completely: the math, the code, the metrics, the hyperparameter dials, and the specific questions interviewers ask.

---

## CSS CLASSES — COMPLETE REFERENCE

### Layout classes
| Class | Use |
|---|---|
| `.page` | One A4 printed page (210×297mm, 10mm padding). Every page is a `<div class="page">`. |
| `.splash` | Dark chapter title page. Contains `.ctag`, `<h1>`, subtitle div, large SVG. |
| `.flow` | Vertical flex column — the default content container inside `.page`. |
| `.cols .c2` | Two equal columns. |
| `.cols .c21` | Two columns: left 2× wider than right. |
| `.cols .c12` | Two columns: right 2× wider than left. |
| `.cols .c3` | Three equal columns. Used for interview checklists. |
| `.ch-nav` | Top-right page header. Pattern: `CH.N — TOPIC · SUBTOPIC` |

### Content box classes
| Class | Color | Use |
|---|---|---|
| `.box` | White, grey border | General containers, tables, checklists |
| `.box.tint` | Light blue #f0f7ff | Math term breakdowns, definitions, assumption lists |
| `.box.sepia` | Warm cream #f5f0e0 | **Interview Q&A pairs** — question in `<h4>`, answer in `<p class="sm">` |
| `.note` | Blue left border (--arch) | Historical facts, technical insights, connections between concepts |
| `.lock` | Gold left border (--gold) | Key insights, "bridge facts" interviewers explicitly test |
| `.tip` | Green left border (--regress) | Practical rules of thumb, starting values |
| `.warn` | Orange left border #e67e22 | Common mistakes, traps, gotchas |

### Special element classes
| Class | Use |
|---|---|
| `.diagram` | Wrapper for `<figure>` + SVG. Always include `<figcaption>`. |
| `.bridge` | Dark gradient box at chapter end. Summary + preview + connecting concept. |
| `.code-block` | Dark container for `<pre>` code. Green text on dark background. |
| `.math-block` | Centered KaTeX equation block in light blue. |
| `.ctag` | Uppercase gold label (e.g., "Chapter 1", "Guide Overview"). |
| `.dial` | Gold pill badge for hyperparameter names inline in text. |
| `.pq` | Pull quote — **NO LONGER USED**. Do not add new `.pq` elements. |

### CSS color variables
```
--regress: #27ae60   green      Ch.1–2  regression chapters
--arch:    #2980b9   blue       Ch.4    architecture (neural networks)
--forge:   #e74c3c   red        Ch.5    training / optimizers
--reg:     #8e44ad   purple     Ch.6    regularisation
--conv:    #16a085   teal       Ch.7    CNNs
--mem:     #2471a3   dark blue  Ch.8    RNNs / LSTMs
--gold:    #f39c12   gold       all     dials, highlights, borders
```

### Typography reference
| Element | Size | Note |
|---|---|---|
| `h2` | 1.1rem, bold, gold bottom border | Section headers inside a page |
| `h3` | 0.95rem | Sub-section headers |
| `h4` | 0.82rem, dark | Box and panel titles |
| `p` | 0.78rem, 1.55 line-height | Body text |
| `.sm` | 0.72rem, muted | Secondary text, list items in boxes |

---

## PAGE STRUCTURE TEMPLATE

Each chapter = **1 splash + 4 content pages + 1 interview page = 6 pages total.**

```
Splash    dark .page + .splash: ctag, h1 (topic name), subtitle (keywords), large SVG
Page 1    [THE MATH]          core equation breakdown, annotated diagram SVG, .box.tint term table
Page 2    [CORE CONCEPT]      the second most important idea of the chapter — varies by chapter:
                              Ch.1: loss functions & metrics | Ch.2: metrics & threshold
                              Ch.3: UAT | Ch.4: activation gallery | Ch.5: optimiser progression
                              Ch.6: regulariser mechanisms | Ch.7: pooling & architecture
                              Ch.8: LSTM gates & BPTT | Ch.9: metric families
Page 3    [TRAINING]      training dynamics, hyperparameter dials, learning curve SVG
Page 4    [TF CODE]       full working code + training loop steps + 3-col checklist + .bridge
Page 5    [INTERVIEW]     5x .box.sepia Q&A + .warn traps + .box.tint assumptions + .lock + decision SVG
```

### ch-nav pattern (top-right corner of each page)
- Splash: `CH.N — TOPIC NAME`
- Content pages 1–4: `CH.N — TOPIC NAME · SUBTOPIC`
- Interview page: `CH.N — TOPIC NAME · INTERVIEW PREP`

### .bridge format (always at end of page 4)
1. What this chapter established (one clause)
2. What the next chapter adds (one clause)
3. The specific connecting concept (named explicitly)

---

## SVG CONVENTIONS

- All SVGs are **inline**, inside `<figure class="diagram">` with `<figcaption>`.
- Use `width="100%"` and explicit `viewBox`.
- **No `<marker>` elements** — they conflict across SVGs. Use inline `<polygon>` for arrowheads.
- Every SVG should answer a specific question that prose alone cannot efficiently answer.
- Color: chapter color variable for highlights, `#333` for text, `#e0e0e0` for grid lines.
- Font: `font-family="Georgia,serif"`, body `font-size="9"`, labels `font-size="8"`, titles `font-size="10"`.

---

## THE HYPERPARAMETER DIAL PATTERN

Every chapter introduces the "dial" — the main tunable hyperparameter. Always label it explicitly.

| Ch. | Dial | Too-high effect | Too-low effect | Typical start |
|---|---|---|---| ---|
| 1 | Learning rate α | Oscillates / diverges | Painfully slow | 0.1 (SGD), 1e-3 (Adam) |
| 2 | Decision threshold | High Precision, low Recall | High Recall, low Precision | 0.5 |
| 3 | Hidden layers | Vanishing gradient risk | Can't solve non-linear problems | 1–2 |
| 4 | Layer depth | Vanishing gradient risk | Insufficient expressivity | 3–5 layers |
| 4 | Layer width | Overparameterised | Underfits | 64–256 |
| 5 | LR schedule / optimizer | Chaos | Saddle point stall | Adam default |
| 6 | Dropout rate | Underfitting | No regularisation | 0.2–0.4 |
| 6 | L2 λ | Underfitting | No regularisation | 1e-4 to 1e-2 |
| 6 | Patience | Stops too early | Overfits | 10 |
| 7 | Filter count | Slow, may overfit | Misses fine features | 32→64→128 |
| 8 | LSTM units | Overfit | Forgets long context | 64–128 |

---

## INTERVIEW PREP CONVENTIONS

**3-column checklist** (end of page 4 / TF code page):
```html
<div class="box" style="border-top:3px solid var(--CHAPTER-COLOR);">
  <h4>Chapter N — Core Interview Checklist</h4>
  <div class="cols c3">
    <div><p>The Model</p><ul class="sm">...</ul></div>
    <div><p>The Metrics</p><ul class="sm">...</ul></div>
    <div><p>The Training</p><ul class="sm">...</ul></div>
  </div>
</div>
```

**Interview Q&A page** (page 5 of each chapter):
- 5 `.box.sepia` Q&A pairs — question as `<h4>`, answer as `<p class="sm">`. Address edge cases, not definitions.
- `.warn` box: 4–5 common traps — things smart candidates get wrong.
- `.box.tint`: assumptions of the model — always include; interviewers love testing these.
- `.lock` box: the bridge fact connecting this chapter to adjacent concepts.
- Decision SVG at bottom: quick visual flow for key practical choices.

---

## MATHEMATICS CONVENTIONS

- KaTeX auto-renders `$$...$$` (display) and `$...$` (inline).
- Always follow a display equation with a term-by-term breakdown as a bullet list or `.box.tint` table.
- Prefer matrix notation: `\mathbf{w}^\top \mathbf{x} + b`
- HTML text subscripts/superscripts: use entities (`&sup2;`) not KaTeX.

---

## COMPLETED CHAPTERS — CONTENT SUMMARY

### Ch.1 — Linear Regression (Pages 4–9)

**Accent color:** `var(--regress)` #27ae60

**Core equation:** ŷ = w·x + b

**Covered:**
- Weighted sum model, residuals, Dense(1) = linear regression (the neural net bridge)
- MSE, MAE, RMSE, R² — born in Ch.1, defined with formulas and comparison table
- Gradient descent: w ← w − α·∂L/∂w. Loss bowl SVG, learning rate curves SVG.
- Full TF code: Dense(1), SGD optimizer, 200 epochs, evaluate all 4 metrics, inspect learned weights
- Interview Q&A: R²<0, MSE vs MAE, RMSE vs MSE reporting, batch/step/epoch, Dense(1) bridge
- Traps: comparing R² across feature counts, reporting raw MSE, multicollinearity, feature scaling

**Key facts carried forward:**
- Every neuron computes z = w·x + b — this is the core operation of all deep learning
- MSE and MAE first appear here and are referenced in every subsequent chapter
- Dense(1) + no activation = linear regression = the simplest neural network

---

### Ch.2 — Logistic Regression (Pages 10–15)

**Accent color:** `var(--regress)` #27ae60

**Covered:**
- **Splash:** Mathematically computed sigmoid polyline SVG (actual σ(z) at half-unit steps from z=−6 to +6), decision boundary annotated at σ(0) = 0.5.
- **Page 1 (The Model):** ŷ = σ(w·x+b). Log-odds interpretation: ln(ŷ/(1−ŷ)) = w·x+b. Why logistic regression is a linear classifier (decision boundary w·x+b=0 is a hyperplane). Why MSE fails for binary classification: non-convex loss surface with sigmoid; gradient vanishes when ŷ near 0/1; BCE+sigmoid is convex with gradient ŷ−y (no vanishing term). Ch.1→Ch.2 comparison table (output, range, loss, task). σ'(z) = σ(z)(1−σ(z)), max 0.25 at z=0 — establishes why sigmoid becomes ReLU in hidden layers (Ch.4).
- **Page 2 (Metrics):** BCE formula in KaTeX with four-case breakdown (y=1/ŷ→1, y=1/ŷ→0, y=0/ŷ→0, convex). Accuracy paradox: 95% class imbalance → 95% accuracy by always predicting 0. Precision, Recall, F1 formulas in KaTeX. AUC-ROC: P(model ranks positive higher than negative), threshold-independent.
- **Page 3 (Training):** Decision threshold as the chapter Dial. PR curve vs ROC curve: PR preferred for severe imbalance. AUC interpretation. Threshold-moving and its effect on Precision/Recall.
- **Page 4 (Code + Checklist):** Dense(1, 'sigmoid') + binary_crossentropy + AUC metric. Softmax multiclass extension. 3-column checklist. Bridge to Ch.3.
- **Page 5 (Interview):** Q&As: why not MSE for binary, accuracy paradox, is logistic regression a linear classifier, AUC-ROC interpretation, sigmoid vs softmax.

**Key facts carried forward:**
- BCE gradient ∂L/∂z = ŷ − y — no saturation term, unlike MSE+sigmoid
- Decision boundary w·x+b=0 is a hyperplane — logistic regression is always a linear classifier
- Accuracy is misleading under class imbalance; use F1 or AUC
- σ'(z) = σ(z)(1−σ(z)), max 0.25 at z=0 — saturation motivates ReLU in hidden layers (Ch.4)

---

### Ch.3 — The XOR Problem (Pages 16–21)

**Accent color:** `#7f8c8d` grey

**Covered:**
- **Splash:** XOR dataset SVG showing four corner points with no linear separator, plus correct labels; chapter keywords row.
- **Page 1 (The Collapse Proof):** Linear collapse: W₂(W₁x) = (W₂W₁)x. By induction, n stacked linear layers = one linear transformation. Non-linearity is the only escape. Historical context: Minsky-Papert 1969 proved single-layer perceptrons cannot solve XOR; the field wrongly generalised to "neural networks fail", triggering the first AI winter.
- **Page 2 (UAT):** Universal Approximation Theorem: one hidden layer with enough neurons approximates any continuous function on a compact domain. UAT guarantees expressibility only — not learnability, not efficiency, not that gradient descent finds the solution.
- **Page 3 (Depth vs Width):** Decision region visualisation SVGs (0 hidden layers = linear only; 1 hidden = piecewise linear; 3 hidden = complex regions). Depth → hierarchical composition, exponentially more parameter-efficient for structured data. Width → parallel features at one abstraction level, safer gradient flow. Full comparison table.
- **Page 4 (Code + Checklist):** XOR dataset (4 points), Dense(4, relu) + Dense(1, sigmoid), Adam lr=0.05, 1000 epochs. Init sensitivity warning. Bridge to Ch.4.
- **Page 5 (Interview):** Q&As: XOR problem definition, prove two linear layers = one, what UAT guarantees and does NOT guarantee, AI winter cause, width vs depth trade-off.

**Key facts carried forward:**
- W₂(W₁x) = W_combined·x — the linear collapse, tested directly in interviews
- UAT guarantees expressibility, not learnability — different questions
- Depth is exponentially more parameter-efficient for hierarchical data (vision, language)
- The Rumelhart 1986 backprop paper (Ch.5) solved XOR through hidden layers and ended the AI winter

---

## COMPLETED CHAPTERS — ADDITIONAL SUMMARIES

### Ch.4 — Neural Networks (Pages 22–27)

**Accent color:** `var(--arch)` #2980b9

**Covered:**
- **Page 1 (The Neuron):** z = w·x + b → a = g(z). Term-by-term breakdown. Dense(N) as N parallel neurons = matrix multiply. Forward pass stated for full 2-layer net. Bridge from Ch.3 (stacked linears = one linear, non-linearity breaks the collapse).
- **Page 2 (Activation Gallery):** Activation curves SVG (ReLU, Sigmoid, Tanh, Softmax, GELU). Full comparison table: formula, range, use-case, vanishing-gradient risk. Dead neuron problem explained with `.warn`: z ≤ 0 → gradient = 0 → weight never updates. Fix: Leaky ReLU, He init, lower LR.
- **Page 3 (Weight Initialisation):** Why zeros fail (symmetry problem — all neurons identical). Why naive N(0,1) fails (variance compounds as n^L — explosion or vanishing). Derived fix: scale by fan-in. He Normal (√(2/n_in), for ReLU), Xavier/Glorot (√(2/(n_in+n_out)), for sigmoid/tanh), LeCun (√(1/n_in), for SELU). Variance propagation SVG showing three regimes. Init lookup table with Keras names.
- **Page 4 (Code + Checklist):** Output layer design table (task → activation → loss → metric). Depth and width `.dial` badges. Full Keras code: Sequential, he_normal, softmax output, sparse_categorical_crossentropy, Adam, fit/evaluate, weight std inspection. 3-column checklist. Bridge to Ch.5.
- **Page 5 (Interview):** 5× Q&A (ReLU vs sigmoid, He init derivation, 10-class output, dead neuron detection, sigmoid vs softmax). `.warn` traps: sigmoid in hidden layers, zero init, unscaled N(0,1), softmax for multi-label, missing input_shape. `.box.tint` assumptions. `.lock` bridge to Ch.5. Output layer decision SVG.

**Key facts carried forward:**
- He init std = √(2/n_in) — the factor 2 is for ReLU's ~50% zero output
- Softmax: mutually exclusive classes. Sigmoid: independent binary outputs (including multi-label)
- Dead neuron is a silent failure — no error, no warning; inspect weight stats to detect
- Forward pass is complete; Ch.5 covers the backward pass (backprop + optimisers)

---

### Ch.5 — Backpropagation & Optimisers (Pages 28–33)

**Accent color:** `var(--forge)` #e74c3c

**Covered:**
- **Splash:** Computation graph SVG with forward pass (red solid arrows: x→z→a→L) and backward pass (gold dashed arrows: ∂L/∂a → ∂L/∂z → ∂L/∂w).
- **Page 1 (Chain Rule):** Core equation ∂L/∂w = (∂L/∂a)·(∂a/∂z)·(∂z/∂w) in KaTeX with full term breakdown. Annotated computation graph SVG. Two-pass model: forward stores z and a; backward reuses them. Efficiency insight: all gradients in O(1 forward pass) cost.
- **Page 2 (Optimisers):** Full update rule equations for SGD, SGD+Momentum, RMSProp, Adam in KaTeX. Adam = momentum (m) + RMSProp (s) + bias correction (m̂, ŝ). Convergence path SVG: SGD zigzags, Momentum smooths, Adam goes direct. Comparison table (memory, adaptive LR, bias correction, starting LR). Practical rule: Adam lr=1e-3 is the universal default.
- **Page 3 (Training Dynamics):** Learning rate dial table (too high/low/right). LR schedules: step decay, cosine annealing formula, ReduceLROnPlateau, warmup. Batch size effects with linear scaling rule. Gradient explosion mechanism: backprop multiplies L Jacobians; if spectral radius of weight matrices > 1, gradient norms grow as σ_max^L — exponential in depth and symmetric to vanishing gradient. He/Xavier init prevents it from the start; gradient clipping (rescale when ‖g‖ > τ, `clipnorm=1.0`) handles it during training. Saddle points. Loss curve SVG (four regimes).
- **Page 4 (Code + Checklist):** Full Keras code: Adam, Adam+clipnorm, SGD+Nesterov, CosineDecay schedule, ReduceLROnPlateau callback, LR history logging. Training loop explained step-by-step. 3-column checklist. Bridge to Ch.6.
- **Page 5 (Interview):** 5× Q&As (backprop in one sentence, why Adam beats SGD, gradient clipping, saddle points, momentum geometrically). Traps: local minima vs saddle points, bias correction vs regularisation, clipping vs LR, patience too small, large batch without LR scaling. Assumptions list. Lock. Optimiser selection decision SVG.

**Key facts carried forward:**
- Backprop: all ∂L/∂w in one backward sweep; cost ≈ 2× forward pass
- Adam = momentum + RMSProp + bias correction; β₁=0.9, β₂=0.999, lr=1e-3
- Saddle points (not local minima) are the main obstacle; momentum bypasses them
- Gradient clipping: rescale g when ‖g‖ > threshold; mandatory for RNNs (Ch.8)
- Training now produces fitted weights; Ch.6 addresses the generalisation gap (regularisation)

---

## UPCOMING CHAPTERS — FULL CONTENT PLANS

### Ch.6 — Regularisation (Pages 34–39)

**Accent color:** `var(--reg)` #8e44ad
**ch-nav prefix:** `CH.6 — REGULARISATION`

**Core:** Train/val gap = overfitting. Every regulariser closes the gap by a different mechanism.

**Key techniques:**
- L1: diamond constraint → sparse weights (feature selection)
- L2: sphere constraint → shrinks all weights, none to zero
- Dropout: masks during training, full weights at inference
- BatchNorm: normalise layer inputs per mini-batch (μ=0, σ=1), then apply learnable scale γ and shift β. Stabilises the activation distribution across layers, reducing sensitivity to weight scale → indirect fix for both gradient explosion and vanishing gradient through depth. Also acts as a mild regulariser (noise from mini-batch statistics).
- Early Stopping: patience + restore_best_weights
- Label smoothing: replaces hard 0/1 targets with (ε/K, 1−ε+ε/K) — prevents the model from becoming overconfident on training labels; standard for softmax outputs in production models.
- Data augmentation: artificially expands training distribution via transforms (flip, crop, rotation for images). Regularises by preventing memorisation of specific input patterns. Covered conceptually here; implementation in Ch.7.

**Interview Q&A topics:**
- L1 vs L2: which produces sparse weights? (L1 — diamond corners land on axes.)
- Dropout at inference? (Disabled; weights scaled by (1−p). Handled automatically by TF.)
- Vanishing gradient causes and fixes? (Vanishing: sigmoid derivative max 0.25 multiplied through L layers → exponential shrinkage. Exploding: singular values > 1 multiplied through L layers → exponential growth — same chain rule, opposite direction. Fixes: ReLU + He init address both at initialisation; gradient clipping handles explosion mid-training; BN stabilises activation scale; residual connections bypass the problem structurally in Ch.7.)
- L2 = Ridge regression? (Yes — identical form, same mathematical effect.)
- Dropout vs BatchNorm? (Not exclusive: BN for training stability, Dropout for overfit. Often both.)

**The Dial:** dropout rate (0.2–0.5), L2 λ (1e-4 to 1e-2), patience (5–50)

---

### Ch.7 — CNNs (Pages 40–45)

**Accent color:** `var(--conv)` #16a085
**ch-nav prefix:** `CH.7 — CONVOLUTIONAL NEURAL NETWORKS`

**Core operation:** (I * K)[i,j] = Σ Σ I[i+m, j+n]·K[m,n]

**Architecture genealogy:** LeNet → AlexNet → VGG → ResNet (F(x)+x skip connections)

**Interview Q&A topics:**
- Why not dense for images? (No parameter sharing: 38M params from one layer of 224×224×3 input.)
- What does pooling do? (Spatial downsampling, translation invariance for features.)
- How do residual connections solve vanishing gradient? (Gradient flows through identity shortcut F(x)+x.)
- What is parameter sharing? (Same filter applied at all spatial positions.)
- Stride vs pooling? (Stride reduces size during convolution; pooling after activation. Modern nets prefer strided convs.)
- Output size after conv? (floor((n + 2p − f) / s) + 1. n=input size, p=padding, f=filter size, s=stride. Memorise this — interviewers ask it.)
- Receptive field growth? (Each conv layer sees a window of the previous feature map. Stacking layers compounds this — a 3-layer net with 3×3 filters has a 7×7 receptive field on the input. Deep CNNs see globally despite local filters.)

**The Dial:** filter count, kernel size, number of conv blocks

---

### Ch.8 — RNNs / LSTMs / GRUs (Pages 46–51)

**Accent color:** `var(--mem)` #2471a3
**ch-nav prefix:** `CH.8 — RNNS / LSTMS / GRUS`

**Core:** LSTM cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

**Interview Q&A topics:**
- Why do RNNs fail on long sequences? (Multiplicative gradient decay through W_h. LSTM: additive cell updates preserve gradient.)
- Three LSTM gates? (Forget: what fraction of C_{t-1} to keep. Input: what to write from x_t. Output: what to reveal as h_t.)
- GRU vs LSTM? (GRU: 33% fewer params, faster. LSTM marginal advantage on very long sequences. Try GRU first.)
- What is bidirectional LSTM? (Runs LR and RL simultaneously; concatenates hidden states. Needs full sequence.)
- RNN vanishing gradient vs deep network vanishing gradient? (RNN: same W_h multiplied at every step. Deep net: different weights but sigmoid saturation. Different problems, different fixes.)
- What is BPTT? (Backpropagation Through Time — unfold the RNN across T steps, treat as a T-layer deep net with shared weights, apply chain rule. Gradient ∂L/∂W_h involves a product of T Jacobians — this is why it explodes/vanishes. Truncated BPTT limits unrolling to k steps to make the cost tractable.)

**The Dial:** LSTM units, stacked layers, dropout between recurrent layers, sequence length

---

### Ch.9 — Metrics Deep Dive (Pages 52–57)

> **Positioning note:** Ch.9 is the capstone of the deep learning arc (Ch.1–8). It expands the metric vocabulary introduced across previous chapters into a unified reference. It also bridges into the classical ML section (Ch.10–14) — every metric here applies to classical models too. A reader preparing for a general ML interview can read Ch.9 standalone; BLEU and Perplexity are NLP-specific and relevant mainly after Ch.8.

**Accent color:** `var(--gold)` #f39c12
**ch-nav prefix:** `CH.9 — METRICS DEEP DIVE`

**Core:** Every metric encodes a judgment about which type of wrongness matters most.

**Metrics covered:** MSE/MAE/RMSE/R² (Ch.1 recap), Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, BLEU, Perplexity, Log-loss, Calibration

**Interview Q&A topics:**
- Precision vs Recall trade-off and when each matters more.
- When is PR-AUC better than ROC-AUC? (Severe class imbalance — ROC artificially flatters due to high TN count.)
- BLEU score — what it measures and where it fails. (N-gram overlap; misses semantic meaning.)
- What is perplexity? (2^H per token. Lower better. Not cross-model comparable.)
- What is calibration? (70% confidence → 70% accuracy. Fix: temperature scaling.)

**The Dial:** classification threshold (moves Precision/Recall operating point)

---

## STYLE RULES SUMMARY

1. No story arc, no characters. Historical facts go in `.note` boxes.
2. Key insights go in `.lock` boxes — written as facts, not quotes.
3. Interview traps always in `.warn` boxes — name the specific wrong assumption.
4. Q&A pairs use `.box.sepia` — `<h4>` question, `<p class="sm">` answer addressing edge cases.
5. All equations in KaTeX. All code in `.code-block > pre`. TF/Keras only with comments.
6. Every SVG in `<figure class="diagram">` with `<figcaption>`. No `<marker>` — use `<polygon>`.
7. Bridge at end of every page 4: summary + preview + connecting concept, in one `.bridge` div.
8. Hyperparameter Dial labeled with `.dial` span: too-high consequence, too-low consequence, starting value.
9. Every page must fit in 297mm height. Compress with inline `style="font-size:.7rem"` if needed; do not overflow.

---

## IMAGE CONVENTIONS

Raster images (`.png`) are used for diagrams too complex or spatially rich for inline SVG.

- **Folder:** `notes/ML/img/`
- **Reference in HTML:** `<img src="img/FILENAME.png" style="width:100%;border-radius:4px;" alt="DESCRIPTION">`
- **Wrap in:** `<figure class="diagram"><img ...><figcaption>...</figcaption></figure>`
- **Naming convention:** `chN-concept-slug.png` — e.g. `ch6-dropout-mask.png`, `ch8-lstm-cell.png`
- **Background:** white or very light (#fafbff) to match the page background
- **Style:** flat design, technical diagram, no decorative elements, high contrast for print

### Planned image assets

**PNG files** (`notes/ML/img/`) — AI-generated, reviewed and approved:

| File | Chapter | Content |
|---|---|---|
| `ch6-l1-l2-geometry.png` | Ch.6 Page 1 | L1 diamond vs L2 sphere constraint region with loss contours ✅ |
| `ch6-dropout-mask.png` | Ch.6 Page 2 | Network with masked neurons (training) vs full (inference) ✅ |
| `ch6-batchnorm-distribution.png` | Ch.6 Page 2 | Before/after BN: skewed → μ=0,σ=1 → γ,β re-scaled ✅ |
| `ch7-residual-block.png` | Ch.7 Page 3 | F(x)+x skip connection with gradient flow annotation ✅ |
| `ch8-rnn-unrolled.png` | Ch.8 Page 1 | RNN unrolled across T steps showing same W_h at each timestep ✅ |

**Inline SVGs** — write directly in HTML when authoring the chapter (AI generation produced incorrect labels):

| Diagram | Chapter | Notes |
|---|---|---|
| Convolution operation | Ch.7 Page 1 | 6×6 input grid, highlighted 3×3 region, orange filter, green output cell |
| Feature hierarchy | Ch.7 Page 2 | 4 boxes: edge strokes → checkerboard → oval+arc → head silhouette |
| LSTM cell | Ch.8 Page 1 | Gate stack (forget/input/candidate/output), cell state path C_t, h_t output |

---

## NEW CSS COLOR VARIABLES (Ch.10–14)

Add these to `:root` in the HTML `<style>` block:

```
--class:  #c0392b   crimson    Ch.10  classical classifiers (DT, KNN)
--ens:    #784212   amber      Ch.11  SVM & ensembles (XGBoost)
--clust:  #117a65   emerald    Ch.12  clustering (K-Means, DBSCAN, HDBSCAN)
--dimred: #1a5276   navy       Ch.13  dimensionality reduction (PCA, t-SNE, UMAP)
--eval2:  #6c3483   violet     Ch.14  unsupervised metrics
```

## NEW CHAPTERS — CLASSICAL ML & UNSUPERVISED LEARNING (Ch.10–14)

These chapters extend the guide beyond deep learning into classical ML and unsupervised learning.
They cover topics that appear frequently in data science interviews alongside the deep learning questions.

**Page numbers (after Ch.1–9 are written):**
- Pages 58–63: Ch.10 Classical Classifiers
- Pages 64–69: Ch.11 SVM & Ensembles
- Pages 70–75: Ch.12 Clustering
- Pages 76–81: Ch.13 Dimensionality Reduction
- Pages 82–87: Ch.14 Unsupervised Metrics

---

### Ch.10 — Classical Classifiers (Pages 58–63)

**Accent color:** `var(--class)` #c0392b
**ch-nav prefix:** `CH.10 — CLASSICAL CLASSIFIERS`
**Connection to Ch.9:** Metrics from Ch.9 (Accuracy, F1, AUC) apply directly to all classifiers here.

**Covered:**
- Decision Trees: Gini impurity, information entropy, information gain, greedy splitting, axis-aligned boundaries
- KNN: Minkowski distance, k selection, curse of dimensionality, O(nd) predict cost
- DT vs Logistic Regression vs KNN: full comparison table + decision flowchart

**Dials:**
- DT: max_depth (3–8), min_samples_leaf (1–10), criterion (gini/entropy)
- KNN: k (√n start), metric (euclidean default), weights (uniform/distance)

**Interview Q&A topics:**
- Gini vs entropy — which is better? (Near-identical trees. Gini faster — no log. criterion matters less than max_depth.)
- Why does DT overfit? Fix? (Grows until pure leaves. Fix: max_depth, min_samples_leaf, or use ensembles.)
- Why scale before KNN? (Euclidean distance dominated by large-scale features.)
- DT vs LR — when does each win? (LR: linear boundary, calibrated probs. DT: interactions, mixed types, missing values.)
- KNN as k → n? (Predicts majority class — max bias. k → 1: memorises training — max variance.)

**Bridge to Ch.11:** DT is the building block of Random Forest (bagging) and XGBoost (boosting). Ch.11 covers both.

---

### Ch.11 — SVM & Ensembles (Pages 64–69)

**Accent color:** `var(--ens)` #784212
**ch-nav prefix:** `CH.11 — SVM & ENSEMBLES`

**Covered:**
- SVM: maximum-margin hyperplane, support vectors, C parameter (soft margin), ε-SVR for regression
- Kernel trick: RBF (γ), polynomial, linear — mapping to higher-dimensional space implicitly
- Bagging vs Boosting: conceptual comparison — variance reduction vs bias reduction
- Random Forest: bootstrap aggregation, feature subsampling at each split, out-of-bag error
- XGBoost: gradient boosting, additive tree model, η, n_estimators, max_depth, subsample, L1/L2 regularisation

**Dials:**
- SVM: C (large C = small margin = overfit risk), γ for RBF (large γ = overfit)
- XGBoost: n_estimators (100–1000), learning_rate η (0.01–0.3), max_depth (3–8), subsample (0.6–0.9)

**Interview Q&A topics:**
- What is the kernel trick? (Maps data to high-d space implicitly via K(x,x') — no explicit transformation.)
- C in SVM — what does it do? (Large C: hard margin, overfit risk. Small C: wide margin, more misclassifications tolerated.)
- Bagging vs boosting? (Bagging: parallel trees, reduces variance. Boosting: sequential, corrects errors, reduces bias.)
- Why XGBoost over Random Forest? (Often more accurate; built-in L1/L2 reg; handles missing values; histogram binning.)
- What is ε-insensitive loss in SVR? (Ignores errors < ε. Only support vectors outside the tube define the model.)

**Bridge to Ch.12:** SVM and XGBoost require labelled data. Ch.12 covers unsupervised learning — structure without labels.

---

### Ch.12 — Unsupervised Learning: Clustering (Pages 70–75)

**Accent color:** `var(--clust)` #117a65
**ch-nav prefix:** `CH.12 — CLUSTERING`

**Covered:**
- K-Means: centroid update (Lloyd's algorithm), inertia, elbow method, k-Means++ initialisation
- DBSCAN: ε (radius), min_samples, core/border/noise classification, arbitrary shapes, outlier detection
- HDBSCAN: hierarchical density clustering, no ε parameter, soft cluster membership, varying density
- Feature engineering via clustering: cluster assignments as categorical features; distance-to-centroid as numeric features

**Dials:**
- K-Means: n_clusters k (elbow/silhouette), n_init (10), max_iter (300)
- DBSCAN: ε (k-distance graph), min_samples (2×d starting point)
- HDBSCAN: min_cluster_size (5–50), min_samples (1–15)

**Interview Q&A topics:**
- K-Means vs DBSCAN — when does each fail? (K-Means: non-spherical clusters, outlier sensitivity. DBSCAN: varying density, ε tuning.)
- DBSCAN core / border / noise? (Core: ≥ min_samples within ε. Border: within ε of core, fewer than min_samples itself. Noise: neither.)
- How to choose k for K-Means? (Elbow method: plot inertia vs k. Silhouette score: pick k maximising average width.)
- HDBSCAN vs DBSCAN? (HDBSCAN: no ε; builds hierarchy, extracts stable clusters; handles varying density; soft membership.)
- Feature engineering with clustering? (Cluster labels as categorical; distance to each centroid as numeric; membership probabilities for downstream supervised model.)

**Bridge to Ch.13:** Clustering groups points in original space. Dimensionality reduction projects into lower-d space, revealing global structure not visible in original dimensions.

---

### Ch.13 — Dimensionality Reduction (Pages 76–81)

**Accent color:** `var(--dimred)` #1a5276
**ch-nav prefix:** `CH.13 — DIMENSIONALITY REDUCTION`

**Covered:**
- PCA: eigenvector decomposition, explained variance ratio, reconstruction error, linear assumption
- t-SNE: perplexity, crowding problem, KL divergence minimisation — visualisation only, non-parametric
- UMAP: n_neighbors (global vs local structure), min_dist, manifold assumption, faster + topology-preserving
- Feature selection via dim reduction: PCA loadings, UMAP embeddings as features for downstream models

**Dials:**
- PCA: n_components (choose by explained variance ≥ 0.95), whiten
- t-SNE: perplexity (5–50), learning_rate (200), n_iter (1000+)
- UMAP: n_neighbors (15 default), min_dist (0.1), n_components (2 for viz, higher for features)

**Interview Q&A topics:**
- What does perplexity control in t-SNE? (Effective number of neighbours — local vs global structure balance. Low: tight clusters. High: more global.)
- t-SNE vs UMAP — which to use? (t-SNE: well-established, good cluster separation; slow; local only. UMAP: faster; global topology; transform() for new points.)
- Why can't you use t-SNE coordinates as features? (Non-parametric — can't embed new points without re-running. UMAP has transform().)
- Explained variance ratio? (Fraction of total variance per component. Sum until ≥ 0.95 → n_components needed.)
- PCA assumptions? (Linearity only; variance = signal; sensitive to scale — always standardise before PCA.)

**Bridge to Ch.14:** You've applied unsupervised methods. Ch.14 covers how to evaluate them — no labels means standard accuracy doesn't apply.

---

### Ch.14 — Unsupervised Metrics (Pages 82–87)

**Accent color:** `var(--eval2)` #6c3483
**ch-nav prefix:** `CH.14 — UNSUPERVISED METRICS`

**Covered:**

**Clustering Evaluation:**
- Silhouette Score: (b − a) / max(a, b); a = mean intra-cluster dist, b = mean nearest-cluster dist. Range [−1, 1], higher better.
- Davies-Bouldin Index: avg ratio of within-cluster scatter to between-cluster separation. Lower better.
- Adjusted Rand Index (ARI): agreement between two cluster assignments, corrected for chance. Range [−1, 1]; 1 = perfect, 0 = random. Requires ground-truth labels.

**Dimensionality Reduction Evaluation:**
- Explained Variance Ratio (PCA): fraction of total variance retained per component.
- Reconstruction Error: MSE between original and reconstructed points (PCA, autoencoders).
- Neighborhood Preservation: fraction of k-NN in original space preserved in reduced space (trustworthiness = no false neighbours; continuity = no lost neighbours).

**Dials:**
- Silhouette: no tuning — computed after clustering. Use as model selection criterion for k.
- ARI: requires ground-truth labels — validation only, not operational.

**Interview Q&A topics:**
- Silhouette ≈ 0 means? (Point is on boundary between clusters — ambiguous assignment.)
- When is ARI not useful? (No ground truth labels. Use Silhouette or Davies-Bouldin in purely unsupervised settings.)
- Davies-Bouldin vs Silhouette? (DB: lower better; punishes high within-scatter relative to separation. Silhouette: higher better; intuitive per-sample interpretation.)
- What is neighborhood preservation? (Measures t-SNE/UMAP quality: did k-NN survive the projection? Trustworthiness = no false neighbours introduced; continuity = no real neighbours lost.)
- Reconstruction error vs explained variance? (Reconstruction error: direct distance in original space. Explained variance: scale-free fraction. Both measure information loss; EVR only applies to linear models like PCA.)
