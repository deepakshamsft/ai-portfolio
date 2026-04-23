# Regression Track

> **The Mission**: Build **SmartVal AI** — a production-grade home valuation system that predicts California house values within $40k MAE while meeting strict business and regulatory requirements.

This is not a Kaggle competition. Every chapter builds toward a single production challenge: you're the Lead ML Engineer at a real estate platform, and the CEO demands a system that agents, lenders, and homebuyers can trust for multi-million-dollar decisions.

---

## The Grand Challenge: 5 Core Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **ACCURACY** | <$40k MAE on median house values | Appraisal regulations require ±20% accuracy ($40k on $200k median). This is the regulatory minimum — we optimize for lowest MAE without overfitting, measured on held-out test districts |
| **#2** | **GENERALIZATION** | Work on unseen districts (CA→nationwide expansion) | Can't memorize training ZIP codes. Must learn true patterns that transfer to new markets |
| **#3** | **MULTI-TASK** | Predict value (regression) AND market segment (classification) | Investors need both price estimates and "High-value coastal" vs "Affordable inland" labels |
| **#4** | **INTERPRETABILITY** | Explainable predictions for non-technical stakeholders | Lending decisions require justifiable valuations (regulatory compliance). "Neural net said so" doesn't fly with underwriters |
| **#5** | **PRODUCTION** | Scale + Monitor: <100ms inference, handle outliers, version control | Research notebooks ≠ production systems. Need MLOps infrastructure for deployment |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | MAE | Constraints | Status |
|----|-------|--------------|-----|-------------|--------|
| **1** | [Linear Regression](ch01-linear-regression/) | Single-feature baseline (MedInc → MedHouseVal) | ~$70k | #1 Partial | ✅ Complete |
| **2** | [Multiple Regression](ch02-multiple-regression/) | All 8 features, vectorization | ~$55k | #1 Improved | ✅ Complete |
| **3** | [Feature Scaling, Importance & Multicollinearity](ch03-feature-importance/) | Univariate R², VIF, permutation importance, collinearity detection | ~$55k | #4 ✅ Interpretability | ✅ Complete |
| **4** | [Polynomial Features](ch04-polynomial-features/) | Non-linear relationships (MedInc², interactions) | ~$48k | #1 Close! | ✅ Complete |
| **5** | [Regularization](ch05-regularization/) | Ridge/Lasso/Elastic Net prevents overfitting | **$38k** | **#1 ✅ #2 ✅** | ✅ Complete |
| **6** | [Evaluation Metrics](ch06-metrics/) | Cross-validation, residual analysis, learning curves | $38k | Validation | ✅ Complete |
| **7** | [Hyperparameter Tuning](ch07-hyperparameter-tuning/) | Grid/Random/Bayesian search + XGBoost | **$32k** | **#1 ✅ #2 ✅** | ✅ Complete |

---

## Narrative Arc: From $70k Baseline to $32k Production Model

### 🎬 Act 1: Foundations (Ch.1-2)
**Build simple models, understand their limits**

- **Ch.1**: Can we predict value from income alone? → Yes, but $70k MAE (too high!)
  - *"This is embarrassing. My teenage nephew could do better by Googling 'California home prices.'" — CEO*
  
- **Ch.2**: Use all 8 features? → Better ($55k MAE), but still $15k away from $40k target
  - *"We're moving in the right direction, but $55k is still unacceptable for lending decisions." — Head of Product*

**Status**: ❌ All constraints unmet. Need better features.

---

### 🔍 Act 2: Feature Diagnostics (Ch.3)
**Understand what the 8 features are actually doing before adding complexity**

- **Ch.3**: Univariate R², VIF, permutation importance → same $55k MAE but full interpretability
  - *"Before we add polynomial features, I need to know which features are driving the model and which are redundant." — Compliance Officer*
  - *"MedInc alone explains 47% of variance. Lat/Lon are jointly irreplaceable. AveBedrms is redundant with AveRooms." — Lead ML Engineer*

**Status**: ✅ Constraint #4 (INTERPRETABILITY) unlocked.

---

### ⚡ Act 3: Feature Engineering & Regularization (Ch.4-5)
**Unlock non-linear power, prevent overfitting**

- **Ch.4**: Polynomial features + interactions → $48k MAE (close!)
  - *"Now we're talking! $48k is almost there. But I'm worried about overfitting — will this work on Texas homes?" — CEO*
  
- **Ch.5**: Regularization (Ridge/Lasso/Elastic Net) → **$38k MAE ✅ Target achieved!**
  - *"Amazing! <$40k MAE AND it generalizes to test districts. We can start piloting this!" — VP Engineering*

**Status**: #1 ✅ ACCURACY · #2 ✅ GENERALIZATION · #4 ✅ INTERPRETABILITY | #3 ❌ #5 ❌ not yet addressed

---

### 📊 Act 4: Evaluation & Production (Ch.6-7)
**Measure properly, optimize for production, add explainability**

- **Ch.6**: Proper evaluation → Cross-validation, residual analysis, learning curves
  - *"I need confidence intervals and proper validation before we show this to investors." — CFO*
  
- **Ch.7**: XGBoost + SHAP → **$32k MAE + feature explanations ✅ Interpretability reinforced!**
  - *"Now THIS is production-grade! $32k MAE, SHAP explains every prediction, and underwriters love it." — Head of Compliance*

**Status**: #1 ✅ ACCURACY · #2 ✅ GENERALIZATION · #4 ✅ INTERPRETABILITY | #3 ❌ (classification is a separate track) · #5 ⚠️ pipeline established

---

## The Dataset: California Housing

Every chapter uses the same dataset: [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from `sklearn.datasets.fetch_california_housing`.

**20,640 districts** with 8 features:
- `MedInc` — Median income (×$10k)
- `HouseAge` — Median house age (years)
- `AveRooms` — Average rooms per household
- `AveBedrms` — Average bedrooms per household
- `Population` — Block population
- `AveOccup` — Average household size
- `Latitude` — Block latitude
- `Longitude` — Block longitude

**Target**: `MedHouseVal` — Median house value (×$100k)

**Why this dataset is perfect**:
- ✅ Real-world data (1990 California census)
- ✅ Built into sklearn (no downloads)
- ✅ Interpretable features (everyone understands income, location, house age)
- ✅ Clear business metric (MAE in dollars, not abstract percentages)
- ✅ Natural progression: Start with 1 feature → add complexity → 8 features + interactions

---

## What You'll Build

By the end of this track, you'll have:

1. ✅ **Linear regression pipeline** from scratch (gradient descent + closed-form solution)
2. ✅ **Feature engineering** skills (polynomial features, interaction terms)
3. ✅ **Regularization mastery** (Ridge/Lasso/Elastic Net, λ tuning)
4. ✅ **Proper evaluation** (cross-validation, residual analysis, learning curves)
5. ✅ **Production ensemble model** (XGBoost achieving $32k MAE)
6. ✅ **Interpretable predictions** (SHAP values for every estimate)
7. 🎓 **Deep understanding** of when to use MSE vs MAE vs Huber loss, how to detect overfitting, and when linear models hit their limits

---

## How to Use This Track

### Sequential (Recommended)
Work through Ch.1 → Ch.7 in order, then finish with `GRAND_CHALLENGE.md`. Each chapter builds on previous concepts and explicitly unlocks new capabilities.

**Time commitment**: ~6-7 weeks (1 chapter per week, ~4-6 hours each, plus grand challenge)

### By Learning Goal

**"I just need basic regression"**  
→ Ch.1-2 (Linear + Multiple Linear Regression)

**"I need production-grade regression models"**  
→ Ch.1-5 (Foundation + Regularization to achieve $38k MAE)

**"I need interpretable regression for compliance"**  
→ Ch.1-7 (Full track through XGBoost + SHAP)

**"I need to deploy regression at scale"**  
→ Complete Ch.1–7 and the grand challenge

### By Constraint

- **#1 Accuracy**: Ch.1, 2, 4, 5, 7
- **#2 Generalization**: Ch.5, 6, 7
- **#3 Multi-Task**: ➡️ Out of scope here — continues in [02-Classification](../02-Classification/) and [03-NeuralNetworks](../03-NeuralNetworks/)
- **#4 Interpretability**: Ch.3 (VIF + permutation importance), Ch.7 (XGBoost + SHAP explanations)
- **#5 Production**: Ch.6 (evaluation framework), Ch.7 (systematic tuning pipeline), `GRAND_CHALLENGE.md`

---

## Prerequisites

Before starting this track, you should have:

- **Python programming**: NumPy, Pandas, Matplotlib basics
- **Basic statistics**: Mean, variance, correlation
- **Linear algebra**: Vectors, dot products, matrices — see [Math Under The Hood](../../MathUnderTheHood/ch01-linear-algebra/)

**Recommended** (but not required):
- Calculus (derivatives, gradients) — covered in [Math Ch.3-6](../../MathUnderTheHood/)
- Jupyter notebooks (all code examples use notebooks)

**Not required**:
- Machine learning experience (we build from scratch)
- Deep learning knowledge (that's Topic 03 — Neural Networks)

---

## What Comes Next?

After mastering regression:

### Same paradigm, different loss:
- **[02-Classification](../02-Classification/README.md)** — FaceAI with CelebA dataset  
  Learn how logistic regression is just linear regression with sigmoid + cross-entropy loss

### Different paradigm entirely:
- **[03-Neural Networks](../03-NeuralNetworks/README.md)** — UnifiedAI  
  See how neural networks unify regression + classification with the same architecture

### Production infrastructure:
- **[08-Ensemble Methods](../08-EnsembleMethods/README.md)** — EnsembleAI  
  Deep dive into Random Forest, XGBoost, LightGBM, CatBoost, and stacking

---

## Chapter Structure

Every chapter follows the same template:

1. **§0 The Challenge** — Current MAE, what's blocked, what this unlocks
2. **§1 Core Idea** — Plain-English intuition
3. **§2 Running Example** — How SmartVal AI uses this technique
4. **§3 Math** — Key equations with inline explanations
5. **§4-N** — Diagrams, code walkthroughs, hyperparameter tuning
6. **§N What Can Go Wrong** — Diagnostic flowcharts, common traps
7. **§N+1 Progress Check** — ✅ Unlocked capabilities, ❌ Still blocked, next preview

---

## Track Structure Snapshot

This topic is organized as:

- `ch01-linear-regression/`
- `ch02-multiple-regression/`
- `ch03-feature-importance/`
- `ch04-polynomial-features/`
- `ch05-regularization/`
- `ch06-metrics/`
- `ch07-hyperparameter-tuning/`
- `GRAND_CHALLENGE.md`

Use chapter READMEs + notebooks for stepwise learning, then use `GRAND_CHALLENGE.md` as the capstone integration exercise.

---

## Let's Build

🎯 **Your first milestone**: Predict California house values from median income alone.

**Start here**: [Ch.1 — Linear Regression](ch01-linear-regression/)

*"The journey from $70k MAE to $32k production model starts with a single gradient descent step."*
