# Classification Track

> **The Mission**: Launch **FaceAI** — an automated face attribute classification system that tags 40 binary facial attributes with >90% average accuracy, enabling a photo organization app to replace expensive manual tagging ($0.05/image) with real-time ML inference.

This is not an academic exercise. Every chapter builds toward a single production challenge: you're the ML Engineer at a photo-tech startup, and the product team demands a system that reliably classifies attributes like Smiling, Eyeglasses, Bald, and 37 more — across celebrity faces the model has never seen.

---

## The Grand Challenge: 5 FaceAI Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **ACCURACY** | >90% average accuracy across 40 attributes | Manual tagging costs $0.05/image × 200k images = $10k. Automation must beat human error rates |
| **#2** | **GENERALIZATION** | Work on unseen celebrity faces | Can't memorize training faces. Must learn attribute patterns that transfer to new people |
| **#3** | **MULTI-LABEL** | Predict 40 simultaneous binary attributes | Each face is Smiling AND Young AND Black_Hair AND ... Real faces have ~20 attributes active |
| **#4** | **INTERPRETABILITY** | Which features matter for each attribute | Users need to understand *why* "Bald" was predicted. Debug misclassifications visually |
| **#5** | **PRODUCTION** | Real-time inference <200ms per image | Photo app needs instant tagging. Can't wait seconds per face |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | Accuracy | Constraints | Status |
|----|-------|--------------|----------|-------------|--------|
| **1** | [Logistic Regression](ch01-logistic-regression/) | Binary baseline: Smiling vs Not-Smiling | ~88% | #1 Partial | ⬚ |
| **2** | [Classical Classifiers](ch02-classical-classifiers/) | Interpretable rules: trees, KNN, NB | ~85% | #4 Partial | ⬚ |
| **3** | [Evaluation Metrics](ch03-metrics/) | Proper measurement: ROC, PR, multi-label | 88% validated | #1 Validated | ⬚ |
| **4** | [Support Vector Machines](ch04-svm/) | Maximum-margin separation | ~89% | #1 Improved | ⬚ |
| **5** | [Hyperparameter Tuning](ch05-hyperparameter-tuning/) | Optimized classifiers | **~92%** | **#1 ✅ #2 ✅** | ⬚ |

> **Track scope**: This 5-chapter track achieves >90% on the *Smiling* attribute and proves the classical classification toolkit end-to-end. **Constraint #3** (predict all 40 attributes simultaneously with multi-label heads) requires multi-output neural networks and continues in [Topic 03 — Neural Networks](../03-NeuralNetworks/README.md) — the recommended next step after completing this track.

---

## Narrative Arc: From 88% Baseline to 92% Tuned System

### Act 1: Binary Foundations (Ch.1–2)
**Build simple classifiers, understand their limits**

- **Ch.1**: Can logistic regression detect Smiling? → Yes, ~88% accuracy (decent baseline!)
  - *"88% is promising, but we have 39 more attributes to classify. And Bald is only 2.5% of faces — accuracy alone won't cut it." — Product Lead*

- **Ch.2**: Decision trees, KNN, Naive Bayes → ~85% accuracy but interpretable rules
  - *"I love that the tree shows 'if pixel[32,40] > 128 → likely Smiling'. But 85% is lower than logistic regression?" — CEO*

**Status**: ❌ Need proper evaluation and better models.

---

### Act 2: Measurement & Margin (Ch.3–4)
**Learn to measure correctly, then push the boundary**

- **Ch.3**: Proper metrics → 88% was misleading! Bald recall is only 12%
  - *"96% accuracy on Bald by always predicting Not-Bald? That's the accuracy paradox. We need balanced metrics." — Data Scientist*

- **Ch.4**: SVM with RBF kernel → ~89% accuracy, maximum-margin separation
  - *"SVM finds the widest gap between Smiling and Not-Smiling in feature space. More robust than logistic regression." — ML Lead*

**Status**: ✅ Proper evaluation framework. SVM improves accuracy.

---

### Act 3: Optimization (Ch.5)
**Tune everything, unlock production quality**

- **Ch.5**: Grid/Random/Bayesian search → **~92% accuracy ✅ Target nearly met!**
  - *"Per-attribute threshold tuning pushed Bald recall from 12% to 68%. And we found optimal C, gamma for SVM." — ML Engineer*

**Status**: ✅✅ Accuracy + Generalization achieved. Multi-label and production remain for later chapters.

---

## The Dataset: CelebA (Celebrity Faces Attributes)

Every chapter uses the same dataset: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (official site — if unavailable, use the Kaggle mirror `jessicali9530/celeba-dataset`) — 202,599 celebrity face images with 40 binary attribute annotations.

**Key attributes** (selected from 40):
- `Smiling` (48%) — Primary binary target in Ch.1–4
- `Male` (42%), `Young` (77%), `Attractive` (51%) — Balanced attributes
- `Eyeglasses` (13%), `Wearing_Hat` (4.8%) — Moderate imbalance
- `Bald` (2.5%), `Mustache` (4.2%) — Severe imbalance (class imbalance examples)

**Image format**: 178×218 color → resized to 64×64 grayscale for classical ML (flattened to 4,096 features or HOG descriptors)

**Why CelebA is perfect for classification**:
- ✅ Natural binary labels (not manufactured thresholds)
- ✅ Natural class imbalance (Bald 2.5% vs Smiling 48%)
- ✅ Natural multi-label (each face has ~20 attributes active)
- ✅ Visual intuition (see what the model gets right/wrong)
- ✅ Standard research benchmark with known baselines
- ✅ Progression: Binary → Multi-class → Multi-label

---

## How to Run

Each chapter has a `notebook.ipynb` that uses a subset of CelebA (5,000 images for speed). Notebooks auto-download the dataset on first run via `torchvision.datasets.CelebA` or fall back to a synthetic proxy if download fails.

```bash
cd notes/ML/02-Classification/ch01-logistic-regression
jupyter notebook notebook.ipynb
```
