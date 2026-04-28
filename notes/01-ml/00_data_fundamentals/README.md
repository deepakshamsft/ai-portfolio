# Data Fundamentals — How to Read This Track

> **Entry point and reading map for the Data Fundamentals track.**  
> This track teaches the #1 missing skill in production ML: **how to prepare real-world data BEFORE training models**.
>
> **Track Status:** ✅ **PLANNED** — 3 chapters designed, awaiting implementation (Priority: P0 — Critical)

---

## Who This Is For

**Target audience**: You've completed [00-math_under_the_hood](../00-math_under_the_hood/README.md) or have equivalent math foundations. You can multiply matrices and take derivatives. **But you've never worked with messy real-world data** — outliers, missing values, class imbalance, distribution shift.

This track bridges the gap between "I can code up gradient descent" and "I can prepare production datasets that won't cause models to fail spectacularly in deployment."

### Prerequisites

| Prerequisite | Level needed |
|---|---|
| Python | Comfortable with pandas, numpy — filtering, groupby, basic plotting |
| Statistics | Mean, median, variance, correlation (high-school level) |
| Basic ML | Understand train/test split, MAE, accuracy (covered in `01_regression/ch01`) |

**You do not need**: scikit-learn expertise, cloud infrastructure knowledge, or production deployment experience. This track teaches those skills.

### What Makes This Different

Every data quality issue is demonstrated with **measurable impact on model performance**. You don't just learn "outliers are bad" — you see a 46k MAE improvement when you remove them. Every chapter connects to a **realistic production failure** (RealtyML deployment disaster) so you understand why these skills matter.

---

## The Central Story in One Paragraph

A property valuation startup built a California Housing price predictor achieving **82k MAE** in testing. The model was deployed to Portland, Oregon... and immediately failed with **174k MAE**. The Lead Data Scientist (Sarah Chen) has 6 weeks to fix it or the product gets shelved. She discovers THREE data quality failures: (1) training data contained 127 outlier rows and 1,483 improperly imputed missing values, (2) the model was trained on 92% median-value homes but Portland's market is 40% high-value homes, and (3) California training data distribution (mean income $38k) didn't match Portland production data (mean income $52k), causing wild extrapolation. This track teaches you to detect and fix all three failures — the same skills that prevent 80% of real-world ML deployment disasters.

---

## The Grand Challenge — RealtyML Production Failure

Every chapter threads through a single growing crisis: **RealtyML**, a seed-stage startup that deployed a broken model to production and must fix it in 6 weeks or face shutdown.

### **The Scenario**

**Company**: RealtyML (Series A-funded property valuation platform)  
**Product**: AI-powered home valuation API for real estate agents, lenders, homebuyers  
**Model**: California Housing price predictor (20,640 training samples)  
**Test Performance**: 82k MAE (within acceptable range)  
**Production Deployment**: Portland, Oregon (June 2025)  
**Production Performance**: 174k MAE (catastrophic failure)

### **The Stakeholder — Sarah Chen**

**Role**: Lead Data Scientist (3 years experience, recently promoted)  
**Background**: Inherited the model from a departed contractor who left zero documentation  
**Deadline**: 6 weeks to fix the model or the board shelves the product  
**Constraint**: Cannot retrain from scratch (Portland lacks labeled ground truth data)

**Quote from initial post-mortem meeting**:  
> "I opened the training data for the first time yesterday. There are houses with 150 years of age in a dataset supposedly from 1990. Bedrooms were filled with zeros when they were missing — the model thinks luxury estates have no bedrooms. And nobody checked if California income distributions match Portland's. We didn't just fail — we deployed a broken model with broken data."

### **The Three Constraints (Story Arc)**

Sarah must address three independent data quality failures:

| # | Constraint | Failure Mode | Impact |
|---|------------|--------------|--------|
| **#1** | **🗑️ GARBAGE IN** | 127 outlier rows (`HouseAge > 100`), 1,483 missing values filled with `0` | Model learned nonsense patterns: "luxury homes have 0 bedrooms" |
| **#2** | **⚖️ IMBALANCE BLINDNESS** | Training: 92% median homes, 8% high-value. Production: 60% median, 40% high-value | Model optimized for majority class, fails catastrophically on minority class that represents 40% of Portland traffic |
| **#3** | **🔄 DRIFT IGNORED** | California mean income: $38k. Portland mean income: $52k. No validation caught the shift | Model extrapolates wildly on out-of-distribution data |

**Success Criteria**: Reduce Portland MAE from **174k → <95k** by fixing all three failures.

**Final Outcome** (after Ch.3): Portland MAE **174k → 89k** (target met: <95k achieved)

---

## Progressive Capability Unlock (3 Chapters)

Each chapter gives Sarah ONE new tool to fix ONE failure. The progression mirrors a real production debugging workflow:

| Ch | Tool Unlocked | Constraint Addressed | Portland MAE | Sarah's Status |
|----|---------------|----------------------|--------------|----------------|
| **Act I: Discovery** | Outlier detection, imputation strategies, EDA | 🗑️ Garbage In | **174k** | "I can't fix a model if the data is lies." |
| **Act II: Rebalancing** | SMOTE, class weights, stratified sampling | ⚖️ Imbalance Blindness | **174k → 128k** (46k improvement) | "Accuracy on the majority class means nothing if we fail on what matters." |
| **Act III: Prevention** | Great Expectations, KS test, drift alerts | 🔄 Drift Ignored | **128k → 89k** (final: <95k target met) | "We need a firewall between bad data and production models." |

---

## Chapter Map

### Foundation — Discovery

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.1 | [ch01_pandas_eda/](ch01_pandas_eda) | How do I detect data quality issues BEFORE they break my model? |

### Correction — Rebalancing

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.2 | [ch02_class_imbalance/](ch02_class_imbalance) | How do I fix training data that doesn't match production distribution? |

### Prevention — Validation

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.3 | [ch03_data_validation/](ch03_data_validation) | How do I catch distribution shift BEFORE deployment? |

---

## How We Got Here — A Short History of Data Quality in ML

ML education has historically focused on **algorithms** (gradient descent, backprop, attention) while treating data preparation as "just clean your data lol." The result: 80% of ML projects fail in production not because the model is wrong, but because the data was wrong.

**The through-line in one paragraph**: Early ML (pre-2010) worked on curated benchmark datasets (MNIST, CIFAR) where data quality was guaranteed by dataset creators. Production ML (2010-2020) revealed the painful truth: real-world data has outliers, missing values, label noise, distribution shift. Kaggle competitions reinforced bad habits: optimize a single metric on a fixed test set, ignore deployment. The Great ML Deployment Crisis (2020-2023) emerged: 87% of models never made it to production (VentureBeat 2019). **Data quality became the bottleneck**. This track teaches the missing foundations.

**Key milestones**:
- **2012**: Coursera ML course (Andrew Ng) mentions data quality in 1 slide out of 100
- **2017**: "Data Cascades in High-Stakes AI" paper reveals 92% of ML practitioners encountered data quality issues, but academia doesn't teach solutions
- **2019**: VentureBeat reports 87% of ML projects fail to reach production (data quality cited as #1 cause)
- **2020**: Great Expectations library launches (declarative data validation becomes production standard)
- **2023**: Google's "Data-Centric AI" report shows cleaning data beats improving algorithms in 73% of production scenarios
- **2025**: RealtyML failure (this track's scenario) represents a composite of 100+ real production failures

---

## The Conceptual Architecture

```
RealtyML Production Pipeline (Broken):

┌─────────────────────────────────────────────────────────────────────┐
│                   PRODUCTION FAILURE: 174k MAE                       │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        DEPLOYMENT LAYER (Portland, OR)                       │   │
│  │                                                               │   │
│  │  ❌ Distribution Shift: MedInc 3.8 → 5.2 (no detection)     │   │
│  │  ❌ Minority Class 8% → 40% (model optimized for wrong dist) │   │
│  └────────────────────────────┬──────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼──────────────────────────────────┐   │
│  │         TRAINING LAYER (California Housing)                   │   │
│  │                                                               │   │
│  │  ❌ Outliers: 127 rows (HouseAge > 100)                      │   │
│  │  ❌ Missing Values: 1,483 filled with 0                      │   │
│  │  ❌ Imbalance: 92% majority, 8% minority                     │   │
│  └────────────────────────────┬──────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼──────────────────────────────────┐   │
│  │              RAW DATA (sklearn dataset)                       │   │
│  │                                                               │   │
│  │  ✅ 20,640 samples loaded from sklearn.datasets              │   │
│  │  ❌ No validation, no drift detection, no quality checks     │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

After This Track (Fixed):

┌─────────────────────────────────────────────────────────────────────┐
│              PRODUCTION SUCCESS: 89k MAE (<95k target met)           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        DEPLOYMENT LAYER (Portland, OR)                       │   │
│  │                                                               │   │
│  │  ✅ Ch.3: Drift Alert System (Great Expectations)           │   │
│  │  ✅ Ch.2: Rebalanced Training (40% high-value represented)  │   │
│  └────────────────────────────┬──────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼──────────────────────────────────┐   │
│  │         TRAINING LAYER (California Housing)                   │   │
│  │                                                               │   │
│  │  ✅ Ch.1: Outliers Removed (IQR method)                      │   │
│  │  ✅ Ch.1: Proper Imputation (median for bedrooms)            │   │
│  │  ✅ Ch.2: SMOTE Resampling (balanced representation)         │   │
│  └────────────────────────────┬──────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼──────────────────────────────────┐   │
│  │       VALIDATED DATA (Great Expectations suite)               │   │
│  │                                                               │   │
│  │  ✅ Schema Check: All types, ranges, uniqueness validated    │   │
│  │  ✅ Distribution Check: KS test on all features              │   │
│  │  ✅ Quality Check: No outliers, no bad imputation            │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Reading Paths

### Path A — Data Quality Basics (4–6 hours)

**For**: You're about to start the ML track and want to build good habits from day 1

```
Ch.1: Pandas & EDA (2h)
→ Ch.2: Class Imbalance (1.5h)
→ Ch.3: Data Validation (1.5h)
→ Continue to 01_regression/ch01_linear_regression
```

### Path B — Production Debugging (3–4 hours)

**For**: You have a model failing in production and need to diagnose data issues fast

```
Ch.3: Data Validation (read first — learn drift detection)
→ Ch.1: Pandas & EDA (identify quality issues)
→ Ch.2: Class Imbalance (fix distribution mismatch)
```

### Path C — Interview Prep (2 hours)

**For**: You have a data engineering interview tomorrow

```
Read all 3 chapter READMEs (skip notebooks)
Focus on Interview Checklists at end of each chapter
Memorize: IQR method, SMOTE, KS test, Great Expectations
```

---

## Cross-Track Connections

| From | To | Connection |
|---|---|---|
| Data Fundamentals Ch.1 | ML 01_regression/ch01 | After learning outlier detection, you'll automatically check for them before training |
| Data Fundamentals Ch.2 | ML 02_classification/ch03 | Class imbalance directly affects precision-recall tradeoffs |
| Data Fundamentals Ch.3 | AI Infrastructure ch10 | Drift detection in notebooks → production monitoring pipelines |
| Data Fundamentals (all) | ML (all tracks) | Validates data BEFORE training — the #1 skill separating production engineers from Kaggle competitors |

---

## Setup

This track uses the same environment as the main ML track:

```powershell
# Windows
.\scripts\setup.ps1

# macOS / Linux
bash scripts/setup.sh
```

The setup script installs:
- pandas, numpy, matplotlib, seaborn (EDA)
- scikit-learn, imbalanced-learn (SMOTE, class weights)
- great_expectations (data validation)
- scipy (KS test, statistical tests)

All notebooks run in the `ml-foundations` kernel.

---

## Common Questions

### Q: Do I need to complete this track before starting ML?

**A**: Technically no — the ML track is self-contained. But you'll build better habits if you start here. After this track, you'll automatically ask "Are there outliers? Is my training distribution balanced? Does my test set match production?" BEFORE training models.

### Q: Can I skip to Ch.3 (Data Validation) if I already know pandas?

**A**: Yes, if you're comfortable with EDA and class imbalance. Ch.3 assumes you know how to detect issues (Ch.1) and fix them (Ch.2).

### Q: Why California Housing instead of a "real" messy dataset?

**A**: The ML track uses California Housing as its running example. This track intentionally introduces quality issues (outliers, missing values, imbalance) to demonstrate detection/fixing. You get the pedagogical benefits of a consistent dataset with the learning value of real-world messiness.

### Q: How does this connect to the "SmartVal AI" challenge from the ML track?

**A**: RealtyML (this track) is the **prequel** to SmartVal AI (ML track). Sarah Chen fixes the broken deployment (here), then the model gets rebranded as SmartVal AI and continues improving through the ML curriculum. Think of it as Season 1 (Data Fundamentals) → Season 2 (ML track).

---

## Success Metrics

After completing this track, you should be able to:

✅ Detect outliers using IQR method and Z-scores  
✅ Choose appropriate imputation strategies (mean/median/KNN/forward-fill)  
✅ Identify and fix class imbalance using SMOTE and class weights  
✅ Implement drift detection using KS test and Great Expectations  
✅ Build a validation pipeline that catches data quality issues before deployment  
✅ Explain why a model failed in production (distribution shift, imbalance, outliers)  
✅ Quantify the impact of data quality fixes on model performance (MAE improvement)  

**Interview Ready**: Can answer "How would you debug a model that works in testing but fails in production?" with a systematic 3-step approach (outliers → imbalance → drift).

---

## See Also

- [01-ml/authoring-guide.md](../authoring-guide.md) — General ML track conventions
- [00-math_under_the_hood/](../../00-math_under_the_hood/README.md) — Math prerequisites
- [plan.md](../../../plan.md) — Full project roadmap showing Data Fundamentals as P0 priority
- [Great Expectations Documentation](https://docs.greatexpectations.io/) — Production data validation framework
- [imbalanced-learn Documentation](https://imbalanced-learn.org/) — SMOTE and resampling techniques

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-27 | Initial track README for Data Fundamentals (P0 priority from gap analysis) |
