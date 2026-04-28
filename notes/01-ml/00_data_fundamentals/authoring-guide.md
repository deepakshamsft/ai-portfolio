# Data Fundamentals Track — Authoring Guide

> **This document defines the chapter-by-chapter structure for the Data Fundamentals track.**  
> Each chapter lives under `notes/01-ml/00_data_fundamentals/` in its own folder, containing a README and a Jupyter notebook.  
> Read this before authoring any chapter to keep tone, structure, and the RealtyML running example consistent.
>
> **📚 Status:** Track created April 2026 as P0 priority to address production readiness gap.

<!-- LLM-STYLE-FINGERPRINT-V1
canonical_chapters: ["notes/01-ml/01_regression/ch01_linear_regression/README.md", "notes/01-ml/00_data_fundamentals/ch01_pandas_eda/README.md"]
voice: second_person_practitioner
register: technical_but_conversational
formula_motivation: required_before_each_formula
numerical_walkthroughs: california_housing_with_explicit_data_quality_issues
dataset: california_housing_with_intentional_outliers_missing_values_imbalance
failure_first_pedagogy: true
callout_system: {insight:"💡", warning:"⚠️", constraint:"⚡", optional_depth:"📖", forward_pointer:"➡️"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_plots
section_template: [story_header, challenge_0, animation, core_idea_1, running_example_2, tools_3, step_by_step_4, key_diagrams_5, hyperparameter_dial_6, what_can_go_wrong_7, progress_check_N, bridge_N1]
math_style: minimal_math_focus_on_practical_tools
data_quality_emphasis: every_chapter_shows_before_after_data_quality_impact_on_mae
forward_backward_links: every_concept_links_to_where_it_was_introduced_and_where_it_reappears
conformance_check: compare_new_chapter_against_ml_track_ch01_style
red_lines: [no_tool_without_verbal_explanation, no_technique_without_california_housing_grounding, no_section_without_forward_backward_context, no_data_quality_issue_without_mae_impact_demonstration, no_callout_box_without_actionable_content]
-->

---

## The Plan

The Data Fundamentals track is **3 chapters** that act as a prerequisite to the main ML curriculum. It addresses the critical gap: "You can train models, but can you prepare real-world data?"

```
notes/01-ml/
├── 00_data_fundamentals/              ← NEW TOPIC (you are here)
│   ├── README.md                      ← Topic overview, RealtyML Grand Challenge
│   ├── authoring-guide.md             ← This file
│   ├── ch01_pandas_eda/
│   │   ├── README.md                  ← Act I: Discovery
│   │   ├── notebook.ipynb
│   │   └── img/
│   ├── ch02_class_imbalance/
│   │   ├── README.md                  ← Act II: Rebalancing
│   │   ├── notebook.ipynb
│   │   └── img/
│   └── ch03_data_validation/
│       ├── README.md                  ← Act III: Prevention
│       ├── notebook.ipynb
│       └── img/
│
├── 01_regression/                     ← Existing ML track continues here
│   └── ...
```

Each chapter is self-contained. Read the README to understand the concept, run the notebook to see it in action. The README and notebook teach exactly the same things in the same order.

---

## The Running Example — RealtyML Production Failure

> ⚠️ **Scope note:** The RealtyML running example applies ONLY to the **00_data_fundamentals** track. This is a standalone story that bridges into the main ML curriculum's California Housing dataset.

Every chapter in this track uses the **same scenario**: "RealtyML" — a property valuation startup that built a California Housing price predictor achieving **82k MAE** in testing but failed with **174k MAE** when deployed to Portland, Oregon.

**The Stakeholder**: **Sarah Chen**, Lead Data Scientist (3 years experience, promoted from junior role). She inherited the model from a departed contractor who left no documentation. The board has given her 6 weeks to fix it or the product gets shelved.

### **The Three Hidden Failures** (Story Arc)

| Act | Chapter | Constraint | What Sarah Discovers | Her Quote |
|-----|---------|------------|----------------------|-----------|
| **Act I: Discovery** | Ch.1 | 🗑️ Garbage In | Training data has 127 outliers (`HouseAge=150`), 1,483 missing values filled with `0`, right-skewed distribution | **"I can't fix a model if the data is lies."** |
| **Act II: Rebalancing** | Ch.2 | ⚖️ Imbalance Blindness | Model trained on 92% median homes, 8% high-value — but Portland is 60% median, 40% high-value | **"Accuracy on the majority class means nothing if we fail on what matters."** |
| **Act III: Prevention** | Ch.3 | 🔄 Drift Ignored | California training data mean `MedInc=3.8`, Portland production data mean `MedInc=5.2` — no validation caught the shift | **"We need a firewall between bad data and production models."** |

### **Success Criteria**

Reduce Portland MAE from **174k → <95k** by:
1. Fixing data quality (outlier removal, proper imputation)
2. Rebalancing training set (SMOTE, class weights, stratified sampling)
3. Implementing drift detection (Great Expectations, KS test, schema validation)

**Final Outcome**: Portland MAE **174k → 89k** (target met: <95k)

---

## The Grand Challenge — RealtyML Production System

Every chapter explicitly tracks which constraint it helps solve:

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **GARBAGE IN** | Clean outliers, proper imputation | Data entry errors → nonsense patterns learned |
| **#2** | **IMBALANCE BLINDNESS** | Training distribution matches production | 8% minority class in training, 40% in production → systematic failure on what matters |
| **#3** | **DRIFT IGNORED** | Distribution shift detection | California vs Portland feature distributions differ → model extrapolates wildly |

### Progressive Capability Unlock (3 Chapters)

| Ch | What Unlocks | Constraints Addressed | Portland MAE Impact |
|----|--------------|----------------------|---------------------|
| 1 | Outlier detection, imputation strategies, EDA | #1 Garbage In | Can SEE problems, can't fix yet |
| 2 | SMOTE, class weights, stratified sampling | #2 Imbalance Blindness | **174k → 128k** (46k drop) |
| 3 | Great Expectations, KS test, drift alerts | #3 Drift Ignored | **128k → 89k** (constraint met) |

**🎉 Grand Challenge Complete:** Portland MAE **174k → 89k** (target: <95k) — Sarah presents the fix to the board with a data quality dashboard showing Before/After

---

## Chapter README Template

Every chapter README follows this structure (matches ML track conventions):

```markdown
# Ch.N — [Topic Name]

> **The story.** (Historical context — who discovered this problem, when, what production failures motivated the solution)
>
> **Where you are in the RealtyML story.** (Links to previous chapters, what Sarah has discovered so far, what's still broken)
>
> **Notation in this chapter.** (Declare any statistical terms upfront: IQR, Z-score, KS statistic, etc.)

---

## 0 · The Challenge — Where Sarah Is

> 🎯 **The goal**: Fix RealtyML's model — reduce Portland MAE from **174k → <95k** by addressing THREE data quality failures:
> 1. 🗑️ GARBAGE IN: Outliers & bad imputation
> 2. ⚖️ IMBALANCE BLINDNESS: Training 92% median homes, production 40% high-value
> 3. 🔄 DRIFT IGNORED: Distribution shift California → Portland

**What Sarah knows so far:**
- ✅ [Summary of previous chapters' discoveries]
- ❌ **But the model still fails because [X]!**

**What's blocking Sarah:**
[Concrete description of the data quality gap this chapter addresses]

**What this chapter unlocks:**
[Specific capability that advances one or more constraints, with measurable MAE impact]

---

## 1 · The Core Idea (2–3 sentences, plain English)

[The data preparation concept in practitioner language — focus on "why this matters in production"]

## 2 · Running Example: What Sarah Discovers

[Specific manifestation of the data quality issue in the California Housing → Portland deployment scenario]

## 3 · The Tools

[Pandas/scikit-learn/Great Expectations API patterns — show before/after code]

## 4 · How It Works — Step by Step

[Numbered workflow: 1. Detect issue, 2. Quantify impact, 3. Apply fix, 4. Validate improvement]

## 5 · The Key Diagrams

[Mermaid diagrams or plots showing: distribution before/after, MAE improvement, drift detection alerts]

## 6 · The Hyperparameter Dial

[What knobs to turn: imputation strategy choice, SMOTE k-neighbors, validation threshold, etc.]

## 7 · What Can Go Wrong

[Common anti-patterns: over-imputation, synthetic data leakage, ignoring domain knowledge]

## N · Progress Check — [Act Name]

**Constraint Status:**

| # | Constraint | Before Ch.N | After Ch.N | Evidence |
|---|------------|-------------|------------|----------|
| 1 | Garbage In | ❌ 127 outliers present | [status] | [metric] |
| 2 | Imbalance | ❌ 92% majority class | [status] | [metric] |
| 3 | Drift | ❌ No detection | [status] | [metric] |

**Portland MAE:** [before] → [after]

**Sarah's Status Update:**
[One-sentence quote capturing progress or frustration]

**Next Chapter Preview:**
[What Sarah needs to tackle next to reach the <95k target]

---

## N+1 · Bridge to Next Chapter

[Link forward: "But we still haven't addressed [X]. Ch.N+1 shows how to..."]
```

---

## Writing Voice & Tone

### Practitioner Second-Person

Use **second person** (`you`, `your`) throughout:

✅ **Good**: "You discover 127 rows with impossible ages. Your model learned nonsense patterns."  
❌ **Bad**: "We discover..." or "One discovers..." or "Sarah discovers..." (third person narrative)

### Conversational but Technical

✅ **Good**: "Mean imputation assumes Missing At Random (MAR). But what if high-value homes systematically have missing bedroom counts because they're custom estates? Now you've just inserted bias."

❌ **Bad**: "The imputation methodology selected herein must account for the data generation mechanism." (academic)

❌ **Bad**: "Just fill it with the mean, lol." (too casual)

### Failure-First Pedagogy

Show the **wrong way first**, then explain why it fails, THEN show the right way.

✅ **Good**:
```markdown
## Naive Attempt: Fill Missing Values with Zero

You might try:
```python
df['TotalBedrooms'].fillna(0, inplace=True)
```

**Problem**: You just told the model "missing data = no bedrooms" when it actually means "unknown".
The model now sees:
- House A: 0 bedrooms, $500k value → learns "luxury homes have no bedrooms"

**The Fix**: Use median imputation or forward-fill based on domain knowledge.
```

### Constraint-First Structure

Every section should explicitly connect to **which constraint** it helps solve. Use emoji shortcuts:

- 🗑️ = Garbage In (outliers, missing values)
- ⚖️ = Imbalance Blindness (class distribution mismatch)
- 🔄 = Drift Ignored (distribution shift, schema violations)

Example:
```markdown
## Outlier Detection with IQR Method

> ⚡ **Constraint #1: Garbage In** — This section shows how to detect and remove data entry errors that cause the model to learn nonsense patterns.
```

---

## Dataset & Code Conventions

### Use California Housing Consistently

Every notebook loads the same dataset:
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
X, y = data.data, data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['MedHouseVal'] = y
```

### Introduce Intentional Quality Issues

For pedagogical purposes, ADD data quality issues to demonstrate detection/fixing:

```python
# Ch.1: Add outliers for demonstration
import numpy as np
np.random.seed(42)
outlier_indices = np.random.choice(df.index, size=127, replace=False)
df.loc[outlier_indices, 'HouseAge'] = np.random.uniform(100, 200, size=127)

# Ch.1: Introduce missing values
missing_indices = np.random.choice(df.index, size=1483, replace=False)
df.loc[missing_indices, 'AveBedrms'] = np.nan
```

**Document this clearly in the notebook**: "For demonstration purposes, we've intentionally degraded the data to simulate real-world quality issues."

### Show Before/After MAE Impact

Every fix must demonstrate measurable improvement:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Before: with outliers
X_train, X_test, y_train, y_test = train_test_split(X_bad, y, test_size=0.2, random_state=42)
model_bad = LinearRegression().fit(X_train, y_train)
mae_bad = mean_absolute_error(y_test, model_bad.predict(X_test))

# After: outliers removed
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
model_clean = LinearRegression().fit(X_train_clean, y_train_clean)
mae_clean = mean_absolute_error(y_test_clean, model_clean.predict(X_test_clean))

print(f"MAE before: ${mae_bad*1000:.0f}k")
print(f"MAE after: ${mae_clean*1000:.0f}k")
print(f"Improvement: ${(mae_bad - mae_clean)*1000:.0f}k ({((mae_bad - mae_clean)/mae_bad)*100:.1f}%)")
```

### Portland Simulation (Ch.3 Only)

To demonstrate distribution shift:

```python
# Simulate Portland data: shift MedInc distribution
df_portland = df.copy()
df_portland['MedInc'] = df_portland['MedInc'] * 1.37  # Scale to Portland income levels (mean 3.8 → 5.2)

# Train on California, test on Portland
model_california = LinearRegression().fit(X_california, y_california)
mae_portland = mean_absolute_error(y_portland, model_california.predict(X_portland))
print(f"Portland MAE (no drift detection): ${mae_portland*1000:.0f}k")
```

---

## Diagrams & Visualizations

### Required Plots Per Chapter

**Ch.1 (Pandas & EDA)**:
- Distribution histograms (before/after outlier removal)
- Missing value heatmap
- Correlation matrix heatmap
- Box plots showing outliers

**Ch.2 (Class Imbalance)**:
- Class distribution bar chart (before/after SMOTE)
- Precision-recall curve
- Confusion matrix (before/after rebalancing)

**Ch.3 (Data Validation)**:
- KS statistic plot (California vs Portland distributions)
- Schema validation dashboard (Great Expectations)
- Drift alert timeline

### Plot Style (Dark Theme)

All generated plots must use:
```python
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
```

---

## Interview Checklist Template

Every chapter ends with an interview checklist:

```markdown
## Interview Checklist

| Category | Question |
|----------|----------|
| **Must Know** | What is [concept]? How do you detect it? |
| **Must Know** | Name 3 [techniques] for [problem] |
| **Likely Asked** | How would you implement [solution] in production? |
| **Likely Asked** | What's the difference between [A] and [B]? |
| **Trap to Avoid** | "[Common misconception]" — [why it's wrong] |

### Must Know (Can't Be Hired Without This)
- [Bullet point summary of absolutely critical knowledge]

### Likely Asked (Prepare a 60-Second Answer)
- [Bullet point summary of common follow-up questions]

### Trap to Avoid (Instant Red Flag)
- [Common wrong answer that signals lack of production experience]
```

---

## Cross-References & Forward Links

### Link to ML Track

Every chapter should include a forward pointer:

```markdown
## Bridge to ML Track

After completing this track, when you move to [01_regression/ch01_linear_regression](../01_regression/ch01_linear_regression/README.md), you'll see the California Housing dataset again. But now you'll automatically ask:

- ✅ Are there outliers? (Ch.1)
- ✅ Is the training distribution balanced? (Ch.2)
- ✅ Does my test set match production data? (Ch.3)

You've built the habit of **validating data BEFORE training models** — the #1 skill that separates production ML engineers from Kaggle competitors.
```

### Link Between Data Fundamentals Chapters

Use consistent forward/backward references:

```markdown
## What We've Unlocked So Far

- ✅ **Ch.1**: Detect and fix garbage data (outliers, missing values)
- 🔄 **Ch.2**: YOU ARE HERE — rebalance class distributions
- ⏭️ **Ch.3**: Coming up — drift detection and schema validation
```

---

## Exercises

Each chapter includes 3 exercises (easy → medium → hard):

```markdown
## Exercises

1. **Basic**: Apply [technique] to a new dataset. Expected outcome: [metric improvement]

2. **Intermediate**: Compare 3 different [methods]. Which performs best on [scenario]? Why?

3. **Advanced**: Build a production pipeline that [end-to-end workflow]. Bonus: Add monitoring for [drift/schema/quality].
```

---

## Production Patterns

### Scale→Engineer→Fit

Show the proper order:
1. **Scale** the data (understand distributions first)
2. **Engineer** features (fix quality issues, handle imbalance)
3. **Fit** the model (only after data is production-ready)

### Validation Before Deployment

Every chapter should emphasize:
- ✅ Validate schema (expected types, ranges)
- ✅ Validate distributions (train vs test vs production)
- ✅ Validate quality (missing values, outliers, duplicates)
- ❌ NEVER deploy without these checks

---

## Common Pitfalls to Avoid

### Don't Skip the "Why"

❌ **Bad**: "Use SMOTE to fix imbalance."  
✅ **Good**: "Use SMOTE to fix imbalance because the model is optimizing overall accuracy (92%) while failing catastrophically on the minority class (8%) that represents 40% of production traffic."

### Don't Use Toy Datasets

❌ **Bad**: Iris dataset for imbalance demonstration  
✅ **Good**: California Housing with stratified price tiers (consistent with ML track)

### Don't Ignore Domain Knowledge

❌ **Bad**: "Fill all missing values with median."  
✅ **Good**: "For `TotalBedrooms`, use median imputation. For `Latitude/Longitude`, use forward-fill (missing coordinates likely means same location as previous row). For `MedInc`, missing likely indicates data collection error — flag for review."

---

## Conformance Checklist

Before considering a chapter "done", verify:

- [ ] Uses California Housing dataset (with intentional quality issues introduced)
- [ ] Links to RealtyML Grand Challenge (Sarah Chen's story)
- [ ] Shows before/after MAE impact (must be measurable)
- [ ] Includes Progress Check section tracking all 3 constraints
- [ ] Contains 3 exercises (basic/intermediate/advanced)
- [ ] Has Interview Checklist with Must Know / Likely Asked / Trap to Avoid
- [ ] Uses dark theme plots (`facecolor='#1a1a2e'`)
- [ ] Forward link to next chapter or ML track
- [ ] No jargon without immediate definition
- [ ] No formula without verbal explanation
- [ ] All code cells run in <5min total

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-27 | Initial authoring guide for Data Fundamentals track |

---

## Contact & Questions

For questions about this authoring guide:
1. Review the [ML track authoring guide](../authoring-guide.md) for general conventions
2. Check existing chapters: `01_regression/ch01_linear_regression/` for style reference
3. Consult the [plan.md](../../../plan.md) for the RealtyML Grand Challenge full specification
