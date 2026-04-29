# ML Track Authoring Guide Update — Implementation Plan

**Date:** April 29, 2026  
**Target:** `notes/01-ml/authoring-guide.md`  
**Status:** Planning → Implementation  
**Estimated Effort:** 3-4 hours (single addendum)

---

## Quick Context (50 words)

Add "Workflow-Based Chapter Pattern" section to authoring guide to support procedural chapters (Ch.0 Data Prep, Ch.3 Feature Engineering, Ch.8 Data Validation). Provide template, decision criteria, and industry tools guidance without replacing existing concept-based patterns.

---

## Changes Required

### Single Addition: Workflow-Based Chapter Pattern Section

**Location:** Insert after "Chapter README Template" section (after line ~200)

**Content structure:**
```markdown
## Workflow-Based Chapter Pattern (Procedural Chapters)

### When to Use (Decision Criteria)
### Modified Template for Workflow Chapters
### Key Differences from Concept-Based Template
### Code Snippet Guidelines for Workflow Chapters
### Decision Checkpoint Format
### Visualization Strategy for Workflow Chapters
### Industry Standard Tools Integration
### When NOT to Use Workflow Structure
```

---

## Implementation Instructions (Minimal LLM Calls)

### Call 1: Insert Workflow Pattern Section (Single Edit)

**Context required:** Lines 150-250 of authoring-guide.md (section boundaries)

**Task:** Insert new section after "Chapter README Template"

**Content to insert:** (~600 lines total)

<details>
<summary>Full section content (click to expand)</summary>

```markdown
## Workflow-Based Chapter Pattern (Procedural Chapters)

> **When to use:** Chapters covering procedures practitioners follow (feature engineering, 
> data validation, model diagnostics, hyperparameter tuning) should use workflow-based structure 
> instead of concept-based structure.

### Identifying Procedural Chapters

A chapter is workflow-based if:
- ✅ It teaches a **sequence of decisions** more than a single concept
- ✅ Practitioner asks "what should I do next?" after each section
- ✅ Multiple tools/techniques are chosen based on data characteristics
- ✅ The chapter reads like a troubleshooting guide, not a concept introduction

**Examples:**
- **Workflow-based:** Feature Engineering (inspect → decide scaler → check VIF → transform)
- **Concept-based:** Linear Regression (concept → math → training → evaluation)

### Modified Template for Workflow Chapters

```
# Ch.N — [Topic Name]

[Same header: story, curriculum context, notation]

---

## 0 · The Challenge — Where We Are
[Same as concept-based template]

## 1 · The Workflow at a Glance
[Numbered list or flowchart showing all phases]

## 2 · Phase 1: [Action Verb] (e.g., "Inspect Features")

### 2.1 What to Look For
[Diagnostic criteria with thresholds]

### 2.2 How to Measure It
[Code snippet showing inspection loop]

### 2.3 Visual Diagnosis
[Histograms, heatmaps, plots with real data]

### 2.4 DECISION CHECKPOINT

**What you just saw:** [Observation from data]
**What it means:** [Interpretation]
**What to do next:** [Action with specific choice]

[Repeat for all phases]

## N-1 · The Complete Decision Tree
[Mermaid flowchart showing all phases + decisions integrated]

## N · Progress Check — What We Can Solve Now
[Same as concept-based template]
```

### Key Differences from Concept-Based Template

| Element | Concept-Based | Workflow-Based |
|---------|---------------|----------------|
| **§1 content** | Core Idea (2-3 sentences) | The Workflow (numbered phases) |
| **Section headers** | Nouns (The Math, The Hyperparameter Dial) | Action verbs (Inspect, Audit, Transform) |
| **Decision points** | End (What Can Go Wrong) | After each phase (Decision Checkpoint) |
| **Code placement** | Primarily in notebook | Inline snippets + notebook |
| **Progress tracking** | Final section only | Phase-level + final |

### Code Snippet Guidelines for Workflow Chapters

**Rule 1: Each phase ends with executable code showing that phase's workflow**

```python
# ✅ Good: Phase 1 code snippet (inspection loop)
for col in numeric_cols:
    skew = df[col].skew()
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    
    # Plot side-by-side: raw vs log-transformed
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[col], bins=50, edgecolor='black')
    axes[1].hist(np.log1p(df[col]), bins=50, edgecolor='black', color='green')
    
    # DECISION LOGIC
    if abs(skew) > 1.0:
        print(f"{col}: Skew={skew:.2f} → log1p + StandardScaler")
    elif iqr / df[col].std() > 2.5:
        print(f"{col}: Heavy outliers → RobustScaler")
    else:
        print(f"{col}: Symmetric → StandardScaler")
```

**Rule 2: Decision logic appears in code comments, not just prose**

```python
# ✅ Good: Inline decision annotation
if vif > 10:
    print(f"{feat}: VIF={vif:.1f} ❌ SEVERE - Drop or combine")
elif vif > 5 and target_corr < 0.3:
    print(f"{feat}: VIF={vif:.1f}, weak signal → Drop candidate")
else:
    print(f"{feat}: VIF={vif:.1f} ✅ OK")
```

**Rule 3: Example usage blocks after defining functions**

```python
# Define utility
def inspect_feature_distribution(df, col):
    # ... implementation ...

# Example usage — show it in action
inspect_feature_distribution(housing_df, 'AveRooms')
# Output: "AveRooms: Skew=2.4 → log1p + StandardScaler"
```

### Decision Checkpoint Format

Every checkpoint follows this 3-part structure:

```markdown
### N.M DECISION CHECKPOINT

**What you just saw:**
- [Observation 1 with specific numbers]
- [Observation 2 with specific numbers]

**What it means:**
- [Interpretation of observations]
- [Why this matters for the model]

**What to do next:**
→ [Action 1: specific, executable step]
→ [Action 2: validation to perform]
```

### Industry Standard Tools Integration

**Principle:** Show manual implementation first (build intuition), then show industry-standard one-liner.

**Pattern for every major technique:**

```markdown
### [Technique Name] — Manual Implementation

[Explanation of concept]

[Code showing from-scratch implementation]

---

> 💡 **Industry Standard:** `library.module.Function`
> 
> ```python
> from sklearn.preprocessing import StandardScaler
> scaler = StandardScaler()
> X_scaled = scaler.fit_transform(X_train)  # One line!
> ```
> 
> **When to use:** Always in production. Manual implementation shown for learning only.
> **Common alternatives:** `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`
> **See also:** [sklearn preprocessing docs](https://scikit-learn.org/stable/modules/preprocessing.html)
```

**Required callout boxes per chapter type:**

| Chapter Type | Industry Tools to Show |
|--------------|------------------------|
| Feature Engineering | `StandardScaler`, `RobustScaler`, `ColumnTransformer`, `variance_inflation_factor`, `permutation_importance` |
| Data Validation | `pandas_profiling`, `great_expectations`, `pandera` |
| Hyperparameter Tuning | `GridSearchCV`, `RandomizedSearchCV`, `Optuna` |
| Model Training | `sklearn.model_selection`, `cross_val_score`, `Pipeline` |

### When NOT to Use Workflow Structure

**Stick with concept-based structure for:**
- Chapters introducing single algorithms (CNNs, Transformers, SVMs)
- Chapters where the "workflow" is just "train → evaluate" (no branching decisions)
- Mathematical foundations (MLE, backprop derivation)
- Chapters with single hyperparameter dial (learning rate tuning)

**Use workflow structure for:**
- Multi-tool selection processes (feature selection, scaler choice, regularization)
- Diagnostic procedures (data quality audit, model debugging)
- Tuning strategies with decision trees (hyperparameter search paths)
- Chapters answering "what should I check first?" questions

---

## Procedural Chapters in ML Track

| Chapter | Current Structure | Should Restructure? | Priority |
|---------|------------------|---------------------|----------|
| **Ch.0 Data Prep** | Concept-based | ✅ YES | HIGH (7-phase EDA workflow) |
| **Ch.3 Feature Importance** | Concept-based | ✅ YES | HIGH (inspect → VIF → transform) |
| Ch.1 Linear Regression | Concept-based | ❌ NO | - |
| Ch.2 Multiple Regression | Concept-based | ❌ NO | - |
| Ch.7 Hyperparameter Tuning | Concept-based | ⚠️ CONSIDER | MEDIUM (has decision tree) |
| **Ch.8 Data Validation** | Concept-based | ✅ YES | MEDIUM (validation workflow) |

---

```
</details>

**Output:** Authoring guide with new section added at line ~200

---

## TODO: Chapter Audits

### Procedural Chapter Audit Checklist

For each procedural chapter (Ch.0, Ch.3, Ch.8):
- [ ] Does README follow workflow structure?
- [ ] Are decision checkpoints present after each phase?
- [ ] Are code snippets inline (not notebook-only)?
- [ ] Are industry tools shown alongside manual implementations?
- [ ] Does notebook match README structure exactly?

### Industry Tools Coverage Audit

For ALL chapters:
- [ ] Shows manual implementation first (learning)
- [ ] Shows sklearn/pandas/numpy equivalent (production)
- [ ] Explains WHEN to use each
- [ ] Pattern: `Manual 10 lines → Industry standard 1 line`
- [ ] Callout box format: "💡 Industry Standard: `library.function`"

---

## Success Criteria

1. ✅ Authoring guide has workflow-based pattern section
2. ✅ Clear criteria for when to use workflow vs concept structure
3. ✅ Decision checkpoint template provided
4. ✅ Industry tools integration pattern documented
5. ✅ Code snippet placement rules explicit
6. ✅ All existing patterns remain intact (failure-first, numerical walkthroughs, etc.)
7. ✅ Future chapters can self-assess which structure to use

---

## Implementation Steps

1. ✅ Review this plan
2. ✅ Execute single insert operation (Call 1)
3. ✅ Verify section integrates with existing guide
4. ✅ Use new pattern to restructure Ch.3 (proof-of-concept)
5. ✅ Refine pattern based on Ch.3 experience
6. ✅ Apply to Ch.0 and Ch.8

**Estimated completion:** 3-4 hours (guide update only)  
**Chapter restructuring:** Separate 10-15 hour effort per chapter

---

## Files Modified

- `notes/01-ml/authoring-guide.md` (insert ~600 lines at line 200)

**Total LLM calls:** 1 (single insert operation)  
**Total context:** ~500 lines (section boundaries)
