# Feature Engineering Chapter Restructure — Implementation Plan

**Date:** April 29, 2026  
**Target:** `notes/01-ml/01_regression/ch03_feature_importance/README.md`  
**Status:** Planning → Implementation  
**Estimated Effort:** 10-15 hours (4 phases)

---

## Quick Context (50 words)

Restructure Ch.3 from **concept-stacked theory** (§Filter Methods → §Scaling → §Importance Methods → §VIF) to **workflow-based handbook** (§Inspect → §Multicollinearity Audit → §Transform → §Validate). Add decision checkpoints, inline code snippets, real data visualizations, and industry tool comparisons.

---

## Changes Required (Prioritized)

### Phase 1: README Restructure (4-6 hours)
**Goal:** Reorganize sections to follow practitioner workflow

**Current structure:**
```
§3.1 Filter Methods (Pearson, MI)
§3A Prerequisite: Feature Scaling
§3.2 Method 1: Univariate R²
§3.3 Method 2: Standardized Weights
§3.5 Method 3: Permutation Importance
§3.7 Multicollinearity
§3.8 VIF
```

**New structure:**
```
§1 The Feature Engineering Workflow (Overview with flowchart)
§2 Phase 1: Inspect Features (mean/std/IQR/skew → choose scaler)
  §2.4 DECISION CHECKPOINT
§3 Phase 2: Check Multicollinearity (correlation matrix + VIF → drop/combine)
  §3.3 DECISION CHECKPOINT
§4 Phase 3: Apply Transformations (build pipeline)
  §4.3 Validation checkpoint
§5 Phase 4: Feature Importance Rankings (M1/M2/M3 → final candidacy)
  §5.5 DECISION CHECKPOINT
```

**Implementation:**
1. ✅ Create new section headers (5 min - simple markdown)
2. ✅ Move existing content into new sections (30 min - cut/paste with context preservation)
3. ✅ Add decision checkpoint boxes after each phase (20 min - template-based)
4. ✅ Add workflow overview flowchart (15 min - Mermaid diagram)

**Minimal context needed:** Current README.md sections §3.1-§3.9 (lines 100-850)

---

### Phase 2: Code Snippet Additions (3-4 hours)
**Goal:** Add executable code at end of each phase showing that phase's workflow

**Required snippets:**
1. **Phase 1 inspection loop** (end of §2):
```python
for col in numeric_cols:
    skew = df[col].skew()
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[col], bins=50)  # raw
    axes[1].hist(np.log1p(df[col]), bins=50)  # log-transformed
    
    # DECISION LOGIC
    if abs(skew) > 1.0:
        print(f"{col}: Skew={skew:.2f} → log1p + StandardScaler")
```

2. **Phase 2 VIF calculation** (end of §3):
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# DECISION LOGIC
for _, row in vif_data.iterrows():
    if row['VIF'] > 10:
        print(f"{row['Feature']}: VIF={row['VIF']:.1f} ❌ SEVERE - Drop")
    elif row['VIF'] > 5:
        print(f"{row['Feature']}: VIF={row['VIF']:.1f} ⚠️ HIGH - Monitor")
```

3. **Phase 3 transformation pipeline** (end of §4):
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Industry standard: ColumnTransformer for mixed transformations
transformers = [
    ('log_scale', Pipeline([
        ('log', FunctionTransformer(np.log1p)),
        ('scale', StandardScaler())
    ]), high_skew_features),
    ('standard', StandardScaler(), normal_features)
]

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
X_train_transformed = preprocessor.fit_transform(X_train)  # Fit on train only!
```

4. **Phase 4 importance comparison** (end of §5):
```python
# Industry standard: sklearn's permutation_importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'M1_Univariate_R2': univariate_r2_scores,
    'M2_Std_Weight': np.abs(model.coef_),
    'M3_Permutation': perm_importance.importances_mean
})
```

**Implementation:**
1. ✅ Add 4 code snippets (60 min - adapt from existing notebook)
2. ✅ Add "Industry Standard Tools" callout boxes (30 min - one per snippet)
3. ✅ Add example output annotations (20 min - show what you should see)

**Minimal context needed:** Existing notebook cells showing these operations

---

### Phase 3: Visualization Generation (3-4 hours)
**Goal:** Create real-data plots showing distributions and decisions

**Required plots (all with California Housing data):**
1. **Side-by-side histograms** for each feature (raw vs log-transformed) with skewness annotations
2. **Correlation heatmap** (8×8) with VIF annotations on high-correlation pairs
3. **VIF bar chart** sorted by severity (color-coded: green<5, orange 5-10, red>10)
4. **3-method importance comparison** (grouped bar chart)
5. **Before/after transformation** plots showing symmetry improvement

**Implementation:**
1. ✅ Generate plots using existing notebook code (90 min - run cells, save PNGs)
2. ✅ Add decision annotations directly on plots (30 min - matplotlib text())
3. ✅ Insert plots into README with decision captions (20 min)

**Minimal context needed:** Existing notebook outputs + matplotlib styling

---

### Phase 4: Notebook Alignment (2-3 hours)
**Goal:** Restructure notebook to match README workflow exactly

**Current notebook structure:** Follows old README (concept-stacked)

**New notebook structure:**
```
Cell 1: [markdown] Overview + imports
Cell 2: [code] Load California Housing data
Cell 3: [markdown] Phase 1: Inspect Features
Cell 4: [code] Inspect AveRooms (example walkthrough)
Cell 5: [code] Inspect all features (loop)
Cell 6: [markdown] DECISION: Scaler choices based on skewness
Cell 7: [markdown] Phase 2: Check Multicollinearity
Cell 8: [code] Correlation matrix heatmap
Cell 9: [code] VIF calculation
Cell 10: [markdown] DECISION: Drop AveBedrms (VIF=6.8)
Cell 11: [markdown] Phase 3: Apply Transformations
Cell 12: [code] Build ColumnTransformer pipeline
Cell 13: [code] Validate transformed distributions
Cell 14: [markdown] Phase 4: Feature Importance
Cell 15: [code] Compute M1/M2/M3
Cell 16: [code] Feature candidacy table
Cell 17: [markdown] FINAL DECISIONS
```

**Implementation:**
1. ✅ Reorder existing cells to match new structure (30 min - cut/paste)
2. ✅ Add markdown decision checkpoints (20 min - template-based)
3. ✅ Add "Industry Standard" comparison cells (40 min - sklearn vs manual)
4. ✅ Verify all cells run top-to-bottom (20 min - test execution)

**Minimal context needed:** Existing notebook cells (can reference by number)

---

## Industry Standard Tools Integration

**Principle:** Show manual implementation first (build intuition), then show industry-standard one-liner.

### Required "Industry Standard" Callout Boxes

**Box 1: After manual StandardScaler explanation**
```markdown
> 💡 **Industry Standard:** `sklearn.preprocessing.StandardScaler`
> ```python
> from sklearn.preprocessing import StandardScaler
> scaler = StandardScaler()
> X_train_scaled = scaler.fit_transform(X_train)  # One line!
> X_test_scaled = scaler.transform(X_test)
> ```
> **When to use:** Always in production. Manual implementation shown for learning only.
```

**Box 2: After manual VIF calculation**
```markdown
> 💡 **Industry Standard:** `statsmodels.stats.outliers_influence.variance_inflation_factor`
> ```python
> from statsmodels.stats.outliers_influence import variance_inflation_factor
> vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
> ```
> **When to use:** Part of standard feature engineering pipelines before training.
```

**Box 3: After manual permutation importance**
```markdown
> 💡 **Industry Standard:** `sklearn.inspection.permutation_importance`
> ```python
> from sklearn.inspection import permutation_importance
> result = permutation_importance(model, X_test, y_test, n_repeats=10)
> ```
> **When to use:** Post-training model interpretation. Integrated in many AutoML libraries.
```

**Box 4: After manual correlation matrix**
```markdown
> 💡 **Industry Standard:** `pandas.DataFrame.corr()` + `seaborn.heatmap`
> ```python
> corr_matrix = df.corr()
> sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
> ```
> **When to use:** First step of every EDA workflow. Standard in Jupyter notebooks.
```

---

## Implementation Instructions (Minimal LLM Calls)

### Call 1: Section Restructuring (Single Edit)
**Context required:** Lines 1-850 of current README.md  
**Task:** Multi-replace operation moving §3.1-§3.9 into new §1-§5 structure  
**Output:** Restructured README.md with new section headers

### Call 2: Code Snippet Insertion (Single Edit)
**Context required:** New section boundaries (lines 150, 350, 550, 750)  
**Task:** Insert 4 code snippets at end of each phase section  
**Output:** README with inline code examples

### Call 3: Decision Checkpoint Addition (Single Edit)
**Context required:** End of each phase section (4 locations)  
**Task:** Insert decision checkpoint template at 4 positions  
**Output:** README with decision checkpoints

### Call 4: Industry Tools Callouts (Single Edit)
**Context required:** After each major concept (4 locations)  
**Task:** Insert "Industry Standard" callout boxes  
**Output:** README with tool comparisons

### Call 5: Notebook Restructuring (Single Edit)
**Context required:** Existing notebook cell structure  
**Task:** Reorder cells + add markdown decision cells  
**Output:** Notebook matching README workflow

### Call 6: Visualization Integration (Manual)
**Context required:** Generated plots (manual step)  
**Task:** Insert plot references in README  
**Output:** README with visual workflow

**Total LLM calls:** 6 (5 edits + 1 manual step)  
**Total context:** <2000 lines per call

---

## TODO: Notebook Audit

**Task:** Verify notebooks explain concepts AND provide industry-standard tools

**Checklist per notebook:**
- [ ] Shows manual implementation (learning)
- [ ] Shows sklearn/pandas equivalent (production)
- [ ] Explains WHEN to use each
- [ ] Example: Gradient descent manual → then `model.fit()`
- [ ] Example: Feature scaling manual → then `StandardScaler()`
- [ ] Example: VIF calculation manual → then `statsmodels.VIF`
- [ ] Example: Correlation matrix manual → then `df.corr()`

**Pattern:**
```python
# Manual (learning): 10 lines showing the math
for i in range(n_iterations):
    gradient = compute_gradient(X, y, w, b)
    w = w - alpha * gradient
    # ...

# Industry standard (production): 1 line
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)  # That's it!
```

---

## Success Criteria

1. ✅ Reader can follow README top-to-bottom without jumping around
2. ✅ Each section ends with executable code showing that phase
3. ✅ Decision checkpoints explicitly say "what you saw → what it means → what to do"
4. ✅ All visualizations use California Housing real data
5. ✅ Industry standard tools shown alongside manual implementations
6. ✅ Notebook structure matches README exactly
7. ✅ All code runs without errors (test execution)

---

## Files Modified

- `notes/01-ml/01_regression/ch03_feature_importance/README.md` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/notebook_solution.ipynb` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/notebook_exercise.ipynb` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/img/` (new plots)

---

## Next Steps

1. Review this plan
2. Execute Phase 1 (README restructure) as proof-of-concept
3. Review Phase 1 output before proceeding
4. Execute remaining phases in sequence
5. Update authoring guide with extracted patterns

**Estimated completion:** 2-3 days (with review cycles)
