# Ch.3 Build Plan — Feature Engineering for Linear Regression

> **Context.** The existing README already covers Feature Scaling, Univariate R², Standardised Weights, Permutation Importance, Multicollinearity, and VIF — with full numerical walkthroughs, diagrams, and code. This plan is a **merge**, not a replacement. Todos are sequenced to match the proposed 10-section chapter order. Each todo calls out exactly what already exists, what needs to be written/added, every visual to produce, and every numerical walkthrough to include. Abstract concepts that would exceed a 100-line section are flagged with `[VISUAL+REF]` — implement as an intuition-focused overview with a diagram and a pointer to an external reference.

---

## ⚠️ Alignment Audit Against Authoring Guide + Ch.01/Ch.02

Before executing any todo, read this section. Several items in the plan below conflict with the established authoring guide, the Grand Challenge constraint timeline, and the conventions of Ch.01/Ch.02. Items are classified as **✅ Keep**, **⚠️ Scope-trim**, or **❌ Move/Remove**.

---

### 1 · Scope Violations — Content That Belongs in Other Chapters

The Grand Challenge's regression track journey recap (`GRAND_CHALLENGE.md`) defines Ch.3 as:
> *"Feature Scaling, Importance & Multicollinearity — VIF audit, permutation importance, StandardScaler"*

The plan's proposed 10-section structure turns Ch.3 into a general-purpose feature engineering textbook. Several TODOs cover topics explicitly assigned to other chapters:

| TODO | Content | Where it actually belongs | Verdict |
|------|---------|--------------------------|---------|
| TODO 4 | Encoding Categoricals | Ch.02-Classification / ML track Ch.2 (logistic regression has categoricals) | ❌ Remove |
| TODO 5 | Handling Missing Data | California Housing has **no missing values** — any section here must use fabricated data, violating the "single consistent dataset" rule | ❌ Remove |
| TODO 9 (SHAP) | SHAP waterfall + beeswarm | Grand Challenge assigns SHAP to **Ch.7** ("XGBoost + SHAP = **#4 ✅ Accuracy + explainability**"). Moving it to Ch.3 breaks that narrative | ❌ Move to Ch.7 |
| TODO 10 | PCA, t-SNE, UMAP | Grand Challenge assigns PCA/t-SNE to **Ch.13 — Dimensionality Reduction** | ❌ Remove |
| TODO 11 | FE Mindset closing | Ch.3 already has a closing via "Bridge to Ch.4". A standalone FE philosophy section is not part of the authoring guide template | ❌ Remove |

**What stays in scope for Ch.3:** Scaling (expand), variance threshold (add), filter selection methods (add, brief), multicollinearity (exists), VIF (exists), importance ranking (exists). That's it.

---

### 2 · Running Example Violations

The authoring guide mandates: *"Every chapter uses a single consistent dataset: California Housing."*

- **Categoricals:** California Housing has **zero** categorical features. TODO 4 would require inventing a synthetic `region` or `city` column. This breaks the running example contract.
- **Missing data:** California Housing has **zero** missing values in the sklearn version. TODO 5 would require `np.nan` injection. Same violation.
- **The standard:** Ch.01 and Ch.02 never introduce a feature or concept that can't be demonstrated on the actual California Housing data without modification.

---

### 3 · Section Structure Violations

The authoring guide defines a fixed section skeleton:

```
§0  The Challenge
§1  Core Idea
§2  Running Example
§3  Math
§4  Step by Step
§5  Key Diagrams
§6  Hyperparameter Dial
§7  Code Skeleton
§8  What Can Go Wrong
§N  Progress Check
§N+1 Bridge to Next Chapter
```

Ch.03's existing README follows this exactly (§0 through §10). The plan proposes replacing this with a 10-section content-topic structure (Contract → Scaling → Encoding → Missing → Interactions → Multicollinearity → Selection → Importance → PCA → Mindset) that does **not** map to the required skeleton. This would make Ch.3 structurally inconsistent with Ch.01, Ch.02, and every other chapter in the track.

**The fix:** additions should be inserted **within** the existing section structure, not by replacing it. For example:
- Box-Cox and log transform belong inside existing **§3 Math → Feature Scaling**
- Variance threshold belongs inside existing **§3 Math → Multicollinearity** (as a precondition)
- Filter methods (Pearson, MI) belong inside existing **§3 Math** as a new subsection
- Lasso path belongs inside existing **§10 Bridge to Ch.4/5**

---

### 4 · Grand Challenge Constraint Timeline

The Grand Challenge scoreboard shows Ch.3's constraint unlock as:
- **#4 INTERPRETABILITY — partial** ("VIF audit, permutation importance, StandardScaler")

The plan's SHAP addition would prematurely claim **#4 ✅ full** at Ch.3, which the Grand Challenge awards only at **Ch.7** (XGBoost + SHAP). This would make the progress check in Ch.3 inaccurate.

---

### 5 · Tone & Voice Mismatches

Ch.01 and Ch.02 use a consistent voice — second person, practitioner-framed, discovery narrative:
> *"You're a data scientist at a real estate platform. Your first task…"*  
> *"You just did gradient descent. Very slowly. And by feel."*  
> *"Three weeks later your manager calls…"*

The plan's section descriptions use academic register:
- "Three philosophical stances" (missing data types)
- "Three families — filter, wrapper, embedded"
- "`[VISUAL+REF]`" notation (doesn't exist in the authoring guide or any existing chapter)

All new prose should match the chapters' voice: practitioner, direct, SmartVal AI framing, second person where possible. Drop the `[VISUAL+REF]` tag — instead just write the section with the appropriate level of depth and add the reference inline.

---

### 6 · What the Plan Gets Right ✅

These TODOs are well-scoped, fit the existing chapter, and should proceed:

| TODO | Content | Why it fits |
|------|---------|------------|
| TODO 3a | Log transform / Box-Cox extension | Scaling already exists; this is a natural expansion within §3 |
| TODO 3b | Weight standardisation as importance | Forward-pointer within existing scaling section |
| TODO 6 | Interactions stub (brief bridge to Ch.4) | Already referenced in existing Bridge section; a short addition only |
| TODO 7a | Variance threshold | Precondition for OLS rank-deficiency; belongs before multicollinearity |
| TODO 8 (filter methods only) | Pearson, Spearman, MI — brief | New §3 Math subsection; Ch.3 already has correlation analysis |
| TODO 8b (Lasso pointer) | Lasso as embedded selection | One paragraph; seeds Ch.5 correctly |
| TODO 12 (partial) | Code blocks for log transform, variance threshold, filter methods | Extends §7 Code Skeleton correctly |
| TODO 13 (gen scripts for kept visuals) | log-transform, lasso-path, pearson-vs-mi images | These match kept content |

---

### Revised Section Order (Corrected)

Do **not** use the 10-section structure below. Use the existing chapter skeleton and insert within it:

```
§0  Challenge                    ← exists; add one sentence on FE as pre-model step
§1  Core Idea                    ← exists; no change
§2  Running Example              ← exists; no change
§3  Math
    ├─ Feature Scaling           ← exists; add Box-Cox + log transform + weight-as-beta-importance
    ├─ [NEW] Variance Threshold  ← add before Multicollinearity
    ├─ Multicollinearity / VIF   ← exists; no change
    ├─ [NEW] Filter Methods      ← Pearson, Spearman, MI — brief subsection; Lasso pointer
    ├─ Method 1 — Univariate R²  ← exists
    ├─ Method 2 — Std Weights    ← exists
    └─ Method 3 — Permutation    ← exists
§4  Step by Step                 ← exists; no change
§5  Key Diagrams                 ← exists; add log-transform + pearson-vs-mi images
§6  Hyperparameter Dial          ← exists; no change
§7  Code Skeleton                ← exists; add log transform + variance threshold + MI blocks
§8  What Can Go Wrong            ← exists; no change
§9  Progress Check               ← exists; no change
§10 Bridge to Ch.4               ← exists; add Lasso path diagram as bridge to Ch.5
```

---

---

## Section Order (Target)

> ⚠️ **See the alignment audit above before using this section order.** The 10-section structure below was the original AI draft. It has been reviewed against the authoring guide and Grand Challenge — several sections are out of scope. Use the **Revised Section Order** in the audit instead.

```
0.  Story / Challenge         ← exists, small update
1.  Contract: Data → Model    ← NEW [❌ out of scope — Ch.3 is not an FE primer]
2.  Scaling & Transformation  ← exists, expand with Box-Cox + weight-as-importance bridge [✅]
3.  Encoding Categoricals     ← NEW [❌ California Housing has no categoricals]
4.  Handling Missing Data     ← NEW [❌ California Housing has no missing values]
5.  Feature Interactions      ← NEW stub (bridges to Ch.4) [⚠️ one paragraph max — already in Bridge]
6.  Multicollinearity         ← exists, promote to own section header [✅ keep in §3 Math]
7.  Feature Selection         ← NEW [⚠️ filter methods only — Lasso pointer, no RFE section]
8.  Feature Importance        ← exists as §3, restructure to be the anchor [✅ keep]
9.  Dimensionality Reduction  ← NEW [❌ belongs in Ch.13]
10. FE Mindset (Closing)      ← NEW [❌ not in authoring guide template]
```

---

## TODO List (Original Draft — Apply Alignment Audit Before Executing)

---

### TODO 1 · [MERGE] Section 0 — Extend the narrative hook

**Target location:** Existing § 0 "The Challenge — Where We Are"

**What to add:**
- One paragraph framing that *this chapter is about the pre-model data contract*, not just diagnostics. Make explicit: "Feature engineering is everything that happens before the model sees a single weight update."
- Cross-reference note: *"Optimisers and loss functions live in the training chapter. Here, we're shaping what we hand to the model."*

**Visuals:**
- [ ] **[NEW]** Two-panel diagram: left = raw heterogeneous data (CSV with strings, nulls, dates, a skewed column); right = clean numeric matrix **X**. Arrow labelled "Feature Engineering" bridges them. Save as `img/ch03-fe-contract.png`. Use the chapter's dark background palette.

**Numerical walkthrough:** None needed — this is purely framing.

---

### TODO 2 · [NEW] Section 1 — The Contract Between Data and Model

**Insert after:** Section 0, before existing Notation table.

**What to write (≈ 60 lines):**
- Every ML model expects a fixed-size numeric vector **x** ∈ ℝⁿ. Raw data is rarely that.
- Introduce the feature matrix **X** (rows = observations, columns = features) explicitly here — it's referenced throughout.
- The FE pipeline is a *deterministic, repeatable transform* fitted on training data only (emphasise no test leakage).
- For linear regression specifically: the model is `ŷ = Xw + b` — every column of **X** must be numeric, finite, and on a comparable scale for weight magnitudes to mean anything.

**Visuals:**
- [ ] Re-use the two-panel diagram from TODO 1 (`img/ch03-fe-contract.png`).
- [ ] **[NEW]** Pipeline diagram showing the order: Raw data → Impute → Encode → Scale → **X** → Model. Save as `img/ch03-fe-pipeline.png`.

**Numerical walkthrough:**
- [ ] Take 5 rows of a tiny housing dataset with: a string zip code, an integer year-built, a right-skewed price column, a missing lot size. Walk through *what needs to happen to each column* before a linear model could consume it. No formulas — just build the intuition. Keep it to a 5×4 table + 4 bullet points.

---

### TODO 3 · [EXPAND] Section 2 — Scaling & Transformation (extend existing)

**Target location:** Existing §3 → "Feature Scaling" subsection.

**What already exists:** Z-score / StandardScaler, min-max, gradient descent diagram (`img/feature-scaling-gradient.gif`), the raw-vs-standardised weight walkthrough with 3 districts.

**What to add:**

#### 3a. Right-skewed distributions — log transform and Box-Cox
**Write (≈ 40 lines):**
- When to use log vs standardisation: log for right-skewed count/price data, standardisation for roughly Gaussian data.
- Formula: `x' = log(x + 1)` (the +1 guards against log(0)). Box-Cox generalises this: `x'(λ) = (xλ − 1)/λ` for λ ≠ 0, `log(x)` for λ = 0.
- For Box-Cox: `[VISUAL+REF]` — the derivation of optimal λ is not needed here; show what it does to a histogram and reference: *Box, G.E.P. and Cox, D.R. (1964). "An Analysis of Transformations." JRSS-B 26(2): 211–252.* Also sklearn docs: `sklearn.preprocessing.PowerTransformer`.
- Practical rule for linear regression: if a feature's histogram has a long right tail, log-transform before standardising.

**Visuals:**
- [ ] **[NEW]** Side-by-side histograms: `MedInc` before log transform (right-skewed) vs after. Dark background. Save as `img/ch03-log-transform.png`.

**Numerical walkthrough:**
- [ ] Column `[50000, 55000, 120000, 48000, 200000]`. Apply log1p, compute z-scores on the transformed column. Show the 5 values at each step in a table. Verify the transformed histogram is more symmetric. 5 values, fits in ~15 lines.

#### 3b. Weight standardisation as importance — add forward-pointer
**What to add (≈ 10 lines):**
- After the existing "Why standardisation matters" block, add a forward-pointing note: *"Standardisation doesn't just stabilise gradient descent — it also converts regression coefficients into directly comparable importance scores. A coefficient of 0.83 on standardised MedInc and 0.89 on Latitude now mean the same thing: a 1-std-deviation swing in that feature shifts ŷ by that many units. We exploit this in §8 (Feature Importance) to rank features without any post-hoc calculation."*
- This bridges to the existing Method 2 (Standardised Weights) section without duplicating it.

**Visuals:** None new — forward pointer only.

---

### TODO 4 · [NEW] Section 3 — Encoding Categorical Features

**Insert after:** Scaling section, before missing data.

**Scope (this chapter is linear regression):** Focus only on what's needed to get categorical columns into the design matrix **X**. Classification-specific encodings (target encoding with stratified k-fold, etc.) belong elsewhere.

**What to write (≈ 80 lines):**

#### 4a. Ordinal vs nominal — the first design decision
- Ordinal (size: S < M < L < XL): label encoding preserves order. Safe for linear regression.
- Nominal (color, city, zip): one-hot encoding is the default. Label encoding imposes a spurious numeric ordering that OLS will treat as meaningful (1 < 2 < 3 when there is no such order).

#### 4b. One-hot encoding
- Formula: a column with `k` distinct values becomes `k − 1` binary columns (drop one to avoid perfect multicollinearity — the "dummy variable trap").
- The dropped category is the *reference level* — all other coefficients measure the effect *relative to* the reference.
- High-cardinality warning: a `zip_code` column with 500 values → 499 new columns. Mention frequency encoding or grouping as alternatives.
- `[VISUAL+REF]` for hashing trick and binary encoding: *Weinberger et al. (2009), "Feature Hashing for Large Scale Multitask Learning."* Implemented as `sklearn.feature_extraction.FeatureHasher`.

**Visuals:**
- [ ] **[NEW]** Table showing a `city` column (5 cities) → three side-by-side panels: raw label, one-hot (5 columns shown), and the same one-hot for 500 cities showing sparsity explosion. ASCII table in the README is fine — no image file needed for this one.

**Numerical walkthrough:**
- [ ] 4 rows of California Housing with a synthetic `region` column (NorCal/SoCal/Central). Show the one-hot expansion. Demonstrate the dummy variable trap: include all three dummies, show perfect multicollinearity `dummy1 + dummy2 + dummy3 = 1`; then drop one and show the reference-level interpretation of the remaining two coefficients.

---

### TODO 5 · [NEW] Section 4 — Handling Missing Data

**Insert after:** Encoding section.

**What to write (≈ 80 lines):**

#### 5a. Three types of missingness
- **MCAR** (missing completely at random): safe to impute. Example: sensor glitch.
- **MAR** (missing at random, conditional on other columns): imputation biased if you don't condition. Example: older houses missing renovation date.
- **MNAR** (missing not at random): imputing is risky; the fact of missingness is itself a signal. Example: high-income households more likely to omit income. Add indicator column.
- `[VISUAL+REF]` for full MNAR treatment: *Rubin, D.B. (1976), "Inference and Missing Data," Biometrika 63(3).* Also: *Little & Rubin, "Statistical Analysis with Missing Data," Wiley, 3rd ed.*

#### 5b. Practical toolkit
- **Mean/median imputation** — fast, destroys variance structure. Use median for skewed columns.
- **Indicator column** — alongside any imputed column, add `feature_was_missing` (0/1). Captures MNAR signal.
- **KNN imputation** — uses row similarity. `sklearn.impute.KNNImputer`.
- **Iterative (MICE)** — `sklearn.impute.IterativeImputer`. `[VISUAL+REF]`: *van Buuren & Groothuis-Oudshoorn (2011), "mice: Multivariate Imputation by Chained Equations in R," JSS 45(3).*

**The leakage trap (pipeline rule):**
- One bold sentence: *"Fit all imputers on training data only; transform test data using training statistics."*

**Visuals:**
- [ ] **[NEW]** Diagram: two pipelines side-by-side — (A) wrong: fit imputer on full dataset before split; (B) correct: split first, fit imputer on train, transform both. Save as `img/ch03-imputation-pipeline.png`.
- [ ] **[NEW]** ASCII heatmap (like seaborn `msno.matrix` style) showing a 10-row × 5-column dataset with missingness pattern. Show two missing values clustered in one column (MNAR candidate) vs scattered across rows (MCAR). Can be ASCII in the README.

**Numerical walkthrough:**
- [ ] 5 rows, 3 columns (`MedInc`, `AveRooms`, `HouseAge`), 2 missing values in `AveRooms`. Step 1: median imputation (compute median of non-missing values, fill). Step 2: add `AveRooms_was_missing` column. Step 3: KNN imputation with k=2 — show the two nearest rows (by Euclidean distance on non-missing features), average their `AveRooms` values, fill. All arithmetic by hand. ≈ 20 lines.

---

### TODO 6 · [NEW] Section 5 — Feature Interactions & Polynomial Features (Stub)

**Insert after:** Missing data section.

**Scope:** This is a *stub* — a preview/bridge to Ch.4, not a full treatment. Keep to ≈ 40 lines.

**What to write:**
- Two features can carry more signal in combination than either alone. Example: `MedInc × Latitude` — the coastal premium (same income, different geography → different price).
- Polynomial features: `x₁²`, `x₁ · x₂`, `x₂²` extend a linear model's ability to fit non-linear relationships while staying within the OLS framework.
- Explosion warning: degree-2 expansion on `d` features → `d(d+1)/2 + d` new terms. For `d = 8`: 44 features. For `d = 100`: ~5,150 features. This is why regularisation (Ch.5 Lasso/Ridge) is needed before expanding.
- Ratio features (domain-informed interactions): `price_per_sqft`, `revenue_per_employee`. These are often better than brute-force polynomial expansion because they encode domain knowledge.
- Close with: *"Full treatment in Ch.4: we'll add `MedInc²` and `MedInc × Latitude` and measure the MAE drop."*

**Visuals:**
- [ ] **[NEW]** 2D scatter plot showing two clusters linearly inseparable → 3D plot after adding `x₁ · x₂` as third axis where they become separable. Save as `img/ch03-interaction-separability.png`. Keep it illustrative (synthetic data); label clearly.
- [ ] **[NEW]** Table showing `sklearn.PolynomialFeatures(degree=2)` exploding a 3-column dataset to 10 columns. Show the explicit column names (`1, x1, x2, x3, x1², x1x2, x1x3, x2², x2x3, x3²`). ASCII table in README.

**Numerical walkthrough:**
- [ ] Features `[area=50, rooms=3]`. Manually generate degree-2 polynomial features: `1, 50, 3, 2500, 150, 9`. Show the 6-element feature vector. Then build a tiny 3-row design matrix and show how OLS on the expanded matrix can fit a curved surface with a linear solver. ≈ 15 lines.

---

### TODO 7 · [EXPAND] Section 6 — Multicollinearity (promote to standalone header)

**Target location:** Existing §3 → "Multicollinearity" subsection.

**What already exists:** VIF definition, formula, thresholds table, 3-sample worked example, the AveRooms/AveBedrms bootstrap instability example, the three fix options (drop one / combine / Ridge Ch.5).

**What to add:**

#### 7a. Variance threshold — prepend as precondition
**Write (≈ 20 lines) before the multicollinearity content:**
- Near-zero variance is a *hard* problem for OLS before collinearity even arises: if a column is constant (or near-constant), `(X'X)` is rank-deficient and `(X'X)⁻¹` does not exist — the normal equations have no unique solution.
- Formula: `Var(xⱼ) = (1/n) Σ(xᵢⱼ − x̄ⱼ)²`. Set a threshold τ (e.g. 0.01); drop any feature with `Var(xⱼ) < τ`.
- Practical code: `sklearn.feature_selection.VarianceThreshold`.

**Visuals:**
- [ ] **[NEW]** Histogram of a near-constant feature (e.g., a `zoning_code` that is `R1` for 99.8% of rows). Show the spike at one value. ASCII histogram in the README is fine.

**Numerical walkthrough:**
- [ ] 5-row column: `[2.01, 2.00, 2.02, 2.01, 1.99]`. Compute variance by hand (mean = 2.006, deviations ≈ small). Result: `Var ≈ 0.00013 < 0.01` → drop. One paragraph. ≈ 10 lines.

#### 7b. Rename section header
- Rename existing subsection header from "Multicollinearity — When Features Compete for the Same Signal" to a top-level section `## 6 · Variance Threshold & Multicollinearity` in the final chapter structure. The variance threshold content comes first, then the existing multicollinearity content.

**No other changes to existing multicollinearity content — it is already excellent.**

---

### TODO 8 · [NEW] Section 7 — Feature Selection

**Insert after:** Multicollinearity section.

**What to write (≈ 100 lines):**

Three families — filter, wrapper, embedded. Be opinionated: in regression practice, filter methods are for fast pruning on large feature sets, embedded (Lasso) is the principal selection tool, wrappers are expensive and mostly superseded.

#### 8a. Filter methods

**Variance threshold** — already covered in TODO 7; forward-reference here.

**Pearson correlation (numeric–target):**
- `ρ(xⱼ, y) = SS_{xy} / √(SS_{xx} · SS_{yy})`, range [−1, 1]. For linear regression, this is directly interpretable.
- Rule of thumb: `|ρ| < 0.05` → no linear association worth pursuing; `|ρ| > 0.3` → worth including; `|ρ| > 0.7` → strong.
- Limitation: Pearson only captures *linear* associations. A U-shaped relationship can have ρ ≈ 0.

**Spearman correlation (monotonic, non-parametric):**
- Uses rank transforms. Handles non-linear but monotonic relationships. Use when scatter plot shows a curve rather than a line.
- Formula: `ρₛ = 1 − 6Σdᵢ²/n(n²−1)` where `dᵢ` is the rank difference.

**Mutual information:**
- Captures non-linear, non-monotonic associations. `I(X; Y) = Σ p(x, y) log[p(x,y)/(p(x)p(y))]`.
- `[VISUAL+REF]` for full derivation: *Cover & Thomas, "Elements of Information Theory," Wiley, 2nd ed., Ch.2.* sklearn: `sklearn.feature_selection.mutual_info_regression`.

**Visuals:**
- [ ] **[NEW]** Two scatter plots side-by-side: (A) linear relationship (ρ = 0.8, MI = high); (B) U-shaped (ρ ≈ 0, MI = high). Caption: *"Pearson misses the U-shape; mutual information catches it."* Save as `img/ch03-pearson-vs-mi.png`.

**Numerical walkthrough — Pearson by hand:**
- [ ] 5 data points: `x = [1, 2, 3, 4, 5]`, `y = [2.2, 2.8, 4.5, 3.9, 5.1]`. Compute `SS_xy`, `SS_xx`, `SS_yy`. Result: `ρ ≈ 0.95`. Show each arithmetic step. ≈ 15 lines.

**Numerical walkthrough — Mutual information (discrete case):**
- [ ] 2×2 contingency table: `income_high` (0/1) × `house_value_high` (0/1). Compute joint probabilities, marginals, and entropy terms. Show `I(X; Y) = H(X) − H(X|Y)`. ≈ 15 lines.

#### 8b. Wrapper methods (brief)

**RFE (Recursive Feature Elimination):**
- Train model → remove weakest feature by `|wⱼ|` → retrain → repeat.
- Cost: `O(d)` model fits. For 8 features → 8 fits; for 100 features → 100 fits. Expensive.
- In 2024–25 practice, replaced by Lasso (which does selection in a single regularized fit) and permutation importance.
- Implement if needed: `sklearn.feature_selection.RFE`.

**Numerical walkthrough:** None — the concept is self-evident once Lasso is understood. Describe the loop in pseudocode only.

#### 8c. Embedded methods

**Lasso (L1 regularization):**
- Loss: `L(w) = MSE(w) + λ Σ|wⱼ|`. The L1 penalty drives weak coefficients exactly to zero — feature selection as a byproduct of training.
- Full treatment in Ch.5. Plant the seed here: *"When you're unsure which features to drop, Lasso with cross-validated λ is the principled answer — it removes AveBedrms and Population automatically at the right λ."*
- `[VISUAL+REF]`: Tibshirani, R. (1996), "Regression Shrinkage and Selection via the Lasso," JRSS-B 58(1). sklearn: `sklearn.linear_model.Lasso`.

**Tree-based importance (brief mention):**
- Gini/entropy impurity gain. Built into sklearn's `feature_importances_`. Fast, biased toward high-cardinality features. Not applicable for linear regression as the primary model, but useful for quick cross-model validation.
- No numerical walkthrough. One paragraph + reference: *Breiman et al. (1984), "Classification and Regression Trees," CRC Press.*

**Visuals:**
- [ ] **[NEW]** Correlation heatmap of the 8 California Housing features + target (this already exists as `img/ch03-correlation-heatmap.png` — re-use it here with a pointer). Show how to read the target column for filter selection and the feature–feature block for multicollinearity.
- [ ] **[NEW]** Lasso coefficient path plot: x-axis = λ (regularisation strength), y-axis = coefficient values. Show how `AveBedrms` and `Population` hit zero first. Save as `img/ch03-lasso-path.png`. Generate with `LassoPath` in the gen script.

---

### TODO 9 · [EXPAND] Section 8 — Feature Importance (restructure existing §3 methods)

**Target location:** Existing §3 → Methods 1–3, Three-Method Convergence.

**What already exists:** Full treatment of Univariate R², Standardised Weights, Permutation Importance, the convergence table, and the three-view dashboard.

---
Automated README audit (2026-04-24):
- Scanned README files under `notes/ML/01-Regression/ch03-feature-importance/` and nearby Regression READMEs.
- Findings:
    - Non-approved emojis present (examples: 🎯, ✅). Replace with approved set: {💡, ⚠️, ⚡, 📖, ➡️}.
    - Chapter already contains strong numeric walkthroughs; ensure each numeric example also has a deterministic notebook cell (3–5 rows) so CI can verify math reproduction.
    - Confirm VIF examples include small numeric checks that readers can run in <2 minutes.
- Recommended quick fixes: replace emojis, add deterministic notebook cells replicating examples, and iterate the automated checks to confirm section presence.

**What to add:**

#### 9a. SHAP — brief treatment
**Write (≈ 30 lines):**
- The current gold standard for *local* importance (per-prediction explanation) + global importance.
- Intuition only — no Shapley derivation: *"Imagine splitting a bonus among teammates. Each person's fair share is their average marginal contribution across every possible team composition they could join. SHAP applies this idea to features: how much does feature j contribute to this specific prediction, averaged over all orderings of features?"*
- Two key outputs: waterfall plot (one prediction), beeswarm plot (global). Both are readable without knowing the math.
- Limitation relative to permutation: SHAP values are correlated with the feature's correlation structure; permutation importance is model-agnostic and distribution-free. For heavily correlated features (AveRooms/AveBedrms), SHAP may assign credit differently than permutation.
- `[VISUAL+REF]`: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions," NeurIPS. https://shap.readthedocs.io. Mollnar, C. (2022), "Interpretable Machine Learning," Ch.9.

**Visuals:**
- [ ] **[NEW]** SHAP waterfall plot for one California Housing test prediction: shows each feature's push up (+) or down (−) from the base value (mean prediction) to the final output. Generate with `shap` library. Save as `img/ch03-shap-waterfall.png`.
- [ ] **[NEW]** SHAP beeswarm plot for all 8 features: global importance with value distribution. Each dot is one test sample; colour = feature value; x-axis = SHAP value. Save as `img/ch03-shap-beeswarm.png`.

**Numerical walkthrough — 2-feature Shapley value by hand:**
- [ ] Toy example: 2 features (`income`, `location`), 4 possible orderings (2! × 2 orderings actually = 2; correct: for 2 features there are 2 orderings: [income first, location first]). For each ordering, compute marginal contribution of `income`. Average = Shapley value for income. Keep numbers small and round. ≈ 20 lines.
  - Reference ordering: with both features, `ŷ = 4.0`; income alone, `ŷ = 3.5`; location alone, `ŷ = 2.0`; neither (base), `ŷ = 1.5`.
  - Ordering A (income first): marginal contribution of income = 3.5 − 1.5 = 2.0; with location added: 4.0 − 2.0 = 2.0. Income Shapley = (2.0 + 2.0)/2 = 2.0.
  - These numbers are illustrative — fill in self-consistent values when writing.

#### 9b. Update the three-view dashboard table
- Add a SHAP column to the existing three-view dashboard in §3 "Putting It Together." Not a new table — extend the existing one.

---

### TODO 10 · [NEW] Section 9 — Dimensionality Reduction (VISUAL+REF)

**Insert after:** Feature Importance section.

**Scope:** This is a `[VISUAL+REF]` section — ≈ 50 lines, intuition and one worked diagram per method, with explicit references for deeper reading. Do not derive eigenvectors.

**What to write:**

#### 10a. PCA
- What it is: find the directions of maximum variance in the data cloud; project onto them.
- Key tradeoff: dimensionality drops, but interpretability is lost. You can no longer say "feature 3 matters" — only "PC1 (which loads on features 3 and 7) matters."
- Practical tool: scree plot (explained variance ratio) → choose `k` components where the curve elbows.
- For linear regression: PCA components can be used as features when collinearity is severe enough that VIF > 10 on multiple features. Note: the resulting model is harder to explain to stakeholders.
- Geometric intuition: covariance matrix describes the *shape* of the data cloud. Eigenvectors are the axes of that shape. Eigenvalues measure the stretch along each axis.
- `[VISUAL+REF]`: *Jolliffe, I.T. (2002), "Principal Component Analysis," Springer, 2nd ed.* sklearn: `sklearn.decomposition.PCA`. For an interactive visual: https://setosa.io/ev/principal-component-analysis/.

**Visuals:**
- [ ] **[NEW]** 2D scatter plot with the two principal component axes drawn as arrows — orthogonal, aligned with the data's spread. Show the projection of data points onto PC1 as a 1D line. Save as `img/ch03-pca-projection.png`.
- [ ] **[NEW]** Scree plot (explained variance ratio vs number of components) for the California Housing dataset. Mark the "elbow" at 2 components. Save as `img/ch03-pca-scree.png`.

**Numerical walkthrough:**
- [ ] 4 data points in 2D: `[(1,2), (2,3), (3,3), (4,5)]`. Compute the 2×2 covariance matrix by hand (5 arithmetic steps). State the eigenvectors (don't derive — just verify by multiplying: `Σv = λv` for one eigenvector). Project the 4 points onto PC1. Show the variance retained: `λ₁ / (λ₁ + λ₂)`. ≈ 25 lines.

#### 10b. t-SNE and UMAP (half a page)
- Purpose: visualisation, not preprocessing. Do not use t-SNE/UMAP components as model inputs in production.
- What they do: embed high-dimensional data in 2D while preserving local neighbourhood structure. t-SNE: preserves local clusters; UMAP: also preserves global topology better.
- No formula — genuinely too complex for this section.
- `[VISUAL+REF]`: van der Maaten & Hinton (2008), "Visualizing Data using t-SNE," JMLR 9. McInnes et al. (2018), "UMAP: Uniform Manifold Approximation and Projection." Interactive: https://distill.pub/2016/misread-tsne/.

---

### TODO 11 · [NEW] Section 10 — Feature Engineering Mindset (Closing)

**Insert after:** Dimensionality Reduction. ≈ 30 lines.

**What to write:**
- Feature engineering is *iterative* and *domain-informed*. The best features come from understanding the data-generating process, not from running every sklearn transformer blindly.
- The workflow loop: explore → engineer → diagnose (VIF, permutation importance) → model → repeat.
- For this curriculum: in Ch.4 we'll see directly how `MedInc²` and `MedInc × Latitude` — two domain-informed interactions — drop MAE from \$55k to \$48k. No encoding change, no imputation change — just better columns.
- Close with: *"The ceiling on any linear model is determined by the ceiling of its features. Regularisation (Ch.5) optimises within that ceiling; feature engineering raises it."*

**Visuals:** None needed.

**Numerical walkthrough:** None needed.

---

### TODO 12 · [EXPAND] Code Skeleton — extend existing §7

**Target location:** Existing §7 "Code Skeleton."

**What already exists:** Full 6-block workflow for scaling, univariate R², correlation heatmap, standardised weights, VIF, permutation importance, and three-view dashboard.

**What to add:**

```python
# ── 7. One-hot encoding for a categorical column ─────────────────────────
# (synthetic example — California Housing has no raw categoricals)
region = pd.Series(["NorCal", "SoCal", "Central", "NorCal", "SoCal"])
dummies = pd.get_dummies(region, drop_first=True)  # drop_first avoids dummy trap
# Result: SoCal (0/1), Central (0/1) — NorCal is the reference level
```

```python
# ── 8. Missing value imputation with indicator ────────────────────────────
from sklearn.impute import SimpleImputer
import numpy as np

X_with_missing = X_train.copy()
X_with_missing.iloc[5, 2] = np.nan  # inject one missing value

# Add missingness indicator BEFORE imputing
missing_indicator = X_with_missing.isnull().astype(int)
missing_indicator.columns = [f"{c}_missing" for c in missing_indicator.columns]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_with_missing),
    columns=X_with_missing.columns
)
X_final = pd.concat([X_imputed, missing_indicator], axis=1)
```

```python
# ── 9. Variance threshold filter ─────────────────────────────────────────
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_train_s)
kept = np.array(data.feature_names)[selector.get_support()]
print("Features passing variance threshold:", kept)
```

```python
# ── 10. SHAP values (requires: pip install shap) ──────────────────────────
import shap

explainer  = shap.LinearExplainer(model, X_train_s)
shap_vals  = explainer(X_test_s)

# Waterfall plot — first test sample
shap.plots.waterfall(shap_vals[0], feature_names=data.feature_names, show=False)
plt.savefig("img/ch03-shap-waterfall.png", dpi=150, bbox_inches="tight")

# Beeswarm — global importance
shap.plots.beeswarm(shap_vals, feature_names=data.feature_names, show=False)
plt.savefig("img/ch03-shap-beeswarm.png", dpi=150, bbox_inches="tight")
```

```python
# ── 11. PCA scree plot ────────────────────────────────────────────────────
from sklearn.decomposition import PCA

pca = PCA().fit(X_train_s)
plt.figure(figsize=(7, 4), facecolor="#1a1a2e")
plt.plot(np.cumsum(pca.explained_variance_ratio_), "o-", color="#60a5fa")
plt.axhline(0.95, color="#f59e0b", linestyle="--", label="95% variance")
plt.xlabel("Number of components", color="white")
plt.ylabel("Cumulative explained variance", color="white")
plt.title("PCA Scree — California Housing", color="white")
plt.legend()
plt.tight_layout()
plt.savefig("img/ch03-pca-scree.png", dpi=150, facecolor="#1a1a2e")
plt.close()
```

---

### TODO 13 · [EXPAND] Gen scripts — add generators for new images

**Target location:** `gen_scripts/` folder.

**What to add:**

- [ ] Add `gen_ch03_log_transform.py` — generates `img/ch03-log-transform.png` (histogram before/after log transform).
- [ ] Add `gen_ch03_interaction.py` — generates `img/ch03-interaction-separability.png` (2D→3D interaction plot).
- [ ] Add `gen_ch03_pearson_vs_mi.py` — generates `img/ch03-pearson-vs-mi.png` (two scatter plots).
- [ ] Add `gen_ch03_shap.py` — generates `img/ch03-shap-waterfall.png` and `img/ch03-shap-beeswarm.png`.
- [ ] Add `gen_ch03_pca.py` — generates `img/ch03-pca-projection.png` and `img/ch03-pca-scree.png`.
- [ ] Add `gen_ch03_imputation_pipeline.py` — generates `img/ch03-imputation-pipeline.png`.
- [ ] Add `gen_ch03_lasso_path.py` — generates `img/ch03-lasso-path.png`.
- All new gen scripts should follow the dark background convention (`facecolor="#1a1a2e"`). Check existing gen scripts for the exact style.

---

## Images Checklist

| Image | Status | Generator |
|-------|--------|-----------|
| `img/feature-scaling-gradient.gif` | ✅ Exists | — |
| `img/ch03-correlation-heatmap.png` | ✅ Exists | existing gen script |
| `img/ch03-importance-comparison.png` | ✅ Exists | existing gen script |
| `img/ch03-vif-instability.png` | ✅ Exists | existing gen script |
| `img/ch03-permutation-loop.png` | ✅ Exists | existing gen script |
| `img/ch03-feature-importance-needle.gif` | ✅ Exists | — |
| `img/ch03-fe-contract.png` | ☐ TODO 1 | gen_ch03_fe_contract.py |
| `img/ch03-fe-pipeline.png` | ☐ TODO 2 | gen_ch03_fe_contract.py |
| `img/ch03-log-transform.png` | ☐ TODO 3a | gen_ch03_log_transform.py |
| `img/ch03-imputation-pipeline.png` | ☐ TODO 5 | gen_ch03_imputation_pipeline.py |
| `img/ch03-interaction-separability.png` | ☐ TODO 6 | gen_ch03_interaction.py |
| `img/ch03-pearson-vs-mi.png` | ☐ TODO 8a | gen_ch03_pearson_vs_mi.py |
| `img/ch03-lasso-path.png` | ☐ TODO 8c | gen_ch03_lasso_path.py |
| `img/ch03-shap-waterfall.png` | ☐ TODO 9a | gen_ch03_shap.py |
| `img/ch03-shap-beeswarm.png` | ☐ TODO 9a | gen_ch03_shap.py |
| `img/ch03-pca-projection.png` | ☐ TODO 10a | gen_ch03_pca.py |
| `img/ch03-pca-scree.png` | ☐ TODO 10a | gen_ch03_pca.py |

---

## Numerical Walkthroughs Checklist

| Walkthrough | Section | Status |
|---|---|---|
| Raw housing data → what each column needs | §1 | ☐ TODO 2 |
| Log transform: `[50000…200000]` → log1p → z-score | §2 | ☐ TODO 3a |
| Dummy variable trap: `region` column → one-hot → drop one | §3 | ☐ TODO 4 |
| KNN imputation: 5 rows, 2 missing, k=2 nearest neighbours | §4 | ☐ TODO 5 |
| Degree-2 polynomial expansion: `[area=50, rooms=3]` → 6 features | §5 | ☐ TODO 6 |
| Variance threshold: `[2.01…1.99]` → Var=0.00013 → drop | §6 | ☐ TODO 7a |
| VIF: 3 samples, MedInc vs Lat, R²=0.25, VIF=1.33 | §6 | ✅ Exists |
| Pearson ρ by hand: 5 data points | §7 | ☐ TODO 8a |
| Mutual information: 2×2 contingency table | §7 | ☐ TODO 8a |
| Shapley value: 2 features, 2 orderings | §8 | ☐ TODO 9a |
| PCA by hand: 4 points in 2D, covariance matrix, project | §9 | ☐ TODO 10a |
| z-scores: `[2.0, 4.0, 6.0]` (MedInc) | §2 | ✅ Exists |
| OLS standardised weights: MedInc vs Population raw→std | §2 | ✅ Exists |
| Permutation importance loop (by hand, 5 rows) | §3 | ✅ Exists |

---

## References to Add

| Concept | Reference | Where to add |
|---|---|---|
| Box-Cox transform | Box & Cox (1964), JRSS-B 26(2):211–252 | §2 |
| Hashing trick | Weinberger et al. (2009), ICML | §3 |
| Missing data (MNAR) | Rubin (1976), Biometrika 63(3) | §4 |
| MICE imputation | van Buuren & Groothuis-Oudshoorn (2011), JSS 45(3) | §4 |
| Mutual information | Cover & Thomas, "Elements of Information Theory," Wiley 2nd ed. | §7 |
| Lasso | Tibshirani (1996), JRSS-B 58(1) | §7 |
| CART / tree importance | Breiman et al. (1984), "Classification and Regression Trees," CRC | §7 |
| SHAP | Lundberg & Lee (2017), NeurIPS | §8 |
| Interpretable ML (SHAP chapter) | Molnar (2022), https://christophm.github.io/interpretable-ml-book/ | §8 |
| PCA | Jolliffe (2002), "Principal Component Analysis," Springer 2nd ed. | §9 |
| PCA interactive | https://setosa.io/ev/principal-component-analysis/ | §9 |
| t-SNE | van der Maaten & Hinton (2008), JMLR 9 | §9 |
| UMAP | McInnes et al. (2018), arxiv:1802.03426 | §9 |
| Misread t-SNE guide | https://distill.pub/2016/misread-tsne/ | §9 |

---

## Execution Order

Work in this order to avoid forward references breaking mid-edit:

1. **TODO 7a** (variance threshold) — it's a precondition for TODO 7b (multicollinearity header restructure).
2. **TODO 1 + TODO 2** (narrative hook + contract section) — they're sequential and share the same image.
3. **TODO 3** (scaling expand) — builds on the existing scaling section.
4. **TODO 4** (encoding) — new section, no dependencies.
5. **TODO 5** (missing data) — new section, no dependencies.
6. **TODO 6** (interactions stub) — new section, no dependencies.
7. **TODO 8** (feature selection) — references TODO 7's variance threshold forward.
8. **TODO 9** (SHAP addition to importance) — extends existing section.
9. **TODO 10** (dimensionality reduction) — new section.
10. **TODO 11** (closing mindset) — new section.
11. **TODO 12** (code skeleton additions) — extends existing §7.
12. **TODO 13** (gen scripts) — after all README changes are final.

---

## Notebook TODOs

> **Rule from authoring guide:** *"The notebook mirrors the README exactly — same sections, same order."* Only in-scope README todos (✅ from the alignment audit) have corresponding notebook todos. Out-of-scope content (categoricals, missing data, SHAP, PCA) does **not** get notebook cells here.
>
> **Current notebook state:** 29 cells covering: imports → Feature Scaling → Baseline fit → Univariate R² → Std Weights → Rankings Diverge → Correlation Heatmap → VIF → Weight Instability → Permutation Importance → Three-View Dashboard → Multicollinearity Deep Dive → Action Items → Summary. No exercise cells exist yet.

---

### NOTEBOOK TODO A · [EXPAND] Section 0 — Log Transform cells

**Insert after:** Cell 4 (the before/after scaling bar chart cell), before cell 5 (Baseline Model header).

**Why here:** Section 0 of the notebook is "Feature Scaling." Log transform is a scaling technique — it belongs immediately after the StandardScaler demo, matching where it will sit in the README under `§3 Math → Feature Scaling`.

**Cells to add:**

**Cell A1 — Markdown:**
```
### 0a · Skewed Features — Log Transform

`StandardScaler` handles *scale* (units). It doesn't fix *shape*. A right-skewed column
(e.g. `MedInc` or any count-based feature) violates the near-Gaussian assumption that
makes OLS coefficients most stable.

**Log transform:** `x' = log(x + 1)` (the +1 prevents log(0))

After log-transforming, apply StandardScaler as usual. The two steps compose cleanly.

> ⚠️ Only transform features with a clear right skew. Check the histogram first.
```

**Cell A2 — Python (runnable):**
```python
# ── Log transform: MedInc ──────────────────────────────────────────────────
# MedInc has a visible right skew. Apply log1p, then check histogram.

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

axes[0].hist(X_train_raw['MedInc'], bins=60, color='#94a3b8', edgecolor='none')
axes[0].set_title('MedInc — raw\n(right-skewed)', fontweight='bold')
axes[0].set_xlabel('MedInc (×$10k)')
axes[0].set_ylabel('Count')

medinc_log = np.log1p(X_train_raw['MedInc'])
axes[1].hist(medinc_log, bins=60, color='#1d4ed8', edgecolor='none')
axes[1].set_title('MedInc — after log1p\n(more symmetric)', fontweight='bold')
axes[1].set_xlabel('log(MedInc + 1)')

plt.suptitle('Log Transform: Reducing Right Skew Before Scaling', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('img/ch03-log-transform.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"MedInc skewness — raw:       {X_train_raw['MedInc'].skew():.2f}")
print(f"MedInc skewness — log1p:    {medinc_log.skew():.2f}  (closer to 0 = more symmetric)")
print("\nNote: For this chapter we continue with raw MedInc to match Ch.2's baseline.")
print("Log transforms are most valuable when building new pipelines from scratch.")
```

---

### NOTEBOOK TODO B · [EXPAND] Section 0 — Weight-as-beta-importance forward pointer

**Insert after:** Cell A2 (log transform cell), before cell 5 (Baseline Model header).

**Cells to add:**

**Cell B1 — Markdown:**
```
### 0b · Standardisation → Comparable Coefficients

Standardisation doesn't only stabilise gradient descent. It converts every fitted weight
into a **standardised beta coefficient** — directly comparable across features regardless
of original units.

After fitting on standardised features:
- `|w_j|` = how much ŷ shifts per 1-standard-deviation swing in feature `j`
- The feature with the largest `|w_j|` contributes the most to the linear combination

This is exactly what Section 3 (Method 2 — Standardised Weights) exploits.
We're planting the seed here so the result in Section 3 doesn't feel like magic.
```

*(No code cell needed — this is a conceptual connector.)*

---

### NOTEBOOK TODO C · [NEW] Before Section 6 VIF — Variance Threshold section

**Insert after:** Cell 15 (Correlation Heatmap code cell), before cell 16 (VIF header markdown).

**Why here:** Variance threshold is a *precondition* for VIF — a near-constant feature makes `(X'X)` rank-deficient before collinearity even arises. In the notebook, it should appear just before the VIF section to preserve this causal order.

**Cells to add:**

**Cell C1 — Markdown:**
```
## 5b · Variance Threshold — Before We Can Talk About Collinearity

A feature with near-zero variance is essentially a constant. In OLS terms:
if any column of **X** is constant, **X'X** becomes rank-deficient and the normal
equations have no unique solution — the model *cannot be fit at all*.

**Rule:** drop any feature where `Var(xⱼ) < τ` before running VIF or fitting a model.

$$\text{Var}(x_j) = \frac{1}{n}\sum_i (x_{ij} - \bar{x}_j)^2$$

Typical threshold: `τ = 0.01` on standardised features (`sklearn.feature_selection.VarianceThreshold`).
```

**Cell C2 — Python (runnable):**
```python
from sklearn.feature_selection import VarianceThreshold

# Apply on STANDARDISED train features
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train_s)

variances = X_train_s.var(axis=0) if hasattr(X_train_s, 'var') else np.var(X_train_s, axis=0)

print('Feature variances (standardized — should all be ≈ 1.0):')
print('-' * 50)
for feat, var, kept in zip(housing.feature_names, variances, selector.get_support()):
    status = '✅ kept' if kept else '❌ dropped (near-constant)'
    bar = '█' * int(var * 15)
    print(f'  {feat:12s}  var={var:.3f}  {bar}  {status}')

print(f'\nAll {selector.get_support().sum()}/8 features pass threshold=0.01.')
print('California Housing has no near-constant features — but the check is always worth running.')
print('\nOn a real dataset, a column like zip_code_is_R1 (99.8% the same value)')
print('would fail this check and must be dropped before VIF computation.')
```

---

### NOTEBOOK TODO D · [NEW] After Section 6 VIF — Filter Methods section

**Insert after:** Cell 18 (weight instability demo), before cell 19 (Method 3 — Permutation Importance markdown).

**Why here:** Filter methods (Pearson, Spearman, MI) are pre-model selection tools. In the notebook flow, they sit between the collinearity diagnostics (VIF) and the model-based importance methods (permutation). This matches the README order: `§3 Math → [NEW] Filter Methods → Method 1 → Method 2 → Method 3`.

**Cells to add:**

**Cell D1 — Markdown:**
```
## 6b · Filter Methods — Ranking Features Before Fitting

Before running any model, three filter statistics can rank features by their
association with the target. They're fast, scale to thousands of features,
and catch problems (near-zero correlation, non-linear relationships) early.

| Method | Captures | Use when |
|---|---|---|
| **Pearson ρ** | Linear association | Feature is continuous, roughly Gaussian |
| **Spearman ρₛ** | Monotonic (any shape) | Feature is ordinal or has outliers |
| **Mutual information** | Any association (non-linear) | Scatter plot shows a curve, not a line |

> **Note:** Pearson can be zero even when there's a strong non-linear relationship
> (e.g. a U-shape). Always plot before deciding a feature is uninformative.
```

**Cell D2 — Python (runnable):**
```python
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

results = []
for feat in housing.feature_names:
    x = X_train_raw[feat].values
    pr, _  = pearsonr(x, y_train)
    sr, _  = spearmanr(x, y_train)
    results.append({'Feature': feat, 'Pearson ρ': pr, 'Spearman ρₛ': sr})

filter_df = pd.DataFrame(results).set_index('Feature')

# Mutual information (captures non-linear associations)
mi = mutual_info_regression(X_train_raw, y_train, random_state=42)
filter_df['Mutual Info'] = mi
filter_df['Univariate R²'] = univariate_r2  # already computed in Section 2

filter_df = filter_df.sort_values('Mutual Info', ascending=False)
print('Filter method comparison:')
print(filter_df.round(3).to_string())
print('\nNote: Pearson ρ² ≈ Univariate R² (they are mathematically equivalent for linear regression).')
print('Mutual info picks up any association — here MedInc still leads, but by a different margin.')
```

**Cell D3 — Python (runnable, visualisation):**
```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col, title, color in zip(
    axes,
    ['Pearson ρ', 'Spearman ρₛ', 'Mutual Info'],
    ['Pearson ρ\n(linear)', 'Spearman ρₛ\n(monotonic)', 'Mutual Information\n(any association)'],
    ['#94a3b8', '#1d4ed8', '#15803d']
):
    vals = filter_df[col].sort_values()
    ax.barh(vals.index, vals.abs() if 'ρ' in col else vals, color=color, alpha=0.85)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('|score|' if 'ρ' in col else 'score')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

plt.suptitle('Filter Methods — Feature → Target Association\n'
             'Same dataset, three different lenses', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('img/ch03-pearson-vs-mi.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nKey: MedInc leads on all three. Latitude/Longitude score higher on MI than Pearson')
print('because their geographic signal is non-linear (coastal vs inland, not a straight line).')
```

---

### NOTEBOOK TODO E · [NEW] In Section 10 Action Items — Lasso Path cell

**Insert after:** Cell 28 (action items print cell, the one with the `═══` separator), before cell 29 (Summary markdown).

**Why here:** The action items section already bridges to Ch.5. The Lasso path shows *visually* which features Ch.5 will zero out, making the bridge concrete rather than just text.

**Cells to add:**

**Cell E1 — Markdown:**
```
### Lasso Path — A Preview of Ch.5 Feature Selection

Lasso (L1 regularisation) adds a `λ Σ|wⱼ|` penalty to MSE. As `λ` increases,
weak or redundant features have their coefficients driven exactly to zero — feature
selection as a byproduct of training.

The path plot below shows *which features survive as the penalty tightens*.
Features that hit zero first are the ones Ch.5 will deprioritise.
```

**Cell E2 — Python (runnable):**
```python
from sklearn.linear_model import lasso_path

# Compute Lasso path on standardised training features
alphas, coefs, _ = lasso_path(
    X_train_s, y_train,
    alphas=np.logspace(-3, 1, 200),
    max_iter=5000
)

fig, ax = plt.subplots(figsize=(10, 5))

colors_path = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for i, (feat, color) in enumerate(zip(housing.feature_names, colors_path)):
    ax.plot(np.log10(alphas), coefs[i], label=feat, color=color, linewidth=1.8)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_xlabel('log₁₀(λ)  [← weaker penalty    stronger penalty →]', fontsize=10)
ax.set_ylabel('Coefficient value', fontsize=10)
ax.set_title('Lasso Coefficient Path — Which Features Survive?\n'
             'Features hitting zero = candidates for removal in Ch.5', fontsize=11)
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.invert_xaxis()  # left = small λ (OLS), right = large λ (all zeros)
plt.tight_layout()
plt.savefig('img/ch03-lasso-path.png', dpi=150, bbox_inches='tight')
plt.show()

# Report which features zero out first
zero_alpha = {}
for i, feat in enumerate(housing.feature_names):
    first_zero = np.argmax(np.abs(coefs[i]) < 1e-4)
    if first_zero > 0:
        zero_alpha[feat] = alphas[first_zero]

zero_order = sorted(zero_alpha.items(), key=lambda x: x[1], reverse=True)
print('Order features hit zero (first = eliminated earliest by Lasso):')
for feat, alpha in zero_order:
    print(f'  {feat:12s}  at λ = {alpha:.4f}')
```

---

### NOTEBOOK TODO F · [NEW] Exercise cells

**Insert after:** Cell 29 (Summary markdown), at the end of the notebook.

**Per authoring guide:** 2–3 exercise cells — one markdown scaffold per exercise, one partially-filled code cell.

**Cells to add:**

**Cell F1 — Markdown:**
```
## Exercises

Work through these before moving to Ch.4. Each exercise requires changing one thing
and re-running — no new imports needed.
```

**Cell F2 — Markdown:**
```
### Exercise 1 — Does log-transforming MedInc improve the model?

Re-run the full pipeline (Section 1 → Section 7) using `np.log1p(MedInc)` in place of
the raw `MedInc` column. Compare:
- Test MAE before and after
- The univariate R² of log(MedInc) vs raw MedInc
- The standardised weight of log(MedInc) in the joint model

Does log-transforming a single feature noticeably shift the ranking or MAE?
```

**Cell F3 — Python (exercise scaffold):**
```python
# Exercise 1 — Log-transform MedInc and re-run the pipeline
X_train_log = X_train_raw.copy()
X_train_log['MedInc'] = np.log1p(X_train_log['MedInc'])

X_test_log = X_test_raw.copy()
X_test_log['MedInc'] = np.log1p(X_test_log['MedInc'])

# TODO: scale X_train_log and X_test_log using a new StandardScaler
# TODO: fit a LinearRegression and compute test MAE
# TODO: compare to the baseline MAE printed in Section 1

scaler_log = StandardScaler()
# X_train_log_s = ...
# X_test_log_s  = ...
# model_log      = ...
# mae_log        = ...
# print(f'Baseline MAE: ${mae:,.0f}  |  Log-MedInc MAE: ${mae_log:,.0f}')
```

**Cell F4 — Markdown:**
```
### Exercise 2 — What happens to VIF when you drop AveBedrms?

Drop `AveBedrms` from the feature set and recompute:
- VIF for all remaining 7 features (pay attention to AveRooms)
- Permutation importance on the pruned model
- Test MAE

Does AveRooms's VIF drop? Does MAE change meaningfully?
This previews what Lasso will do automatically in Ch.5.
```

**Cell F5 — Python (exercise scaffold):**
```python
# Exercise 2 — Drop AveBedrms and recompute diagnostics
features_pruned = [f for f in housing.feature_names if f != 'AveBedrms']

X_train_pruned = pd.DataFrame(X_train_s, columns=housing.feature_names)[features_pruned]
X_test_pruned  = pd.DataFrame(X_test_s,  columns=housing.feature_names)[features_pruned]

# TODO: fit a LinearRegression on X_train_pruned, compute test MAE
# TODO: compute VIF for all features in X_train_pruned (hint: reuse the VIF loop from Section 6)
# TODO: compute permutation importance on the pruned model
# TODO: compare AveRooms VIF before (≈7.2) and after dropping AveBedrms

# model_pruned = ...
# mae_pruned   = ...
# print(f'Baseline MAE: ${mae:,.0f}  |  Pruned MAE: ${mae_pruned:,.0f}')
```

---

### Notebook Todo Summary — What Already Exists vs What's New

| Section in notebook | Status | Todo |
|---|---|---|
| Imports | ✅ Exists | Add `from sklearn.feature_selection import VarianceThreshold, mutual_info_regression` and `from scipy.stats import pearsonr, spearmanr` |
| Section 0 — StandardScaler demo | ✅ Exists | No change |
| Section 0a — Log transform | ☐ New | **TODO A** |
| Section 0b — Beta-importance pointer | ☐ New | **TODO B** |
| Section 1 — Baseline fit | ✅ Exists | No change |
| Section 2 — Univariate R² | ✅ Exists | No change |
| Section 3 — Std Weights | ✅ Exists | No change |
| Section 4 — Rankings Diverge | ✅ Exists | No change |
| Section 5 — Correlation Heatmap | ✅ Exists | No change |
| Section 5b — Variance Threshold | ☐ New | **TODO C** |
| Section 6 — VIF | ✅ Exists | No change |
| Section 6b — Filter Methods | ☐ New | **TODO D** |
| Section 7 — Permutation Importance | ✅ Exists | No change |
| Section 8 — Three-View Dashboard | ✅ Exists | No change |
| Section 9 — Multicollinearity Deep Dive | ✅ Exists | No change |
| Section 10 — Action Items | ✅ Exists | No change |
| Lasso Path (bridge to Ch.5) | ☐ New | **TODO E** |
| Exercises | ☐ Missing | **TODO F** |

### Import cell update (required before TODO C/D cells will run)

Add to the existing imports cell (cell 2):

```python
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import lasso_path
```
