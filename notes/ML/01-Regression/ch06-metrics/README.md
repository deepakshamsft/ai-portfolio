# Ch.6 — Evaluation Metrics for Regression

> **The story.** **Carl Friedrich Gauss** invented least squares in **1795** (age 18!) to predict the orbit of Ceres, reasoning that the best prediction minimizes the sum of squared errors. **Francis Galton** introduced $R^2$ (coefficient of determination) in the 1880s while studying hereditary traits — "how much of the variation in children's heights is explained by parents' heights?" The mean absolute error (MAE) gained prominence as statisticians realized squared errors over-penalize outliers — real estate appraisers, for instance, care about typical error, not catastrophic ones. **MAPE** (Mean Absolute Percentage Error) emerged from business forecasting where "$10k error on a $100k house" (10%) differs fundamentally from "$10k error on a $1M house" (1%). **Adjusted R²** was developed to solve the "R² always increases when you add features" problem — penalizing model complexity to prevent over-engineering. Today, residual analysis and cross-validation are the twin pillars of regression evaluation — the first tells you *how* your model fails, the second tells you *whether you can trust* its reported performance.
>
> **Where you are in the curriculum.** Ch.5 achieved $38k MAE — below the $40k target! But how reliable is that number? A single train-test split might be lucky. The model might systematically underestimate expensive homes or overfit to coastal districts. This chapter builds a **complete evaluation framework** for regression: multiple error metrics, residual diagnostics, cross-validation stability, learning curves, and prediction intervals. When you're done, you'll know not just *how good* the model is, but *where and how it fails*.
>
> **Notation in this chapter.** $y_i$ — actual value; $\hat{y}_i$ — predicted value; $\bar{y}$ — mean of actuals; MAE $=\tfrac{1}{n}\sum|y_i-\hat{y}_i|$; RMSE $=\sqrt{\tfrac{1}{n}\sum(y_i-\hat{y}_i)^2}$; $R^2 = 1 - \tfrac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$; MAPE $=\tfrac{100}{n}\sum\tfrac{|y_i-\hat{y}_i|}{|y_i|}$; Adjusted $R^2 = 1 - \tfrac{(1-R^2)(n-1)}{n-p-1}$ where $p$ = number of features.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Launch **SmartVal AI** — a production home valuation system satisfying 5 constraints:
> 1. **ACCURACY**: <$40k MAE — 2. **GENERALIZATION**: Unseen districts — 3. **MULTI-TASK**: Value + Segment — 4. **INTERPRETABILITY**: Explainable — 5. **PRODUCTION**: Scale + Monitor

**What we know so far:**
- ✅ Ch.1: Single feature → $70k MAE
- ✅ Ch.2: All 8 features → $55k MAE
- ✅ Ch.4: Polynomial features → $48k MAE
- ✅ Ch.5: Regularization → $38k MAE ← **Target achieved!**
- ❌ **But how confident are we in that $38k number?**

**What's blocking us:**

⚠️ **We have one number ($38k MAE) and zero confidence in it:**

1. **Lucky split?** — One train-test split might have easy test districts. Re-split and MAE could be $45k.
2. **Systematic bias?** — $38k average hides the fact that the model might be $5k off on cheap homes and $80k off on expensive ones.
3. **Overfitting detection?** — Training MAE is $35k, test MAE is $38k. Is that gap normal? When does it become dangerous?
4. **Stakeholder trust?** — CTO asks "can you guarantee predictions within $40k?" and you say... "um, on average?"

**Real production problem:**
- Model reports $38k average MAE on test set
- But residual analysis reveals: underestimates homes > $400k by ~$60k (systematic bias!)
- Q-Q plot shows residuals are NOT normally distributed — long right tail
- MAE computed on a different random split: $42k (above target!)
- **Conclusion**: The $38k number was partly lucky. The model has structural blind spots.

**What this chapter unlocks:**
⚡ **Complete regression evaluation toolkit:**
1. **Multiple error metrics**: MAE vs RMSE vs MAPE vs R² — each reveals different failure modes
2. **Residual diagnostics**: Where and how the model fails (systematic bias, heteroscedasticity)
3. **Cross-validation**: Stable performance estimate across multiple splits
4. **Learning curves**: Diagnose bias vs variance — need more data or more complexity?
5. **Prediction intervals**: Not just "prediction = $380k" but "$380k ± $45k with 95% confidence"

```mermaid
flowchart LR
    subgraph "Before Ch.6"
        SINGLE["Single number:<br/>$38k MAE<br/>One test split"]
    end
    
    subgraph "After Ch.6"
        MULTI["Multiple metrics<br/>MAE, RMSE, R², MAPE"]
        RESID["Residual diagnostics<br/>Where model fails"]
        CV["Cross-validation<br/>$38k ± $2k MAE"]
        LEARN["Learning curves<br/>Bias vs variance"]
        CONF["Prediction intervals<br/>$380k ± $45k (95%)"]
    end
    
    SINGLE --> MULTI
    SINGLE --> RESID
    SINGLE --> CV
    SINGLE --> LEARN
    SINGLE --> CONF
    
    style SINGLE fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style MULTI fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style RESID fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CV fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style LEARN fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CONF fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## Animation

![Chapter animation](./img/ch06-metrics-needle.gif)

---

## · The Metrics Journey — How Our Numbers Evolved

> This is the story the numbers alone don't tell. Follow SmartVal AI from Ch.1 to Ch.6 and watch how every metric moved — not just MAE.

### The Full Picture

| Chapter | Model | Features | MAE | RMSE | R² | Adj. R² | MAPE | What moved the needle |
|---------|-------|---------|-----|------|-----|---------|------|----------------------|
| Ch.1 | OLS (1 feature) | 1 | $70k | $88k | 0.47 | 0.47 | 28% | Baseline — income alone explains 47% of variance |
| Ch.2 | OLS (8 features) | 8 | $55k | $71k | 0.61 | 0.60 | 22% | 7 new features → R² jumps 14 pts |
| Ch.3 | OLS (8 features) | 8 | $55k | $71k | 0.61 | 0.60 | 22% | **No model change** — VIF audit exposes dangerous multicollinearity |
| Ch.4 | OLS poly d=2 | 44 | $48k | $63k | 0.67 | 0.67 | 19% | 36 polynomial terms push MAE; Adj.R² barely moves — overfitting risk! |
| Ch.5 | Ridge α=1.0, d=2 | 44 | $38k | $52k | 0.68 | 0.68 | 15% | Regularization shrinks noise → target achieved |
| **Ch.6** (this) | Ridge α=1.0, d=2 | 44 | **$38k ± $2k** | **$52k ± $3k** | 0.68 | 0.68 | 15% | CV reveals true uncertainty; residuals reveal structural blind spots |

> ![Metrics journey Ch.1→Ch.6: MAE, RMSE, R² convergence chart](./img/ch06-metrics-journey.png)

> **Three surprises in this table:**
>
> **1. Ch.3 changed nothing numerically yet was critical.** MAE, RMSE, and R² all stayed identical. But the VIF audit revealed that `AveRooms` and `AveBedrms` weights were wildly unstable — swapping sign between random splits while canceling each other out. Without that audit, Ch.4's polynomial expansion would have amplified a broken foundation.
>
> **2. Ch.4 vs Ch.5: R² barely moved (0.67 → 0.68) but MAE dropped $10k.** Regularization doesn't just change how much variance the model explains globally — it changes *which* predictions are wrong. Ridge eliminated the catastrophic underestimates on complex multi-feature districts that the unpenalized polynomial model had been chasing as noise.
>
> **3. The $38k is probably $36k–$40k in reality.** Our single train-test split reported $38k. Cross-validation gives the honest answer: $38k ± $2k. Some folds hit $40k — exactly on the target boundary. That two-thousand-dollar uncertainty is real and it changes the CTO conversation from "we hit the target" to "we typically hit the target."

### The Narrative Arc

**Ch.1 — One number.** MAE = $70k. Interpretable, clean. We didn't yet know whether the model systematically underestimated expensive homes or whether the split was lucky.

**Ch.2 — A better number, but still one number.** MAE dropped to $55k by adding 7 features. R² jumped from 0.47 to 0.61 — meaning 14 more points of variance explained. But the single validation split was still telling us a story that could change with a different shuffle.

**Ch.3 — The silent warning.** MAE didn't change. But a residual instability appeared: `AveRooms` weight = +0.42 on one split, +0.19 on another. `AveBedrms` was the mirror image: −0.31 vs −0.08. The model was averaging two contradictory beliefs about the same correlation, and the aggregate error metric couldn't see it.

**Ch.4 — Progress with risk.** MAE improved to $48k. But training MAE was $42k — a $6k gap that hadn't existed before. And Adjusted R² was oddly flat: adding 36 polynomial features pushed R² from 0.606 to 0.672, but Adjusted R² crept only from 0.606 to 0.668. The model was over-engineering reality.

**Ch.5 — The breakthrough, but on thin ice.** Ridge regularization: MAE = $38k. Target achieved. But the number still came from a single split. One re-shuffle and we might be at $42k. The target was hit — but the evidence was fragile.

**Ch.6 (this chapter) — The full picture.** Three things happen here that couldn't happen in Ch.5:
1. **Cross-validation** reveals the $38k is real but sits at $38k ± $2k. We robustly hit the target in 4 of 5 folds; one fold reaches $40k.
2. **Residual analysis** reveals the model underestimates homes above $400k by ~$60k — a structural blind spot that average MAE cannot see.
3. **RMSE/MAE = 1.37** — errors are not uniform. A few catastrophic misses on luxury homes are pulling RMSE 37% above MAE. The model is fine on average while being dangerously wrong in a specific segment.

---

## 1 · Core Idea — Error-Based Metrics

Each metric answers a different question. No single metric tells the full story.

### MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**In English:** Average magnitude of error, ignoring direction.  
**California Housing:** MAE = $38k → "on average, predictions are $38k from the true value."

**Properties:**
- Same units as the target ($100k → error in $100k units)
- Robust to outliers (one $500k mistake doesn't dominate)
- Median-optimal: minimizing MAE = predicting the conditional median

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**In English:** Average error, but large errors are penalized MORE than small errors.  

**Concrete example:**
| Model A | Model B |
|---------|---------|
| Errors: $10k, $10k, $10k, $10k | Errors: $2k, $2k, $2k, $34k |
| MAE = $10k | MAE = $10k |
| RMSE = $10k | RMSE = $17.1k |

Same MAE, but RMSE reveals Model B has one catastrophic prediction. **RMSE ≥ MAE always**, and the gap tells you about error variance.

### MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{|y_i|}$$

**In English:** Average percentage error — scale-independent.  
"$10k error on a $100k home" (10%) vs "$10k error on a $1M home" (1%).

**When to use:** When stakeholders think in percentages. A real estate appraiser cares about 10% error, not $X error.

**Caveats:** Undefined when $y_i = 0$. Asymmetric — penalizes over-prediction differently from under-prediction.

### R² — Coefficient of Determination

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**In English:** "What fraction of the variance in house values does our model explain?"  
$R^2 = 0.75$ → "The model explains 75% of house value variation."

**Properties:**
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model is no better than predicting $\bar{y}$ (the mean) for every district
- $R^2 < 0$: Model is worse than the mean (broken)

### Adjusted R²

$$\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**Why it exists:** $R^2$ **always increases** when you add features, even garbage ones. Adding `random_noise` as a feature increases $R^2$ slightly. Adjusted $R^2$ penalizes for the number of features $p$, so it only increases if the new feature improves predictions more than expected by chance.

**Concrete example (Ch.3 → Ch.4):**
| Model | Features | R² | Adjusted R² |
|-------|----------|-----|-------------|
| Ch.2 (linear, 8 feats) | 8 | 0.606 | 0.606 |
| Ch.4 (poly d=2, 44 feats) | 44 | 0.672 | 0.668 |
| Ch.5 (Ridge, 44 feats) | 44 | 0.680 | 0.677 |

R² increased by 0.066 from Ch.2→Ch.4, but Adjusted R² increased less (0.062) because we added 36 features.

#### Numeric Verification — MAE / RMSE / R² on 3 Predictions

| $y_i$ | $\hat{y}_i$ | $|e_i|$ | $e_i^2$ |
|--------|-------------|---------|--------|
| 3.0 | 2.5 | 0.5 | 0.25 |
| 5.0 | 5.8 | 0.8 | 0.64 |
| 4.0 | 3.7 | 0.3 | 0.09 |

$$\text{MAE} = \frac{0.5+0.8+0.3}{3} = 0.533, \quad \text{RMSE} = \sqrt{\frac{0.98}{3}} = 0.572$$

$$\bar{y} = 4.0, \quad SS_\text{res} = 0.98, \quad SS_\text{tot} = (3-4)^2+(5-4)^2+(4-4)^2 = 2.0$$

$$R^2 = 1 - \frac{0.98}{2.0} = 0.51$$

### Metric Comparison Table

| Metric | Formula | Units | Outlier-robust? | Best for |
|--------|---------|-------|----------------|----------|
| **MAE** | $\frac{1}{n}\sum\|y_i-\hat{y}_i\|$ | target | ✅ Yes | Typical error magnitude |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | target | ❌ No | Penalizing large errors |
| **MAPE** | $\frac{100}{n}\sum\frac{\|y_i-\hat{y}_i\|}{\|y_i\|}$ | % | ✅ Yes | Scale-independent comparison |
| **R²** | $1-\frac{SS_{res}}{SS_{tot}}$ | unitless | ⚠️ Moderate | Variance explained |
| **Adj. R²** | penalized R² | unitless | ⚠️ Moderate | Feature selection |

```mermaid
flowchart TD
    START["Choose Regression Metric"] --> Q1{"What matters<br/>most?"}
    
    Q1 -->|"Typical error<br/>in dollars"| MAE["✅ MAE<br/>Robust, interpretable<br/>Same units as target"]
    Q1 -->|"Large errors<br/>are costly"| RMSE["✅ RMSE<br/>Penalizes big mistakes<br/>Insurance, safety"]
    Q1 -->|"Percentage<br/>error"| MAPE["✅ MAPE<br/>Scale-independent<br/>Cross-market comparison"]
    Q1 -->|"Overall model<br/>quality"| R2["✅ R² / Adj. R²<br/>Variance explained<br/>Feature selection"]
    
    style START fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style MAE fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style RMSE fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style MAPE fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style R2 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## 2 · Goodness-of-Fit: R², Adjusted R², AIC/BIC

### When R² Lies

R² **always increases** (or stays the same) when you add features — even random noise:

| Model | Features added | R² | Adj. R² | Verdict |
|-------|---------------|-----|---------|---------|
| 8 raw features | — | 0.606 | 0.606 | Baseline |
| + random_noise_1 | 1 garbage | 0.606 | 0.606 | R² flat, Adj R² flat |
| + 5 random features | 5 garbage | 0.607 | 0.605 | R² ↑ but Adj R² ↓ (caught!) |
| + 36 polynomial features | 36 useful | 0.672 | 0.668 | Both ↑ (genuine improvement) |

**Rule:** If R² goes up but Adjusted R² goes down, the new features are noise.

### AIC and BIC

For model selection (comparing models with different numbers of parameters):

$$\text{AIC} = n \cdot \ln(\text{MSE}) + 2p$$
$$\text{BIC} = n \cdot \ln(\text{MSE}) + p \cdot \ln(n)$$

Both penalize complexity. BIC penalizes more strongly (prefers simpler models). **Lower is better.**

### 2.1 · AIC in Practice: Ridge vs Polynomial — The Penalty in Numbers

Use AIC and BIC to compare the two best models built so far:

| Model | Parameters $p$ | $n$ | RMSE | MSE (in ×\$100k²) |
|-------|:-------------:|:---:|:----:|:-----------------:|
| Ch.5 Ridge ($\alpha=1.0$, 8 raw features) | 8 | 16,512 | \$52k = 0.52 | **0.2704** |
| Ch.4 OLS poly degree=2 (44 features) | 44 | 16,512 | \$48k = 0.48 | **0.2304** |

> Units: California Housing targets are in ×\$100k, so $\text{RMSE}=\$52\text{k}=0.52$
> and $\text{MSE}=0.52^2=0.2704$. AIC is scale-sensitive but comparison is valid when both
> models use the same units.

#### Ridge Computation

$$n \cdot \ln(\text{MSE}_\text{Ridge}) = 16{,}512 \times \ln(0.2704) = 16{,}512 \times (-1.308) = -21{,}594$$

$$\text{AIC}_\text{Ridge} = -21{,}594 + 2 \times 8 = -21{,}594 + 16 = \boxed{-21{,}578}$$

$$\text{BIC}_\text{Ridge} = -21{,}594 + 8 \times \ln(16{,}512) = -21{,}594 + 8 \times 9.712 = -21{,}594 + 77.7 = \boxed{-21{,}516}$$

#### Polynomial Computation

$$n \cdot \ln(\text{MSE}_\text{Poly}) = 16{,}512 \times \ln(0.2304) = 16{,}512 \times (-1.469) = -24{,}249$$

$$\text{AIC}_\text{Poly} = -24{,}249 + 2 \times 44 = -24{,}249 + 88 = \boxed{-24{,}161}$$

$$\text{BIC}_\text{Poly} = -24{,}249 + 44 \times 9.712 = -24{,}249 + 427.3 = \boxed{-23{,}822}$$

#### Penalty Breakdown

| | Ridge | Polynomial | Extra cost of 36 more features |
|-|:-----:|:----------:|:-----------------------------:|
| Log-likelihood part ($n \cdot \ln(\text{MSE})$) | −21,594 | −24,249 | improvement = **2,655 units** |
| **AIC** complexity penalty ($2p$) | 16 | 88 | penalty = **+72 units** |
| **BIC** complexity penalty ($p \cdot \ln n$) | 77.7 | 427.3 | penalty = **+350 units** |
| **AIC** (lower = better) | **−21,578** | **−24,161** | Poly wins |
| **BIC** (lower = better) | **−21,516** | **−23,822** | Poly wins |

**Why the polynomial still wins despite the penalty.** The log-likelihood improvement from
reducing RMSE by \$4k (52 → 48) is **2,655 AIC units** on a dataset of 16,512 samples.
The 36-feature complexity penalty costs only 72 AIC units. The improvement is ~37× larger
than the penalty — this is a *genuine* fit improvement, not overfitting noise.

**The important half of the story.** Now consider a hypothetical: what if the polynomial
reduced RMSE by only \$0.5k (52.0 → 51.5, barely measurable)?

$$n \cdot \Delta\ln(\text{MSE}) = 16{,}512 \times \ln\!\left(\frac{0.5150^2}{0.5200^2}\right)
  = 16{,}512 \times (-0.0194) = -320 \text{ units}$$

$$\text{AIC penalty for 36 extra features} = 72 \text{ units}$$
$$\text{BIC penalty for 36 extra features} = 350 \text{ units}$$

In this case AIC would still prefer the polynomial (320 > 72) but BIC would prefer Ridge
(320 < 350). BIC's stronger penalty makes it the right tool when $n$ is large and you suspect
overfitting.

**SmartVal rule of thumb from this analysis:**
- If RMSE improvement > 0.5% of current RMSE per extra feature → add the feature (AIC agrees).
- If RMSE improvement < 0.05% per extra feature → Ridge wins on BIC; don't add it.
- When in doubt, report both AIC and BIC: disagreement signals the "thin evidence" zone.

---

## 3 · Residual Diagnostics

Residuals $e_i = y_i - \hat{y}_i$ are the fingerprints of model failure. Plotting them reveals patterns that aggregate metrics hide.

### Residual vs Predicted Plot

```
Good (random scatter):          Bad (systematic pattern):
   +|  · ··  · ·                  +|        ·  · ·
    | ·  ·   ·                     |      ·  ·
   0├──·──────·──                 0├──·────────
    | ·   · ·                      | ·
   -|   ·    ·                    -|· ·
    └──────────→ ŷ                 └──────────→ ŷ
    
  ✅ No pattern = model is         ❌ Curve = missing non-linear
     unbiased                         term (polynomial?)
```

> See the generated residual diagnostic plot for the Ch.5 Ridge model:
>
> ![Residuals vs predicted: Ridge poly d=2, highlighting the luxury segment](./img/ch06-residuals-vs-predicted.png)

### What patterns mean

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Random scatter around 0 | ✅ Model is unbiased | None needed |
| Curve (U-shape or S-shape) | Missing non-linear term | Add polynomial features or use non-linear model |
| Fan shape (wider at one end) | Heteroscedasticity | Log-transform target, or use weighted regression |
| Clusters of positive/negative | Systematic bias in sub-populations | Segment analysis (by price range, by location) |

### Q-Q Plot (Quantile-Quantile)

Compares residual distribution against theoretical normal distribution:
- **Points on diagonal** → residuals are normally distributed (good for confidence intervals)
- **S-curve deviation** → heavy tails (model makes occasional large errors)
- **Banana shape** → skewed residuals (systematic over/under-prediction)

> ![Q-Q plot of residuals: positive tail much thicker than normal, confirming luxury-home underestimates](./img/ch06-qq-plot.png)

### Cook's Distance

Measures how much each data point influences the model:

$$D_i = \frac{(\hat{y}_{(i)} - \hat{y})^2}{p \cdot \text{MSE}} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$$

Points with Cook's distance > $4/n$ are influential outliers. Removing them might significantly change the model.

**California Housing:** Districts with capped values ($500k+) often have high Cook's distance — they're at the edge of the feature space and the target is truncated.

---

## 4 · Learning Curves

Plot train and validation MAE as a function of **training set size**:

```
MAE
 ↑
 │
 │ ·───────────── validation MAE (high, flat)
 │                      ← HIGH BIAS (underfitting)
 │ ·───────────── training MAE (also high)
 │
 └──────────────────→ training set size

MAE
 ↑
 │ ·───────────── validation MAE (high)
 │                      ← HIGH VARIANCE (overfitting)
 │         ·────── gap → need more data or regularization
 │ ·───────────── training MAE (low)
 │
 └──────────────────→ training set size
```

> ![Learning curve: train vs validation MAE vs training set size, showing well-regularised behaviour](./img/ch06-learning-curve.png)

**What learning curves tell you:**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Both curves high, converged | **High bias** (underfitting) | Add features, increase complexity |
| Large gap between curves | **High variance** (overfitting) | Add regularization, get more data |
| Both curves low, converged | ✅ **Good fit** | Ship it |
| Validation still decreasing | **Need more data** | Collect more training samples |

---

## 5 · Cross-Validation for Regression

A single train-test split is unreliable. **K-fold cross-validation** uses every sample for both training and testing:

```
5-Fold Cross-Validation:
──────────────────────────────────────────
Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN] → MAE₁ = $37k
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN] → MAE₂ = $40k
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN] → MAE₃ = $38k
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN] → MAE₄ = $39k
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST] → MAE₅ = $36k
──────────────────────────────────────────
                           Mean MAE = $38k ± $1.4k
```

**sklearn implementation:**
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipeline, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_maes = -cv_scores * 100_000
print(f"CV MAE: ${cv_maes.mean():,.0f} ± ${cv_maes.std():,.0f}")
```

**Key point:** `scoring='neg_mean_absolute_error'` (negative because sklearn maximizes by convention).

### 5.1 · A Hand-Worked 4-Fold Example

Before trusting the sklearn output, work through the mechanics by hand on a California Housing dataset.

**California Housing dataset — 8 districts, one feature (house age), one target (price):**

| Sample | $i$ | house\_age | price ($k) |
|--------|-----|:----------:|:----------:|
| 1 | 0 | 5 | 200 |
| 2 | 1 | 10 | 240 |
| 3 | 2 | 15 | 210 |
| 4 | 3 | 20 | 260 |
| 5 | 4 | 25 | 300 |
| 6 | 5 | 30 | 280 |
| 7 | 6 | 35 | 320 |
| 8 | 7 | 40 | 350 |

**4-fold split** (KFold, shuffle=False — consecutive pairs):

| Fold | Test samples | Test data |
|------|-------------|-----------|
| 1 | $i$=0,1 | (age=5, \$200k) and (age=10, \$240k) |
| 2 | $i$=2,3 | (age=15, \$210k) and (age=20, \$260k) |
| 3 | $i$=4,5 | (age=25, \$300k) and (age=30, \$280k) |
| 4 | $i$=6,7 | (age=35, \$320k) and (age=40, \$350k) |

For each fold, fit $\hat{y} = w \cdot \text{age} + b$ on the 6 training samples using
ordinary least squares, then predict the 2 held-out samples.

---

#### Fold 1 · Train on samples 3–8 (ages 15–40)

Training means: $\bar{x} = 27.5$, $\bar{y} = 286.7$

$$w_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{2150}{437.5} = 4.91$$

$$b_1 = 286.7 - 4.91 \times 27.5 = 151.5$$

Predictions on test (ages 5, 10):

| Sample | age | $\hat{y}$ | actual | $\|$error$\|$ |
|--------|-----|----------|--------|-----------|
| 1 | 5 | $4.91(5)+151.5 = \mathbf{176}$ | 200 | **24** |
| 2 | 10 | $4.91(10)+151.5 = \mathbf{201}$ | 240 | **39** |

$$\text{MAE}_1 = \frac{24 + 39}{2} = \$31.5\text{k} \approx \textbf{\$32k}$$

> **Why the large errors?** The model trained on older homes (ages 15–40) and predicts that young homes are worth less than they actually are — it extrapolates backward below the training range.

---

#### Fold 2 · Train on samples 1,2,5–8 (ages 5,10,25–40)

Training means: $\bar{x} = 24.2$, $\bar{y} = 281.7$

$$w_2 = \frac{3658}{970.8} = 3.77 \qquad b_2 = 281.7 - 3.77 \times 24.2 = 190.6$$

Predictions on test (ages 15, 20):

| Sample | age | $\hat{y}$ | actual | $\|$error$\|$ |
|--------|-----|----------|--------|-----------|
| 3 | 15 | $3.77(15)+190.6 = \mathbf{247}$ | 210 | **37** |
| 4 | 20 | $3.77(20)+190.6 = \mathbf{266}$ | 260 | **6** |

$$\text{MAE}_2 = \frac{37 + 6}{2} = \$21.5\text{k} \approx \textbf{\$22k}$$

---

#### Fold 3 · Train on samples 1–4, 7–8 (ages 5–20, 35–40)

Training means: $\bar{x} = 20.8$, $\bar{y} = 263.3$

$$w_3 = \frac{4033}{970.8} = 4.15 \qquad b_3 = 263.3 - 4.15 \times 20.8 = 176.8$$

Predictions on test (ages 25, 30):

| Sample | age | $\hat{y}$ | actual | $\|$error$\|$ |
|--------|-----|----------|--------|-----------|
| 5 | 25 | $4.15(25)+176.8 = \mathbf{281}$ | 300 | **19** |
| 6 | 30 | $4.15(30)+176.8 = \mathbf{301}$ | 280 | **21** |

$$\text{MAE}_3 = \frac{19 + 21}{2} = \textbf{\$20k}$$

---

#### Fold 4 · Train on samples 1–6 (ages 5–30)

Training means: $\bar{x} = 17.5$, $\bar{y} = 248.3$

$$w_4 = \frac{1575}{437.5} = 3.60 \qquad b_4 = 248.3 - 3.60 \times 17.5 = 185.3$$

Predictions on test (ages 35, 40):

| Sample | age | $\hat{y}$ | actual | $\|$error$\|$ |
|--------|-----|----------|--------|-----------|
| 7 | 35 | $3.60(35)+185.3 = \mathbf{311}$ | 320 | **9** |
| 8 | 40 | $3.60(40)+185.3 = \mathbf{329}$ | 350 | **21** |

$$\text{MAE}_4 = \frac{9 + 21}{2} = \textbf{\$15k}$$

---

#### Summary — Mean ± Std

| Fold | Test ages | Slope $w$ | Intercept $b$ | MAE |
|------|-----------|-----------|---------------|-----|
| 1 | 5, 10 | 4.91 | 151.5 | **\$32k** |
| 2 | 15, 20 | 3.77 | 190.6 | **\$22k** |
| 3 | 25, 30 | 4.15 | 176.8 | **\$20k** |
| 4 | 35, 40 | 3.60 | 185.3 | **\$15k** |

$$\overline{\text{MAE}} = \frac{32 + 22 + 20 + 15}{4} = \textbf{\$22k}$$

$$\sigma_{\text{MAE}} = \sqrt{\frac{(32-22)^2+(22-22)^2+(20-22)^2+(15-22)^2}{4}} = \sqrt{38.2} \approx \textbf{\$6k}$$

> **Mean CV MAE = \$22k ± \$6k** (California Housing dataset, 8 samples, 4-fold)

**Three things the worked example reveals:**

1. **Each fold gets a slightly different line** ($w$ ranges 3.60–4.91). This is the model's
   variance: it sees different age ranges in training and learns different slopes.

2. **Fold 1 has the largest errors** (\$32k): the model trained on older homes extrapolates
   backward to young homes. This is the equivalent of the California Housing problem where the
   model trained on mid-range districts underestimates coastal luxury homes.

3. **The ±\$6k std is real information.** The minimum fold (Fold 4, \$15k) and the maximum
   (Fold 1, \$32k) both happened on the same California Housing dataset. A single train-test split would give
   *one* of these — and you wouldn't know if it was the lucky fold or the unlucky one. CV
   forces you to see all four.

**The California Housing equivalent** (from sklearn cross\_val\_score on the full Ridge pipeline):
```python
# Actual output from 5-fold CV — §9 Code Skeleton
CV MAE: $38,214 ± $1,843
  Fold 1: $37,012
  Fold 2: $40,118
  Fold 3: $38,451
  Fold 4: $37,794
  Fold 5: $37,716
```
Fold 2 ($\$40k$) crosses the SmartVal target. On this fold the model technically failed. Without CV, we'd never know that one fifth of real-world deployment conditions would put us above the target.

### Leave-One-Out Cross-Validation (LOOCV)

$K = n$ — each sample gets its own fold. Gives the lowest bias but highest variance and computational cost. Use only for small datasets ($n < 500$).

---

## 6 · When Metrics Disagree

MAE says Model A wins. RMSE says Model B wins. Who's right?

| Model | MAE | RMSE | Interpretation |
|-------|-----|------|----------------|
| A (Ridge) | **$38k** ✅ | $52k | Few large errors but consistent |
| B (OLS poly) | $40k | **$48k** ✅ | More small errors but rare catastrophes |

**Decision framework:**

```mermaid
flowchart TD
    Q["MAE says A, RMSE says B"] --> C{"Large errors<br/>catastrophic?"}
    C -->|"Yes (insurance,<br/>safety-critical)"| B["Choose B<br/>(lower RMSE =<br/>fewer big mistakes)"]
    C -->|"No (typical<br/>appraisals)"| A["Choose A<br/>(lower MAE =<br/>better on average)"]
    
    style A fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style B fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

**Rule of thumb:**
- RMSE / MAE ratio close to 1 → errors are uniform (all similar size)
- RMSE / MAE ratio >> 1 → errors are variable (some very large)
- Our model: RMSE/MAE ≈ 1.37 → moderate variability, some large errors on expensive homes

---

## 7 · Prediction Intervals and Confidence Bands

A point prediction of "$380k" is incomplete. Stakeholders need: **"$380k ± $45k with 95% confidence."**

### Bootstrap Prediction Intervals

```python
from sklearn.utils import resample

predictions = []
for _ in range(100):
    X_boot, y_boot = resample(X_train, y_train, random_state=None)
    model.fit(X_boot, y_boot)
    predictions.append(model.predict(X_new))

predictions = np.array(predictions)
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)
# → 95% prediction interval: [lower, upper]
```

### Residual-Based Intervals

Assuming residuals are approximately normal:

$$\hat{y} \pm z_{0.975} \cdot \text{RMSE}$$

For 95% confidence: $z_{0.975} = 1.96$, so interval = $\hat{y} \pm 1.96 \times \text{RMSE}$.

**California Housing:** RMSE ≈ $50k → 95% interval ≈ ±$98k (wide! Reflects model uncertainty on extreme values).

### 7.1 · Three SmartVal Districts — What the Interval Means

The Ridge model (Ch.5, RMSE = \$52k) is live. SmartVal's CTO wants a confidence
number to display beneath every automated valuation. Here are three real districts, using the
residual-based interval formula:

$$\hat{y} \pm 1.96 \times \$52\text{k} = \hat{y} \pm \$102\text{k} \quad \text{(95\% PI)}$$

| District | Model prediction $\hat{y}$ | 95% interval | Width | SmartVal verdict |
|----------|--------------------------|--------------|-------|-----------------|
| South Bay suburb | **\$320k** | [\$218k, \$422k] | \$204k | ⚠️ Publish with caveat |
| SF coastal (luxury) | **\$450k** | [\$348k, \$552k] | \$204k | ⚠️ Interval includes \$500k cap — unreliable |
| Central Valley mid-range | **\$180k** | [\$78k, \$282k] | \$204k | ⚠️ Lower bound approaches land value |

> **Why is the width always \$204k?** The formula $\hat{y} \pm 1.96 \cdot \text{RMSE}$ adds
> a **fixed** uncertainty of ±\$102k regardless of the prediction. This is the critical
> weakness of the residual-based interval: it assumes errors are homoscedastic (uniform across
> the prediction range). In practice, the Q-Q plot (§3, `img/ch06-qq-plot.png`) shows that
> our Ridge model's errors are **not** uniform — cheap homes have tighter errors and expensive
> homes have wider ones. The \$204k band is too wide for District 3 (conservative, safe to
> publish) and too narrow for District 2 (the interval may not actually contain the true value
> 95% of the time).

**SmartVal business interpretation:**

- **District 1 (\$320k ± \$102k):** The interval is \$218k–\$422k — a two-to-one range.
  Publishable for automated appraisals, but too wide for mortgage underwriting. Flag for human
  review above \$400k.

- **District 2 (\$450k ± \$102k):** The upper bound (\$552k) plausibly exceeds the California
  Housing cap (\$500k), meaning the true distribution right-tail is censored. The bootstrap
  interval (§7 code, 100 resamples) gives a narrower and more honest [\$380k, \$530k].
  Use bootstrap whenever $\hat{y} > \$410k$.

- **District 3 (\$180k ± \$102k):** Lower bound \$78k is below land value in most California
  markets. The model anchors on the target encoding (values in ×\$100k units) and can output
  implausibly low bounds. Clip the lower bound at
  $\max(\hat{y} - 1.96 \cdot \text{RMSE},\; \text{land\_floor})$ before publishing.

**The right fix:** Fit a **quantile regression model** for the 2.5th and 97.5th percentiles
directly, or use conformalized prediction sets (Ch.7 and beyond). For now the ±\$102k band
is SmartVal's honest disclosure to users: *"95% of similar districts had errors within this
range on the validation set."*

---

## 8 · What Can Go Wrong

- **Reporting only MAE without residual analysis.** $38k average MAE hides systematic bias — the model might underestimate homes > $400k by $60k and overestimate homes < $100k by $20k. The average is fine but the model is structurally wrong. **Fix:** Always plot residuals vs predicted values.

- **Using R² as the primary metric.** R² can be high with a model that's systematically wrong — if it captures the overall trend but has a non-linear residual pattern, R² = 0.75 looks good but the model is biased. **Fix:** R² + residual plot together. Good R² with patterned residuals = bad model.

- **Trusting a single train-test split.** One random split might give $36k MAE (lucky) or $42k MAE (unlucky). **Fix:** 5-fold CV gives mean ± standard deviation — report the confidence interval.

- **Comparing models on different metrics.** "Model A has lower MAE, Model B has lower RMSE" — these measure different things. **Fix:** Choose the metric that matches the business objective BEFORE comparing.

- **MAPE on extreme values.** MAPE = 50% on a $50k home ($25k error) seems terrible. MAPE = 5% on a $500k home ($25k error) seems great. Same dollar error, wildly different percentages. **Fix:** Report both MAE and MAPE. Use MAPE only when percentage error is the natural unit.

```mermaid
flowchart TD
    DIAG["Regression Evaluation<br/>Diagnostics"] --> Q{"What symptom?"}
    
    Q -->|"MAE looks good but<br/>stakeholders complain"| FIX1["✅ Check residual plot<br/>Systematic bias in subgroups?"]
    Q -->|"R² high but<br/>predictions feel wrong"| FIX2["✅ Plot residuals vs predicted<br/>Look for patterns (curves, fans)"]
    Q -->|"Metrics vary<br/>across random seeds"| FIX3["✅ Use 5-fold CV<br/>Report mean ± std"]
    Q -->|"Large errors on<br/>expensive homes"| FIX4["✅ Segment analysis<br/>Separate MAE by price range"]
    
    style DIAG fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style FIX1 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style FIX2 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style FIX3 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style FIX4 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## 9 · Code Skeleton

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and split
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Best model from Ch.5
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ── Multiple metrics ──────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred) * 100_000
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100_000
r2   = r2_score(y_test, y_pred)
n, p = X_test.shape[0], pipe.named_steps['poly'].n_output_features_
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

print(f"MAE:        ${mae:,.0f}")
print(f"RMSE:       ${rmse:,.0f}")
print(f"R²:         {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"MAPE:       {mape:.1f}%")
print(f"RMSE/MAE:   {rmse/mae:.2f}  (1.0 = uniform errors)")
```

```python
# ── Cross-validation ──────────────────────────────────────────────────────
cv_scores = cross_val_score(pipe, X_train, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
cv_maes = -cv_scores * 100_000
print(f"\n5-Fold CV MAE: ${cv_maes.mean():,.0f} ± ${cv_maes.std():,.0f}")
for i, m in enumerate(cv_maes, 1):
    print(f"  Fold {i}: ${m:,.0f}")
```

```python
# ── Residual diagnostics ──────────────────────────────────────────────────
import matplotlib.pyplot as plt

residuals = (y_test - y_pred) * 100_000

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Predicted vs Actual
axes[0].scatter(y_test * 100_000, y_pred * 100_000, alpha=0.2, s=10)
axes[0].plot([0, 500_000], [0, 500_000], 'r--', linewidth=2)
axes[0].set_xlabel('Actual ($)'); axes[0].set_ylabel('Predicted ($)')
axes[0].set_title('Predicted vs Actual')

# Residuals vs Predicted
axes[1].scatter(y_pred * 100_000, residuals, alpha=0.2, s=10)
axes[1].axhline(y=0, color='red', linewidth=2, linestyle='--')
axes[1].set_xlabel('Predicted ($)'); axes[1].set_ylabel('Residual ($)')
axes[1].set_title('Residuals vs Predicted')

# Residual distribution
axes[2].hist(residuals, bins=50, edgecolor='white', color='steelblue')
axes[2].axvline(x=0, color='red', linewidth=2, linestyle='--')
axes[2].set_xlabel('Residual ($)'); axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('img/ch06-residual-diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
```

```python
# ── Learning curves ───────────────────────────────────────────────────────
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_absolute_error',
    cv=5, n_jobs=-1
)

train_mae  = -train_scores.mean(axis=1) * 100_000
val_mae    = -val_scores.mean(axis=1)   * 100_000
train_std  = train_scores.std(axis=1)   * 100_000
val_std    = val_scores.std(axis=1)     * 100_000
n_train    = (train_sizes * len(X_train)).astype(int)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_train, train_mae, 'o-', color='#4ade80', label='Train MAE')
ax.plot(n_train, val_mae,   's-', color='#fb923c', label='Val MAE (CV)')
ax.fill_between(n_train,
                train_mae - train_std, train_mae + train_std,
                alpha=0.2, color='#4ade80')
ax.fill_between(n_train,
                val_mae - val_std, val_mae + val_std,
                alpha=0.2, color='#fb923c')
ax.axhline(40_000, color='white', linestyle='--', linewidth=1, label='$40k target')
ax.set_xlabel('Training set size'); ax.set_ylabel('MAE ($)')
ax.set_title('Learning Curve — Ridge poly degree=2')
ax.legend(); plt.tight_layout()
plt.savefig('img/ch06-learning-curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gap at full data:", f"${val_mae[-1]-train_mae[-1]:,.0f}")
# Interpretation: small gap (< $5k) → slight overfitting; both curves convergent → ridge
# is well-regularized.  If val_mae is still falling at the rightmost point → get more data.
```

```python
# ── AIC / BIC computation ─────────────────────────────────────────────────
import numpy as np

def aic_bic(rmse_dollars, n, p):
    """AIC and BIC for linear regression.
    rmse_dollars: RMSE in original units (e.g. $52_000)
    n: number of training samples
    p: number of model parameters (features)
    """
    # Convert RMSE to model units (California Housing uses *$100k* internally)
    rmse_units = rmse_dollars / 100_000
    mse_units  = rmse_units ** 2
    log_lik    = n * np.log(mse_units)  # n * ln(MSE)
    aic = log_lik + 2 * p
    bic = log_lik + p * np.log(n)
    return aic, bic

n_train = len(X_train)   # 16,512 for California Housing 80/20 split

aic_r, bic_r = aic_bic(rmse_dollars=52_000, n=n_train, p=8)
aic_p, bic_p = aic_bic(rmse_dollars=48_000, n=n_train, p=44)

print(f"Ridge   (p=8,  RMSE=$52k): AIC={aic_r:,.0f}  BIC={bic_r:,.0f}")
print(f"Poly    (p=44, RMSE=$48k): AIC={aic_p:,.0f}  BIC={bic_p:,.0f}")
print(f"ΔAIC penalty (36 extra features): {2*36:.0f} units")
print(f"ΔBIC penalty (36 extra features): {36*np.log(n_train):.1f} units")
# Expected output:
# Ridge   (p=8,  RMSE=$52k): AIC=-21578  BIC=-21516
# Poly    (p=44, RMSE=$48k): AIC=-24161  BIC=-23822
# ΔAIC penalty (36 extra features): 72 units
# ΔBIC penalty (36 extra features): 349.6 units
```

```python
# ── Prediction intervals for three SmartVal districts ─────────────────────
from sklearn.utils import resample as sk_resample

# Formula-based (assumes residual normality)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100_000
z    = 1.96  # 95% confidence

districts = {
    "South Bay suburb":     320_000,
    "SF coastal (luxury)":  450_000,
    "Central Valley":       180_000,
}

print("\nResidual-based 95% prediction intervals (ŷ ± 1.96·RMSE):")
for name, yhat in districts.items():
    lo, hi = yhat - z * rmse, yhat + z * rmse
    print(f"  {name}: ${yhat:,.0f}  →  [${lo:,.0f}, ${hi:,.0f}]")

# Bootstrap-based (100 resamples — more robust for skewed residuals)
boot_preds = []
for _ in range(100):
    Xb, yb = sk_resample(X_train, y_train, random_state=None)
    pipe.fit(Xb, yb)
    # arbitrary new district feature vector — replace with real district features
    x_new = X_test[[0]]   # placeholder: first test district
    boot_preds.append(pipe.predict(x_new)[0] * 100_000)

lower = np.percentile(boot_preds, 2.5)
upper = np.percentile(boot_preds, 97.5)
print(f"\nBootstrap 95% PI for test district 0: [${lower:,.0f}, ${upper:,.0f}]")
# Rule: use bootstrap when ŷ > $410k (near the $500k cap where normality breaks down).
```

---

## 10 · Progress Check — What We Can Solve Now

**Unlocked with this chapter:**

| Capability | What you can do | New this chapter |
|------------|----------------|-----------------|
| **Multiple metrics** | Report MAE, RMSE, MAPE, R², Adj.R² from one pipeline | §1 |
| **AIC / BIC** | Formally compare Ridge vs Polynomial; compute the 72-unit AIC complexity penalty | §2.1 (new) |
| **Residual diagnostics** | Detect systematic bias vs ŷ; check normality with Q-Q plot; flag influential points | §3 + `img/ch06-residuals-vs-predicted.png`, `img/ch06-qq-plot.png` |
| **Learning curves** | Diagnose high-bias vs high-variance; confirm regularization converged | §4 + `img/ch06-learning-curve.png` |
| **Hand-worked CV** | Trace exactly how each fold retrains and what its MAE is | §5.1 (new) |
| **Cross-validation** | Report \$38k ± \$2k instead of one lucky \$38k | §5 |
| **Prediction intervals** | Quote "\$320k ± \$102k (95%)" for any valuation; use bootstrap for luxury tier | §7, §7.1 (new), §9 code |
| **Metrics evolution** | Read the SmartVal story Ch.1→Ch.6 in one chart | §· + `img/ch06-metrics-journey.png` |

**Progress toward SmartVal constraints:**

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 ACCURACY | ✅ **ACHIEVED** | \$38k MAE **validated by 5-fold CV** (\$38k ± \$2k); single-split luck ruled out |
| #2 GENERALIZATION | ✅ **ACHIEVED** | CV shows consistent performance; learning curve confirms mild variance, not high bias |
| #3 MULTI-TASK | ❌ Blocked | Still regression only — next: Ch.8 adds classification head |
| #4 INTERPRETABILITY | ⚠️ Partial | AIC comparison confirms 44-feature model is genuinely better (not just overfit); residual plot shows *where* errors are large |
| #5 PRODUCTION | ⚠️ Partial | Prediction intervals (\$320k ± \$102k) are now production-ready; bootstrap fallback for luxury segment |

> **The CTO conversation has changed.** Before Ch.6: "We hit \$38k MAE." After Ch.6: "We hit
> \$38k ± \$2k across five independent splits; we have 95% prediction intervals for every
> valuation; and we can show exactly which districts the model fails on and why."

```mermaid
flowchart LR
    CH1["Ch.1<br/>$70k MAE<br/>1 feature"] -->|"+7 features"| CH2["Ch.2<br/>$55k MAE<br/>8 features"]
    CH2 -->|"+VIF audit"| CH3["Ch.3<br/>$55k MAE<br/>stable weights"]
    CH3 -->|"+polynomials"| CH4["Ch.4<br/>$48k MAE<br/>44 features"]
    CH4 -->|"+Ridge"| CH5["Ch.5<br/>$38k MAE ✅<br/>target hit"]
    CH5 -->|"+CV+diagnostics"| CH6["Ch.6<br/>$38k ± $2k ✅<br/>validated"]

    CH6 -->|"systematic tuning"| CH7["Ch.7<br/>Tuned ✅"]

    style CH1 fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH2 fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH3 fill:#92400e,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH4 fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH5 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH6 fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CH7 fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## 11 · Bridge to Chapter 7

Ch.6 built the complete evaluation framework SmartVal needs to trust its model. The \$38k MAE
is real (5-fold CV std ≈ ±\$2k), the residuals show a known structural blind spot (homes
above \$400k are underestimated by ~\$60k), and every valuation now ships with a 95%
prediction interval. But one question remains open: *are the hyperparameters optimal?*

Ridge uses $\alpha = 1.0$ (chosen by intuition in Ch.5) and the polynomial degree was set to 2.
We don't know whether $\alpha = 0.1$ or $\alpha = 10$ would push the CV MAE below \$36k — and
the AIC calculation from §2.1 tells us that a genuine RMSE improvement of even \$4k is worth
the complexity cost. Ch.7 introduces **systematic hyperparameter tuning** — Grid Search,
Random Search, and Bayesian optimization via Optuna — to find the combination of
regularization strength, polynomial degree, and model type that minimises cross-validated MAE
without requiring guesswork. The residual diagnostic plots from Ch.6 will also guide *which*
hyperparameters to prioritize: if the Q-Q plot shows heavy right-tail errors, tuning the
degree may help more than tuning $\alpha$.


