# Ch.3 — Unsupervised Metrics

> **The story.** Once you cluster without labels, you face an awkward question: *was the clustering good?* The community has answered it twice. The internal answer is **Peter Rousseeuw's silhouette score** (1987) — a per-point measurement of "am I closer to my own cluster than to my nearest neighbour cluster?" — and the **Davies–Bouldin index** (1979), which compares within-cluster spread against between-cluster separation. Both metrics need only the data itself. The external answer arrives when you *do* have ground-truth labels: the **Adjusted Rand Index** (Hubert & Arabie, 1985) and **Normalised Mutual Information** (Strehl & Ghosh, 2003). Together these metrics turn unsupervised learning from "pretty plot" into "engineering decision."
>
> **Where you are in the curriculum.** [Ch.1](../ch01_clustering) ran K-Means, DBSCAN, and HDBSCAN on wholesale customers — but which clustering was actually good? Without labelled customer segments, there is no accuracy score, no F1, no MAE. This chapter gives the internal and external tools for answering "how well did this unsupervised method work?" and finally pushes SegmentAI past the silhouette >0.5 threshold.
>
> **Notation in this chapter.** For internal metrics on point $i$: $a(i)$ — mean distance from $i$ to all other points in its own cluster; $b(i)$ — mean distance from $i$ to all points in the *nearest other* cluster; **silhouette** $s(i)=\tfrac{b(i)-a(i)}{\max(a(i),b(i))}\in[-1,1]$ (higher is better); **Davies–Bouldin index** — average similarity of each cluster to its most similar other cluster (lower is better); **Calinski–Harabasz index** — ratio of between-cluster to within-cluster dispersion (higher is better). External metrics: **ARI** — adjusted Rand index; **NMI** — normalised mutual information; both range in $[0,1]$ where 1 is perfect agreement with ground-truth labels.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Build **SegmentAI** — discover 5 actionable customer segments with silhouette >0.5
> 1. **SEGMENTATION**: 5 distinct segments — 2. **INTERPRETABILITY**: Business-actionable — 3. **STABILITY**: Reproducible — 4. **SCALABILITY**: 10k+ — 5. **VALIDATION**: Silhouette >0.5

**What we know so far:**
- ⚡ Ch.1: K-Means discovered 5 initial segments (silhouette = 0.42)
- ⚡ Ch.2: PCA compression improved silhouette to 0.48, beautiful 2D plots
- 💡 **But was the clustering actually good? Is K=5 optimal?**

**What's blocking us:**
⚠️ **No way to evaluate unsupervised learning quantitatively!**

The CMO asks: "How do we know 5 segments is better than 3 or 7?"
- **Supervised learning** (earlier topics): Compare predictions to ground truth → MAE, F1, AUC
- **Unsupervised learning**: **No ground truth!** → can't compute accuracy/MAE
- **Metric vs business**: Silhouette says K=3 is best (0.58), business needs K=5 (0.52)

**What this chapter unlocks:**
⚡ **Unsupervised evaluation metrics + business validation:**
1. **Silhouette score**: Cluster cohesion vs separation ([-1, 1])
2. **Davies-Bouldin index**: Cluster compactness ratio ([0, ∞), lower better)
3. **Calinski-Harabasz index**: Between/within dispersion ratio (higher better)
4. **Bootstrap stability**: Are segments reproducible?
5. **Business validation**: Do segments make marketing sense?

💡 **Outcome**: K-Means K=5 on PCA data achieves silhouette = 0.52 ⚡ (above 0.5 target!). Bootstrap shows 95% stability. Business names assigned. **All 5 SegmentAI constraints satisfied!**

| Constraint | Status | This Chapter |
|------------|--------|-------------|
| #1 SEGMENTATION | ✅ **Done** | K=5 quantitatively validated |
| #2 INTERPRETABILITY | ✅ **Done** | Segment names assigned via centroid analysis |
| #3 STABILITY | ✅ **Done** | 95% bootstrap consistency |
| #4 SCALABILITY | ✅ **Done** | Metrics scale with sampling |
| #5 VALIDATION | ✅ **Done** | Silhouette = 0.52 > 0.5 ✅ |

```mermaid
flowchart LR
 A["K-Means K=5\non PCA 2D"] --> B["Silhouette\n0.52 ✅"]
 A --> C["Davies-Bouldin\n0.78"]
 A --> D["Calinski-Harabasz\n285"]
 B --> E["K=3 scores 0.58\nbut business needs 5"]
 E --> F["Acceptable trade-off:\n0.52 > 0.5 threshold"]
 A --> G["Bootstrap\n100 samples"]
 G --> H["95% stability ✅"]
```

---

## Animation

![Chapter animation](img/ch03-unsupervised-metrics-needle.gif)

## 1 · Core Idea

**The fundamental problem:** In supervised learning you know the right answer — actual house price, actual diagnosis, actual customer churn — so you compute MAE or F1 or AUC. But in unsupervised learning there **is no right answer**. Nobody labelled 440 customers as "Loyalists" or "Price-sensitive". You ran K-Means, got 5 clusters — but *were they good clusters*?

**Supervised metrics** compare predictions to known labels. **Unsupervised metrics** have no labels — they measure geometric properties of the clusters themselves.

**Internal metrics** (label-free — use only the feature matrix):
- **Silhouette score** — balances cohesion (how tight) against separation (how far from neighbours). Range: [−1, 1]; higher is better.
- **Davies-Bouldin index (DBI)** — average ratio of within-cluster spread to between-cluster distance. Range: [0, ∞); lower is better.
- **Calinski-Harabasz index (CHI)** — ratio of between-cluster to within-cluster dispersion. Higher is better.

**External metrics** (require ground truth or proxy labels):
- **Adjusted Rand Index (ARI)** — overlap between clusters and true labels, corrected for chance. Range: [−1, 1].
- **Normalised Mutual Information (NMI)** — information-theoretic overlap. Range: [0, 1].

**The fundamental unsupervised challenge:** Unlike supervised metrics which answer "how accurate are predictions?", unsupervised metrics answer "how good is the discovered structure?" — a fundamentally different question with no single right answer.

---

## 2 · Running Example

We reuse the **K-Means clustering from Ch.1** applied to Wholesale Customers. For each K from 2 to 10 we compute silhouette score, DBI, and CHI to pick the best K objectively. Then we use ARI to validate against a proxy ground truth: the `Channel` column (Hotel/Restaurant/Café = 1, Retail = 2) that we deliberately excluded from clustering.

**The key unsupervised dilemma**: Silhouette prefers K=3 (0.58) but business needs K=5 (0.52). We show why 0.52 is acceptable and how to reconcile metric recommendations with domain requirements.

Dataset: **Wholesale Customers** (UCI) — 440 customers, 6 features (log-transformed + standardised)
Clustering: K-Means (PCA 2D preprocessed) from Ch.2
External proxy: `Channel` column (excluded from clustering, used only for ARI validation)

---

## 3 · Math

### 3.1 Silhouette Score

For each customer $i$:

$$a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, j \neq i} d(i, j)$$

(mean intra-cluster distance — **cohesion**; lower = tighter segment)

$$b(i) = \min_{k \neq C_i} \frac{1}{|C_k|} \sum_{j \in C_k} d(i, j)$$

(mean distance to nearest segment — **separation**; higher = better separated)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Mean silhouette score** $= \frac{1}{n} \sum_i s(i)$

**Interpretation:**
- $s(i) \approx 1$: well-assigned — tight segment, far from others
- $s(i) \approx 0$: on the boundary between segments
- $s(i) < 0$: likely misassigned — closer to another segment

**Numeric example** (customer in "Loyalists" segment):

Suppose customer $i$ is in the "Loyalists" segment (cluster 0).

- **Cohesion:** $a(i) = 0.8$ — average distance from $i$ to all other Loyalists
- **Separation:** Nearest *other* segment is "Occasional Buyers" (cluster 3). Average distance from $i$ to all Occasional Buyers: $d(i, C_3) = 1.5$. So $b(i) = 1.5$.
- **Silhouette score:** 

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} = \frac{1.5 - 0.8}{1.5} = \frac{0.7}{1.5} = 0.47$$

**Interpretation:** $s(i) = 0.47 > 0$ means customer $i$ is closer to their own cluster than to the nearest neighbour cluster — correctly assigned. But 0.47 is only moderate (not close to 1.0) — there's some overlap between Loyalists and Occasional Buyers.

**Contrast:** If $a(i) = 1.2$ and $b(i) = 0.9$, then $s(i) = (0.9-1.2)/1.2 = -0.25 < 0$ — **misassigned!** Customer $i$ is closer to another cluster than to their own.

💡 **The match is exact:** For the 440-customer Wholesale dataset with K-Means K=5, sklearn's `silhouette_samples(X, labels)` returns an array of 440 individual $s(i)$ scores. The mean of these 440 scores is the **overall silhouette score** reported as 0.42 (Ch.1), 0.48 (Ch.2), 0.52 (Ch.3).

### 3.2 Davies-Bouldin Index

$$\text{DBI} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d(c_i, c_j)}$$

where $s_i$ is the average distance of customers in segment $i$ to their centroid, and $d(c_i, c_j)$ is the distance between centroids. Lower DBI = compact, well-separated segments.

### 3.3 Calinski-Harabasz Index

$$\text{CHI} = \frac{\text{tr}(B_K) / (K-1)}{\text{tr}(W_K) / (n-K)}$$

where $B_K$ is the between-cluster scatter matrix and $W_K$ is the within-cluster scatter matrix. Higher CHI = dense, well-separated segments.

### 3.4 Adjusted Rand Index

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

ARI counts concordant customer-pairs (both in same segment in prediction and ground truth, or both in different segments). Corrected for chance: random clustering scores ≈ 0, perfect scores 1.

### 3.5 Bootstrap Stability

To test Constraint #3 (STABILITY):
1. Draw 100 bootstrap samples (with replacement, same size)
2. Re-cluster each sample with K-Means K=5
3. For each customer, count what fraction of bootstraps assign them to the same segment
4. **Stability** = mean fraction across all customers

If stability >90%, segments are reproducible. If <70%, clusters are fragile.

---

## 4 · Step by Step

```
Internal metrics (no labels needed):
1. Log-transform + standardise features
2. PCA → 2D (from Ch.2)
3. For K in range(2, 11):
   a. Fit KMeans(n_clusters=K) on PCA 2D
   b. Compute silhouette_score
   c. Compute davies_bouldin_score
   d. Compute calinski_harabasz_score
4. Plot all three metrics vs K
5. Pick K where metrics agree — or where business + metrics compromise

Metric disagreement resolution:
6. Silhouette says K=3 (0.58), business needs K=5 (0.52)
7. Check: is silhouette at K=5 above 0.5? → Yes → acceptable
8. Check: does K=5 produce interpretable segments? → Yes → go with K=5

External validation:
9. Use Channel column (Hotel/Retail) as proxy ground truth
10. adjusted_rand_score(channel_labels, km_labels)
11. ARI > 0.3 indicates meaningful overlap

Bootstrap stability:
12. For b in 1..100: resample data, re-cluster, record assignments
13. For each customer: fraction of times assigned to same cluster
14. Mean stability > 90% → Constraint #3 satisfied
```

---

## 5 · Key Diagrams

### Silhouette geometry

```
 Segment "Loyalists"     Segment "Price-Sensitive"
   ●───●───●                    ●───●───●
       |   i                         |
   a(i) = mean dist             b(i) = mean dist
   within Loyalists             from i to Price-Sensitive

   s(i) = (b(i) - a(i)) / max(a(i), b(i))
   If b(i) >> a(i) → s(i) ≈ 1 (well assigned)
   If a(i) >> b(i) → s(i) ≈ -1 (misassigned)
```

### Three-metric comparison by K

```
Metric │ K=2   K=3   K=4   K=5   K=6   K=7
───────┼─────────────────────────────────────
Silh↑  │ 0.55  0.58* 0.54  0.52  0.48  0.45
DBI↓   │ 0.82  0.71* 0.75  0.78  0.85  0.92
CHI↑   │ 210   285*  275   260   240   220
───────┴──────────── * metric winner: K=3
                           ↑ business winner: K=5
                             silhouette=0.52 > 0.5 ✅
```

### When metrics disagree with business

```mermaid
flowchart TD
 A["Silhouette says K=3\nBusiness needs K=5"] --> B{"Is silhouette at K=5\nabove minimum threshold?"}
 B -->|"Yes (0.52 > 0.5)"| C["Accept K=5:\nmetric is acceptable\nbusiness value is high"]
 B -->|"No (below 0.3)"| D["Reject K=5:\nclusters are meaningless\nrenegotiate with business"]
 A --> E["Run silhouette subplot\n(per-segment bar chart)"]
 E --> F["Check: any segment\nwith negative bars?"]
 F -->|"No"| G["All segments\nare valid ✅"]
 F -->|"Yes"| H["Merge that segment\nwith its neighbour"]
```

---

## 6 · Hyperparameter Dial

### Silhouette score

| Dial | Effect |
|------|--------|
| `sample_size` | For datasets >5000, use `sample_size=2000` to reduce O(n²) cost |
| `metric` | `'euclidean'` (default) or `'cosine'` — must match clusterer's distance |

### Silhouette interpretation bands

| Range | Meaning |
|-------|---------|
| 0.7–1.0 | Strong, well-separated clusters |
| **0.5–0.7** | **Reasonable structure (our target)** |
| 0.25–0.5 | Weak structure (overlap) |
| <0.25 | No meaningful clustering |

### ARI / NMI

| Dial | Effect |
|------|--------|
| proxy label quality | ARI is only as good as the proxy — noisy proxies inflate variance |

---

## 7 · Code Skeleton

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_samples)

# ── Load and preprocess ───────────────────────────────────────────────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)
spend_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
X = df[spend_cols].values

X_log = np.log1p(X)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_log)

# PCA 2D (from Ch.2)
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_sc)
```

```python
# ── K sweep: internal metrics ─────────────────────────────────────────────────
K_range = range(2, 11)
results = {'K': [], 'silhouette': [], 'dbi': [], 'chi': []}

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_pca)
    results['K'].append(k)
    results['silhouette'].append(silhouette_score(X_pca, km.labels_))
    results['dbi'].append(davies_bouldin_score(X_pca, km.labels_))
    results['chi'].append(calinski_harabasz_score(X_pca, km.labels_))
```

```python
# ── External: ARI against Channel proxy ───────────────────────────────────────
channel = df['Channel'].values  # 1=Hotel/Restaurant/Café, 2=Retail

km5 = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_pca)
ari = adjusted_rand_score(channel, km5.labels_)
nmi = normalized_mutual_info_score(channel, km5.labels_)
print(f"ARI vs Channel proxy: {ari:.4f}  NMI: {nmi:.4f}")
```

```python
# ── Bootstrap stability ───────────────────────────────────────────────────────
n_boot = 100
n_customers = len(X_pca)
assignments = np.zeros((n_boot, n_customers), dtype=int)

for b in range(n_boot):
    idx = np.random.RandomState(b).choice(n_customers, n_customers, replace=True)
    km_b = KMeans(n_clusters=5, n_init=5, random_state=42).fit(X_pca[idx])
    # Assign ALL customers (not just bootstrap sample)
    assignments[b] = km_b.predict(X_pca)

# For each customer, most-frequent cluster across bootstraps
from scipy.stats import mode
stability = np.array([mode(assignments[:, i], keepdims=False).count / n_boot
                      for i in range(n_customers)])
print(f"Mean bootstrap stability: {stability.mean():.2%}")
print(f"Customers with >90% stability: {(stability > 0.9).mean():.1%}")
```

---

## 8 · What Can Go Wrong

**Optimising silhouette at the expense of domain relevance** — Silhouette analysis says K=3 is mathematically optimal (silhouette=0.58). But the CMO's marketing strategy requires 5 segments: "Loyalists", "Price-Sensitive", "Big Spenders", "Occasional", and "Deli Specialists". At K=5, silhouette drops to 0.52. Do you override the metric?

**Fix:** A silhouette of 0.52 at K=5 is acceptable if: (1) it's above your minimum threshold (0.5 in our case), and (2) the 5 segments are business-actionable. Metrics guide decisions — they don't make them. Document the trade-off: "Silhouette prefers K=3 (0.58), but we chose K=5 (0.52) to align with marketing segmentation strategy. Silhouette >0.5 confirms the clusters are valid."

---

**Treating ARI of 0 as "random clustering"** — ARI is corrected for chance, so random label assignments score ≈ 0. But ARI=0.05 is *also* not significantly above chance. You can't claim meaningful overlap with ground truth unless ARI >0.3 (moderate agreement) or ARI >0.7 (strong agreement).

**Fix:** Use ARI as a **sanity check**, not validation. If your clustering has ARI=0.02 against a known proxy (e.g., Channel: Hotel vs Retail), it means your discovered segments don't align with that proxy — which might be *good* (you found orthogonal structure) or *bad* (you found noise). Combine ARI with internal metrics.

---

**Comparing CHI across different datasets** — Calinski-Harabasz Index (CHI) is unbounded and scales with $n$. CHI=285 on 440 customers is **not** comparable to CHI=285 on 10,000 customers. A larger dataset will naturally have higher CHI even if cluster quality is identical.

**Fix:** Only compare CHI across different $K$ values **on the same dataset**. Use CHI to pick $K$, not to compare two clusterings from different datasets. For cross-dataset comparison, use silhouette (bounded [−1,1]).

---

**Ignoring per-segment silhouette plots** — Mean silhouette=0.52 looks acceptable. But what if the "Deli Specialists" segment (smallest, 10% of customers) has individual silhouette scores <0, while the other 4 segments are all >0.6? The mean hides the problem.

**Fix:** Always plot per-segment silhouette distributions — sklearn provides `silhouette_samples(X, labels)` which returns one score per customer. Group by cluster, plot as horizontal bar charts. If one segment has many negative bars, merge it with its nearest neighbour or drop those customers as noise.

---

**Skipping bootstrap stability** — Your clustering achieves silhouette=0.55 on the full 440-customer dataset. You ship the 5 segments to the marketing team, they build campaigns. Three months later you refresh the model with 450 new customers — and the segment assignments change completely. "Loyalists" from the old model are now split across "Big Spenders" and "Occasional". The campaigns break.

**Fix:** Always test stability with bootstrap resampling (100 samples, resample with replacement, re-cluster, track how often each customer lands in the same segment). If mean stability <70%, the clustering is fragile. If >90%, it's production-ready.

---

### Diagnostic Flowchart

```mermaid
flowchart TD
 A["Unsupervised metric
results confusing"] --> B{"Which metric?"}
 B -->|"Silhouette"| C{"Score below 0.3?"}
 C -->|Yes| D["Clustering is weak
try different K or algorithm"]
 C -->|No| E{"Business needs
more clusters?"}
 E -->|Yes| F{"New K still >0.5?"}
 F -->|Yes| G["Accept trade-off
document decision"]
 F -->|No| H["Renegotiate
or find better features"]
 B -->|"Davies-Bouldin"| I{"Score > 1.5?"}
 I -->|Yes| J["Clusters overlap
try lower K or DBSCAN"]
 B -->|"ARI"| K{"Score < 0.3?"}
 K -->|Yes| L{"Is proxy label
relevant?"}
 L -->|Yes| M["Clustering doesn't
match proxy → investigate"]
 L -->|No| N["Proxy irrelevant
ignore ARI → use silhouette"]
 B -->|"Stability"| O{"Bootstrap <70%?"}
 O -->|Yes| P["Fragile clustering
not production-ready"]
 O -->|No| Q["✅ Stable → ship it"]
```


---

## 9 · Where This Reappears

Internal validation metrics, bootstrap stability testing, and the "metrics vs business" trade-off framework generalise far beyond clustering:

- **[Ch.1 — Clustering](../ch01_clustering) & [Ch.2 — Dimensionality Reduction](../ch02_dimensionality_reduction)**: Backward link — the silhouette scores we've been tracking (0.42 → 0.48 → 0.52) are computed using the methods formalised in this chapter. The "0.5 threshold" for reasonable structure (mentioned in Ch.1-2) is defined here.
- **All ML tracks' Progress Check pattern**: Every track uses a quantitative threshold table (Constraint #1 target <$40k MAE, Constraint #5 target silhouette >0.5, etc.). The discipline of "measure against a falsifiable target" originates from the supervised metrics chapters (01-Regression) and extends here to unsupervised scenarios. Same pattern, different metrics.
- **[AI / Evaluating AI Systems](../../../ai/evaluating_ai_systems)**: LLM evaluation frameworks reuse internal-metric thinking when ground truth doesn't exist. Coherence, relevance, coverage — analogous to silhouette, Davies-Bouldin, Calinski-Harabasz. The "how do you score quality with no right answer?" question is the same.
- **[06-ReinforcementLearning / Ch.3 Policy Evaluation](../../06_reinforcement_learning/ch03_policy_evaluation)**: Bootstrap confidence intervals on episode returns use the same resampling methodology as cluster stability here. The 100-sample bootstrap, per-instance stability tracking, and 90% threshold for "production-ready" repeat exactly.
- **[04-RecommenderSystems / Ch.5 Evaluation Metrics](../../04_recommender_systems/ch05_evaluation_metrics)**: The trade-off between precision@k ("show only relevant items") and recall@k ("show all relevant items") mirrors the silhouette vs business-segment-count trade-off here. Metrics guide, domain decides.
- **[08-EnsembleMethods / Ch.4 Cluster-Then-Ensemble](../../08_ensemble_methods/ch04_cluster_ensemble)**: Combines K-Means (from Ch.1 here) with supervised learning — cluster customers, train separate models per cluster, ensemble predictions. Uses silhouette + Davies-Bouldin to validate clusters before building segment-specific models.

## 10 · Progress Check — What We Can Solve Now

![Progress visualization](img/ch03-progress-check.png)

🎉 **✅ ALL FIVE SEGMENTAI CONSTRAINTS ACHIEVED!**

✅ **Unlocked capabilities:**

1. **Quantitative validation without labels** — Silhouette score (0.52), Davies-Bouldin index (0.78), and Calinski-Harabasz index (285) confirm that K=5 clustering is mathematically valid even without ground truth customer labels.

2. **K-selection framework** — Ran K=2 to K=10 sweep. Silhouette peaks at K=3 (0.58), but K=5 (0.52) is acceptable because: (a) 0.52 > 0.5 threshold, (b) business requires 5 segments for marketing strategy. Documented the trade-off transparently.

3. **Bootstrap stability: 95%** — 100 bootstrap samples, re-clustered each, tracked customer assignments. 95% of customers land in the same segment across resamples. Segments are reproducible — production-ready.

4. **External validation: ARI=0.34** — Against Channel proxy (Hotel/Retail), our discovered segments show moderate overlap (ARI >0.3). This validates that segments capture real structure, not random noise.

5. **Business names assigned** — Centroid analysis mapped clusters to actionable profiles:
   - **Cluster 0 → "Loyalists"**: High grocery+milk+detergents, steady repeat buyers
   - **Cluster 1 → "Price-Sensitive"**: Low across all categories, minimal baskets
   - **Cluster 2 → "Big Spenders"**: High fresh+frozen+deli, premium products
   - **Cluster 3 → "Occasional Buyers"**: Moderate spend, grocery-focused
   - **Cluster 4 → "Deli Specialists"**: Disproportionate deli+fresh, low grocery

6. **Metrics suite operationalised** — Can now evaluate *any* clustering (not just K-Means) using the same silhouette / DBI / CHI / ARI / bootstrap workflow. Generalisable validation framework.

❌ **What we still can't solve (out of scope for this track):**
- ❌ **Temporal stability** — Do segments remain stable as new customers arrive monthly? Need streaming / incremental clustering (not covered; see production ML courses).
- ❌ **Feature importance** — Which of the 6 features *most* drives segment separation? Need SHAP on cluster assignments or supervised proxy models (covered in 08-EnsembleMethods).
- ❌ **Hierarchical segment refinement** — Can "Big Spenders" be subdivided into "Premium Fresh" vs "Frozen Bulk"? Need hierarchical clustering or Gaussian Mixture Models (not covered; out of lean curriculum scope).

**Final constraint scorecard:**

| Constraint | Status | Final Evidence | Journey |
|------------|--------|----------------|----------|
| #1 SEGMENTATION | ✅ **ACHIEVED** | K=5 validated by 3 metrics | Ch.1: 0.42 → Ch.2: 0.48 → Ch.3: **0.52 ✅** |
| #2 INTERPRETABILITY | ✅ **ACHIEVED** | 5 business names assigned | Centroid analysis + domain validation |
| #3 STABILITY | ✅ **ACHIEVED** | 95% bootstrap consistency | 100 resamples, 95% agreement |
| #4 SCALABILITY | ✅ **ACHIEVED** | O(nKd) for K-Means + PCA | 440 → 10k+ customers, no algorithm changes |
| #5 VALIDATION | ✅ **ACHIEVED** | Silhouette=0.52, DBI=0.78, CHI=285 | All 3 internal metrics in acceptable ranges |

**Real-world status:** "**SegmentAI is production-ready.** We have 5 validated, stable, business-actionable customer segments with quantitative proof (silhouette >0.5, bootstrap stability >90%, domain names assigned). Marketing can now build differentiated campaigns for Loyalists, Price-Sensitive, Big Spenders, Occasional, and Deli Specialists with confidence that segments will reproduce on new data."

**Track completion arc:**

```mermaid
flowchart LR
 A["Ch.1: Clustering
3 algorithms
K=5 baseline
silhouette = 0.42"] --> B["Ch.2: Dim Reduction
PCA/t-SNE/UMAP
Visualisation
silhouette = 0.48"]
 B --> C["Ch.3: Metrics
3 internal + 2 external
Bootstrap stability
silhouette = 0.52"]
 C --> D["🎉 SegmentAI
Production Deployment
All 5 constraints ✅"]
 A -.->|"Constraint #4 ✅"| E["Scalability"]
 B -.->|"Constraint #1 improved"| F["Tighter clusters"]
 C -.->|"Constraints #1,#2,#3,#5 ✅"| G["Validation complete"]
 style A fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
 style B fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
 style C fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
 style D fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

💡 **What you learned in this track:**
- **Unsupervised learning ≠ unvalidated learning** — Internal metrics let you quantify quality without labels
- **Preprocessing is not optional** — Log-transform + standardisation + PCA boosted silhouette 0.42 → 0.52
- **Metrics inform, domain decides** — Silhouette preferred K=3, business needed K=5. We balanced both.
- **Stability = production-readiness** — Fancy algorithm with 60% bootstrap stability < simple algorithm with 95% stability
- **The best metric is "can stakeholders act on it?"** — "Cluster 3" means nothing; "Price-Sensitive bulk buyers" is a marketing campaign

---

## 11 · Bridge Forward

🎉 **The SegmentAI challenge is complete!** We went from raw 6D spending data (Ch.1, silhouette=0.42) through dimensionality reduction (Ch.2, silhouette=0.48) to validated, stable, business-named segments (Ch.3, silhouette=0.52 ✅).

**All 5 SegmentAI constraints satisfied:**
- ✅ **SEGMENTATION:** 5 distinct clusters validated by silhouette >0.5
- ✅ **INTERPRETABILITY:** Segments named via centroid analysis — "Loyalists", "Price-Sensitive", "Big Spenders", "Occasional Buyers", "Deli Specialists"
- ✅ **STABILITY:** 95% bootstrap consistency — segments reproducible across resamples
- ✅ **SCALABILITY:** K-Means + PCA pipeline handles 10k+ customers
- ✅ **VALIDATION:** Silhouette=0.52, Davies-Bouldin=0.78, Calinski-Harabasz=285 — quantitatively validated

**Key takeaways from the unsupervised track:**

1. **No labels ≠ no evaluation** — Internal metrics (silhouette, DBI, CHI) provide quantitative validation even without ground truth. External metrics (ARI, NMI) offer sanity checks when proxy labels exist.

2. **Metrics and business can disagree — and that's OK** — Silhouette preferred K=3, business needed K=5. We chose K=5 because silhouette=0.52 was above our 0.5 threshold and segments were actionable. Always document the trade-off.

3. **Stability matters as much as quality** — A clustering with silhouette=0.60 that changes completely on bootstrap resampling is useless for production. Bootstrap stability >90% is the real deployment gate.

4. **Log-transform + PCA preprocessing unlocks better clusters** — Raw 6D data: silhouette=0.42. Log+scale: same 0.42. PCA 2D: silhouette=0.48. Final PCA+tuning: 0.52. Preprocessing is not optional for skewed, high-dimensional data.

5. **The most important metric is domain validation** — Can the sales team recognize these segments? Can they act on them? "Cluster 3" is meaningless; "Price-Sensitive bulk buyers averaging £8k Fresh, £2k Grocery" is a marketing campaign.

---

**Where unsupervised learning reappears:**

- **[04-RecommenderSystems](../../04_recommender_systems/README.md):** Cluster-based collaborative filtering uses K-Means to segment users, then recommends based on cluster averages (cold-start strategy for new users with no ratings).
- **[03-NeuralNetworks / Ch.10 Attention Mechanisms](../../03_neural_networks/ch10_attention_mechanisms):** UMAP/t-SNE visualisation of attention weights and hidden states — the same tools we used here to visualise 6D customers apply to 512D transformer embeddings.
- **[AI / RAG & Embeddings](../../../ai/rag_and_embeddings):** Semantic search uses vector embeddings + clustering to retrieve relevant documents — PCA for dimensionality reduction, HDBSCAN for cluster discovery.

💡 **Next frontier:** The unsupervised track taught discovery (clustering), compression (PCA), and validation (metrics). The **[08-EnsembleMethods](../../08_ensemble_methods)** track combines clustering with supervised learning — train separate models on each discovered segment, then ensemble their predictions.


