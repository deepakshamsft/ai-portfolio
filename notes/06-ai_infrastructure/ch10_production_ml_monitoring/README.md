# Ch.10 — Production ML Monitoring & A/B Testing

> **The story.** In **2019**, engineers at **Uber** discovered their fraud detection model had been silently degrading for **three months** — accuracy dropped from 92% to 76%, costing millions in missed fraud cases. The training data distribution (credit card transactions from 2018) no longer matched production (2019 transactions with new fraud patterns). The problem: **no monitoring**. That same year, **Evidently AI** launched the first open-source drift detection framework that automatically flagged when production inputs drifted from training distributions. Meanwhile, **Facebook** open-sourced their A/B testing framework **PlanOut**, showing how to scientifically measure whether a new model actually improved business metrics (not just training accuracy). Together they established the modern production ML workflow: **deploy → monitor → detect drift → A/B test → rollback or promote**. The insight: training metrics (accuracy on a fixed test set) are a poor proxy for production impact (business KPIs on evolving data).
>
> **Where you are in the curriculum.** You've just finished [Ch.9: ML Experiment Tracking & Model Registry](../ch09_ml_experiment_tracking) where you trained 100 models, picked the best one (94% test accuracy), and registered it for deployment. Now you've deployed it to production and it's serving **10,000 predictions per day**. But within **two weeks**, users start complaining: "The sentiment classifier is wrong more often now." Your test accuracy was 94% — what happened? This chapter teaches the discipline that separates research prototypes from reliable production systems: **continuous monitoring, drift detection, and safe deployment practices**. You'll deploy a BERT sentiment classifier, detect data drift using Evidently AI, A/B test a new model version, and implement automated rollback when performance degrades.
>
> **Notation in this chapter.** `data drift` — change in input feature distribution (P(X) in training vs. production); `prediction drift` — change in model output distribution (P(ŷ) shifts even if X is similar); `concept drift` — change in true relationship between X and y (what was "positive sentiment" in 2020 ≠ 2024); `A/B test` — controlled experiment serving model v1 to 50% of traffic, v2 to 50%, measuring which performs better on business metrics; `rollback` — revert production deployment from v2 → v1 when performance degrades; `canary deployment` — gradual rollout (10% → 50% → 100%) to limit blast radius of bad models.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: You're the Platform Engineer at **InferenceBase** (the AI startup from Ch.1-9). You've just deployed the best BERT sentiment classifier from Ch.9 (94% test accuracy) to production. It processes **10,000 customer reviews per day** for an e-commerce platform. The CEO is happy — until **week 3**, when the customer support team reports:
> - "Users are complaining the product recommendations are wrong (driven by sentiment predictions)"
> - "Manual spot-checks show the model is getting ~70% accuracy now (down from 94%)"
> - **You have no idea when the degradation started or why**

**What's blocking us:**
- **Silent degradation** — No alerts, no dashboards, no idea performance dropped until users complained
- **Unknown root cause** — Is it data drift (input distribution changed)? Concept drift (definition of "positive" changed)? Model bug?
- **No safe rollback** — The previous model version (v1) was deleted after deploying v2
- **Blind deployment** — When you train a new model (v3), you have no way to verify it's better **in production** before rolling out to 100% of traffic

**What this chapter unlocks:**
The **production ML monitoring & A/B testing** discipline:
1. **Monitor continuously** — Track data drift, prediction drift, and performance metrics in real-time
2. **Detect degradation early** — Alerts fire within **24 hours** (not 3 weeks)
3. **A/B test new models** — Deploy v2 to 10% of traffic, compare metrics to v1, promote only if better
4. **Rollback in <5 minutes** — Automated cutover to previous version when metrics degrade
5. **Root cause analysis** — Drill into drift reports to understand *why* performance dropped (data shift, adversarial inputs, concept drift)

✅ **After this chapter**: When model accuracy drops from 94% to 70%, you'll know within 24 hours, have a drift report explaining why (e.g., "production text is 30% shorter than training data"), and roll back to v1 in 2 minutes.

---

## Animation

![Chapter animation](img/ch10-monitoring-needle.gif)

*Detection time: 2 weeks → 2 hours with monitoring*

---

## 1 · The Core Idea — Continuous Monitoring = Data Drift + Prediction Drift + Performance Metrics

Production monitoring solves one problem: **detect when your deployed model stops working**. The solution requires three layers of monitoring:

### Production Monitoring = Data Drift + Prediction Drift + Performance Metrics

| Layer | What It Detects | How to Measure | Tool | Alert Threshold |
|---|---|---|---|---|
| **Data Drift** | Input distribution changed (P(X<sub>prod</sub>) ≠ P(X<sub>train</sub>)) | KL divergence, KS test, PSI | Evidently AI | KL divergence > 0.1 |
| **Prediction Drift** | Output distribution changed (P(ŷ<sub>prod</sub>) ≠ P(ŷ<sub>train</sub>)) | Class imbalance, entropy | Evidently AI | Positive class % > 60% |
| **Performance Drift** | Accuracy dropped (y<sub>true</sub> vs. ŷ<sub>pred</sub>) | Accuracy, F1, precision, recall | Custom logging | Accuracy < 90% |

**Why all three layers?**
- **Data drift alone** — Input changed but model might still work (e.g., text is 10% longer but sentiment is still easy to classify)
- **Prediction drift alone** — Output changed but might be correct (e.g., more negative reviews in production than training — that's reality, not a model bug)
- **Performance drift** — Ground truth (requires human labels or delayed feedback like user clicks)

**The monitoring workflow:**

```
┌────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION MONITORING STACK                      │
│                                                                      │
│  ALERT LAYER                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Automated Alerts (Email, Slack, PagerDuty)                   │  │
│  │  • Data drift detected → Retrain with new data                │  │
│  │  • Accuracy < 90% → Rollback to v1                            │  │
│  │  • Latency > 200ms → Scale up inference servers               │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                       │
│  DETECTION LAYER                                                     │
│  ┌────────────────────────────▼──────────────────────────────────┐ │
│  │  Evidently AI Drift Reports                                    │ │
│  │  • Compare P(X_prod) vs. P(X_train) (KL divergence)           │ │
│  │  • Compare P(ŷ_prod) vs. P(ŷ_train) (class distribution)     │ │
│  │  • Visualize feature histograms (training vs. production)     │ │
│  └───────────────────────────┬────────────────────────────────────┘ │
│                              │                                       │
│  COLLECTION LAYER                                                    │
│  ┌────────────────────────────▼──────────────────────────────────┐ │
│  │  Prediction Logging (SQLite / Postgres / BigQuery)            │ │
│  │  • Log every prediction: (input, output, timestamp, model_v)  │ │
│  │  • Log ground truth when available (user feedback, labels)    │ │
│  │  • Log metadata: latency, error codes, model_id               │ │
│  └───────────────────────────┬────────────────────────────────────┘ │
│                              │                                       │
│  SERVING LAYER                                                       │
│  ┌────────────────────────────▼──────────────────────────────────┐ │
│  │  Model Serving (MLflow / FastAPI / TorchServe)                │ │
│  │  • v1 (90% traffic) ────┐                                      │ │
│  │  • v2 (10% traffic) ────┴─→ Predictions                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Training metrics (accuracy on a fixed test set from 3 months ago) are a **lagging indicator** of production performance. Drift detection (comparing distributions) is a **leading indicator** — it warns you *before* accuracy drops.

---

## 2 · Running Example — Sentiment Classifier in Production

You're deploying **BERT-base-uncased** fine-tuned for sentiment classification on movie reviews (IMDB dataset from Ch.9). The model achieved **94% test accuracy** in training. Now it's serving production traffic from an e-commerce platform classifying **product reviews** (not movie reviews).

**Initial setup:**
- **Model v1** — Trained on IMDB movie reviews (25k samples, avg length 200 words)
- **Production data** — Product reviews (avg length 50 words, more concise, different vocabulary)
- **Traffic** — 10,000 predictions/day
- **SLA** — 95th percentile latency < 200ms, accuracy > 90%

**What we'll implement:** Monitor data drift, detect degradation, deploy v2 via A/B test, rollback if v2 is worse.

### Step 1: Deploy Model v1 (Baseline)

Deploy the registered model from Ch.9 using MLflow Model Serving:

```bash
# Start MLflow model serving on port 5001
mlflow models serve \
    --model-uri "models:/bert-sentiment-classifier/Production" \
    --port 5001 \
    --no-conda
```

**Inference API:**
```python
import requests

response = requests.post(
    "http://localhost:5001/invocations",
    json={"inputs": ["This product is amazing!", "Terrible quality."]}
)

predictions = response.json()["predictions"]  # [1, 0] (positive, negative)
```

**What we log per prediction:**
```python
# Log to SQLite database
log_prediction({
    "timestamp": datetime.now(),
    "model_version": "v1",
    "input_text": "This product is amazing!",
    "prediction": 1,  # positive
    "latency_ms": 45,
    "confidence": 0.95
})
```

### Step 2: Monitor Data Drift (Input Distribution Changes)

> 💡 **Intuition first:** **Data drift measures how much the production input distribution has shifted away from training data**. If your model was trained on movie reviews (200 words, balanced sentiment) but production serves product reviews (50 words, 75% positive), the input distributions P(X<sub>train</sub>) and P(X<sub>prod</sub>) have diverged. Drift metrics like **KL divergence** and **PSI** (Population Stability Index) quantify this shift — high values mean "your model is seeing data it wasn't trained on, so performance may degrade." Think of drift detection as an **early warning system**: it alerts you *before* accuracy drops, giving you time to retrain.

**After 1 week of production traffic**, compare production inputs vs. training data:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load training data (IMDB reviews)
training_data = pd.read_csv("data/imdb_train.csv")

# Load production data (logged predictions from last 7 days)
production_data = pd.read_sql("SELECT input_text FROM predictions WHERE timestamp > NOW() - INTERVAL '7 days'", conn)

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=training_data, current_data=production_data)
report.save_html("drift_report.html")
```

**Drift report findings:**

| Feature | Training (IMDB) | Production (E-commerce) | Drift Detected? | KL Divergence |
|---|---|---|---|---|
| Text length | 200 ± 50 words | 50 ± 20 words | ✅ YES | 0.35 |
| Vocabulary overlap | — | 60% overlap | ✅ YES | — |
| Positive class % | 50% | 75% | ✅ YES | 0.12 |

**Interpretation:**
- **Text length drift** — Production reviews are **4× shorter** (50 vs. 200 words)
- **Vocabulary drift** — Only 60% of production words appeared in training (new product-specific terms: "fast shipping", "great value")
- **Class imbalance drift** — Production has 75% positive reviews (vs. 50% in training) — could mean genuine happiness OR the model is over-predicting positive

**Alert triggered:** "Data drift detected (KL divergence 0.35 > threshold 0.1) — Consider retraining with production data"

### Step 3: Monitor Prediction Drift (Output Distribution Changes)

Even if inputs look similar, **output distribution** can shift:

```python
from evidently.metric_preset import ClassificationPreset

# Compare prediction distributions
report = Report(metrics=[ClassificationPreset()])
report.run(
    reference_data=training_predictions,  # Predictions on test set
    current_data=production_predictions   # Predictions in production
)
report.save_html("prediction_drift_report.html")
```

**Prediction drift findings:**

| Metric | Training Test Set | Production (Week 1) | Production (Week 3) | Drift? |
|---|---|---|---|---|
| Positive class % | 50% | 75% | 85% | ✅ YES |
| Prediction entropy | 0.68 | 0.72 | 0.45 | ✅ YES (more confident but wrong) |
| Avg confidence | 0.82 | 0.85 | 0.92 | ⚠️ Warning (overconfident) |

**Interpretation:**
- **Positive class % increased** — Model predicts "positive" 85% of the time (vs. 50% in training)
- **Entropy decreased** — Model is more confident in predictions (lower entropy = less uncertainty)
- **Overconfidence** — High confidence (0.92) but users report incorrect predictions → model is **confidently wrong**

**Root cause hypothesis:** Model trained on movie reviews (balanced 50/50 positive/negative) now sees product reviews (naturally more positive), but the **definition of "positive"** differs (movie: "great acting" vs. product: "fast shipping").

### Step 4: Deploy Model v2 (A/B Test: 10% Traffic)

You've retrained a new model (v2) on **1,000 labeled production reviews**. Before rolling out to 100% of traffic, run an **A/B test**:

```python
# A/B test controller (traffic splitter)
def route_traffic(user_id):
    """Route 10% of users to v2, 90% to v1."""
    if hash(user_id) % 10 == 0:
        return "v2"  # 10% traffic
    else:
        return "v1"  # 90% traffic

# Inference with A/B routing
def predict(text, user_id):
    model_version = route_traffic(user_id)
    
    if model_version == "v1":
        prediction = model_v1.predict(text)
    else:
        prediction = model_v2.predict(text)
    
    # Log with model version tag
    log_prediction({
        "model_version": model_version,
        "input": text,
        "prediction": prediction,
        "user_id": user_id
    })
    
    return prediction
```

**A/B test runs for 48 hours**, collecting metrics for both versions.

### Step 5: Compare Business Metrics (v1 vs. v2)

**Performance comparison after 48 hours (10,000 predictions each):**

| Metric | v1 (Baseline) | v2 (Retrained) | Winner? |
|---|---|---|---|
| **Accuracy** (on labeled subset) | 72% | 91% | ✅ v2 |
| **F1 Score** | 0.68 | 0.89 | ✅ v2 |
| **Latency (p95)** | 180ms | 220ms | ⚠️ v1 (but acceptable) |
| **User thumbs-up rate** | 65% | 88% | ✅ v2 |
| **False positive rate** | 35% | 12% | ✅ v2 |

**Statistical significance test:**
```python
from scipy import stats

# Accuracy samples (bootstrap from logged predictions)
v1_accuracy = [0.71, 0.73, 0.72, 0.70, 0.74]  # 5 daily samples
v2_accuracy = [0.90, 0.91, 0.92, 0.89, 0.91]

# T-test: Is v2 significantly better?
t_stat, p_value = stats.ttest_ind(v1_accuracy, v2_accuracy)

if p_value < 0.05:
    print(f"v2 is significantly better (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

**Decision:** v2 is significantly better (p < 0.001) → **Proceed with gradual rollout**

### Step 6: Gradual Rollout (10% → 50% → 100%) or Rollback

**Rollout plan:**
```
Day 1: Deploy v2 to 10% of traffic (A/B test)
Day 3: If metrics hold, increase to 50%
Day 5: If metrics hold, increase to 100%
```

**Automated rollout script:**
```python
def gradual_rollout(target_percentage, current_percentage, step=10):
    """Gradually increase v2 traffic from current to target."""
    for pct in range(current_percentage, target_percentage + 1, step):
        # Update traffic split
        update_traffic_split(v1_pct=100-pct, v2_pct=pct)
        
        # Wait 24 hours and monitor metrics
        time.sleep(24 * 3600)
        
        # Check metrics
        v2_accuracy = get_accuracy(model="v2", last_hours=24)
        
        if v2_accuracy < 0.90:  # Threshold
            print(f"⚠️ v2 accuracy dropped to {v2_accuracy:.2%} — ROLLING BACK")
            rollback_to_v1()
            return False
        else:
            print(f"✅ v2 traffic at {pct}%, accuracy {v2_accuracy:.2%}")
    
    print("🎉 v2 fully deployed to 100% of traffic")
    return True

# Execute rollout
gradual_rollout(current_percentage=10, target_percentage=100, step=10)
```

**Rollback procedure (if metrics degrade):**
```python
def rollback_to_v1():
    """Instant cutover: v2 → v1."""
    update_traffic_split(v1_pct=100, v2_pct=0)
    
    # Log rollback event
    log_event({
        "timestamp": datetime.now(),
        "event": "rollback",
        "from_version": "v2",
        "to_version": "v1",
        "reason": "accuracy < 90%"
    })
    
    # Send alert
    send_alert("🚨 Model v2 rolled back due to performance degradation")
```

**Result:** v2 reaches 100% traffic by Day 5 with 91% accuracy maintained. v1 is archived but kept in registry for emergency rollback.

---

## 3 · Mental Model — Train-Time Metrics ≠ Production Metrics

```
TRAINING ENVIRONMENT                  PRODUCTION ENVIRONMENT
┌─────────────────────────┐          ┌──────────────────────────────┐
│ Fixed test set           │          │ Evolving data distribution   │
│ (IMDB reviews from 2020) │          │ (E-commerce reviews from     │
│                          │          │  2024, changing daily)       │
│ Accuracy: 94% ✅         │    ≠     │ Accuracy: 72% ❌             │
│                          │          │                              │
│ • Balanced classes       │          │ • Class imbalance (75% pos)  │
│ • Clean, labeled data    │          │ • Noisy, unlabeled data      │
│ • Controlled environment │          │ • Adversarial inputs         │
│ • No time pressure       │          │ • Real-time SLA (<200ms)     │
└─────────────────────────┘          └──────────────────────────────┘
```

### Why Models Degrade in Production

| Degradation Type | What Changed | Example | How to Detect |
|---|---|---|---|
| **Data Drift** | P(X) changed (input distribution) | Product reviews are shorter than movie reviews | KL divergence, KS test |
| **Concept Drift** | P(y\|X) changed (true relationship) | "Positive" in 2020 ≠ "Positive" in 2024 (sentiment evolves) | Performance drift (accuracy over time) |
| **Prediction Drift** | P(ŷ) changed (output distribution) | Model predicts 85% positive (vs. 50% in training) | Class distribution shift |
| **Adversarial Drift** | Malicious inputs | Users game the system ("great great great great...") | Outlier detection, anomaly scores |
| **Label Drift** | Ground truth definition changed | Product team changes what counts as "positive" | Manual review, annotation drift |

**Key insight:** **Concept drift is invisible to data drift detection** — inputs look normal, but the true labels (y) have changed meaning. You need **performance monitoring** (accuracy over time) to catch concept drift.

**The monitoring hierarchy:**
1. **Data drift** (leading indicator) — Detects input changes *before* performance drops
2. **Prediction drift** (intermediate indicator) — Detects output changes (could be benign or harmful)
3. **Performance drift** (lagging indicator) — Detects accuracy drops (requires ground truth labels)

**Best practice:** Monitor all three layers. Data drift + prediction drift = early warning system. Performance drift = confirmation.

---

## 4 · What Can Go Wrong — Monitoring Pitfalls and Fixes

| Pitfall | Symptom | Fix |
|---|---|---|
| **Silent degradation (no alerts)** | "Model accuracy dropped from 94% to 70% but we didn't notice for 3 weeks." | Set up automated alerts (email, Slack, PagerDuty) triggered by drift or performance metrics. |
| **False alarms (noisy metrics)** | "We get 50 drift alerts per day, but only 1 is real." | Tune alert thresholds (e.g., KL divergence > 0.1 instead of > 0.01). Use statistical tests (p-value < 0.05). |
| **Rollback too slow (manual process)** | "It took 2 hours to roll back because we had to find the old model, rebuild the container, and redeploy." | Automate rollback (one CLI command or feature flag flip). Keep previous version deployed but idle (blue-green deployment). |
| **A/B test bias (traffic not random)** | "v2 looks better but it only saw easy examples (short reviews)." | Use **stratified sampling** (hash user_id, not request_id) — same user always gets same version. Verify traffic split is truly 50/50. |
| **No ground truth labels** | "We're logging predictions but we don't know if they're correct." | Collect ground truth via user feedback (thumbs up/down), manual labeling (sample 100/day), or delayed labels (e.g., clicks after 24 hours). |
| **Data drift but no concept drift** | "Inputs changed but model still works fine — false alarm." | Don't panic on data drift alone. Check **prediction drift + performance drift**. Data drift is a *warning*, not a *failure*. |
| **Overconfident predictions** | "Model predicts positive with 0.99 confidence but it's wrong 40% of the time." | Calibrate model (Platt scaling, temperature scaling). Monitor **calibration error** (ECE, MCE). Alert when confidence >> accuracy. |
| **Forgot to log model version** | "Users report wrong predictions but we don't know if it's v1 or v2 serving them." | Always log `model_version` with every prediction. Include in error logs, dashboards, and drift reports. |

---

## 5 · Progress Check — Given Drift Report, Decide: Rollback, Retrain, or Do Nothing?

**Scenario:** You've just reviewed the Evidently drift report after 7 days of production traffic. The table shows:

| Metric | Training Data | Production (Week 1) | Drift? | Severity |
|---|---|---|---|---|
| Text length (avg words) | 200 | 50 | ✅ YES | High (KL=0.35) |
| Positive class % | 50% | 75% | ✅ YES | Medium (KL=0.12) |
| Vocabulary overlap | — | 60% | ✅ YES | High |
| Accuracy (labeled subset) | 94% | 91% | ❌ NO | Low (within SLA) |
| Latency (p95) | 150ms | 170ms | ❌ NO | Low (within SLA) |

**Questions:**

1. **Should you roll back to the previous model version?**
   - **Answer:** No — accuracy is 91% (within SLA of 90%), and latency is acceptable
   - **Reasoning:** Data drift is high BUT performance is still good → drift is benign (for now)

2. **Should you retrain the model with production data?**
   - **Answer:** Yes (proactive measure) — high data drift (text length, vocabulary) means training distribution no longer matches production
   - **Reasoning:** Accuracy is 91% *now*, but drift suggests it will degrade further → retrain with 1,000 labeled production samples to align with new distribution

3. **Should you do nothing and keep monitoring?**
   - **Answer:** Not recommended — high data drift + vocabulary shift = red flag
   - **Reasoning:** Even if accuracy is OK today, the model is fragile (trained on wrong distribution) → small changes could cause sudden drop

4. **How would you collect labels for retraining?**
   - **Option 1:** User feedback (thumbs up/down on predictions) — cheap, noisy, biased (users only label bad predictions)
   - **Option 2:** Manual labeling (hire annotators to label 1,000 production reviews) — expensive but accurate
   - **Option 3:** Active learning (label only high-uncertainty predictions) — efficient, requires 200–300 labels
   - **Recommended:** Start with Option 1 (user feedback), supplement with Option 3 (active learning on uncertain cases)

5. **What's the rollback trigger?**
   - **Accuracy < 90%** (SLA violation) → immediate rollback
   - **OR data drift + accuracy drop > 3%** (91% → 88%) in 24 hours → rollback + investigate
   - **OR user complaints spike > 2× baseline** → rollback + manual review

**Takeaway:** **Data drift alone is not a rollback trigger** — it's a signal to prepare for retraining. **Performance drift (accuracy drop) is the rollback trigger**.

---

## 6 · Bridge to Future — Scaling to Multi-Model Systems

You've just built a robust monitoring + A/B testing pipeline for a single model. But production ML systems rarely deploy just one model. **What's next?**

### What Future Chapters Will Address

**Question 1: "We have 10 models in production (recommendation, sentiment, fraud, ranking). How do we monitor all of them?"**
→ **Multi-model monitoring dashboards** (unified metrics across models, anomaly detection, model health scores)

**Question 2: "Our model is part of a pipeline: image → object detection → classification → ranking. How do we debug when the pipeline breaks?"**
→ **Pipeline monitoring** (trace requests across models, identify bottleneck stages, track error propagation)

**Question 3: "We want to serve an ensemble (5 models vote). How do we A/B test ensembles vs. single models?"**
→ **Ensemble serving** (majority vote, weighted voting, stacking) + A/B testing ensemble strategies

**Question 4: "We have a cascade: fast model (BERT-tiny) filters, slow model (BERT-large) refines. How do we monitor both?"**
→ **Cascading models** (latency vs. accuracy tradeoff, when to invoke expensive model)

**Question 5: "Our model retrains daily (online learning). How do we ensure each new version is better than the last?"**
→ **Continuous retraining pipelines** (automated A/B tests for every new model version, champion/challenger framework)

### The Operational Maturity Ladder

```
Ch.10 (Current)                   Future (Advanced Topics)
┌─────────────────────────┐      ┌────────────────────────────────┐
│ Single model            │      │ Multi-model systems            │
│ Manual drift review     │  →   │ Automated drift detection      │
│ Manual A/B test         │      │ Automated A/B tests per deploy │
│ Rollback on failure     │      │ Canary + gradual rollout       │
│ Weekly retraining       │      │ Online learning (daily)        │
└─────────────────────────┘      └────────────────────────────────┘
```

**What you bring from Ch.10:**
- ✅ Drift detection (Evidently AI)
- ✅ A/B testing framework (traffic splitting, metrics comparison)
- ✅ Rollback procedures (automated cutover in <5 min)
- ✅ Production logging (predictions + ground truth)

**What you'll need for multi-model systems:**
- Feature stores (shared features across models) → revisit Ch.8
- Model registry (versioning 10+ models) → revisit Ch.9
- Distributed tracing (track requests across model pipeline)
- Cost monitoring (inference cost per model, per user)

---

## Navigation

**AI Infrastructure Track:**
- [← Ch.9: ML Experiment Tracking & Model Registry](../ch09_ml_experiment_tracking)
- [→ Future chapters TBD] *(multi-model systems, pipeline monitoring, online learning)*
- [↑ Track README](../README.md)

**Related Content:**
- [DevOps Fundamentals: Ch.5 — Monitoring & Observability](../../devops_fundamentals) — Prometheus, Grafana, alerting (prerequisites)
- [DevOps Fundamentals: Ch.7 — Networking & Reverse Proxies](../../devops_fundamentals) — Load balancing for A/B tests
- [ML Track: Model Evaluation](../../ml/01_regression/ch06_evaluation) — Metrics, confusion matrix, precision/recall

**Tools Documentation:**
- [Evidently AI Official Docs](https://docs.evidentlyai.com/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
- [A/B Testing Best Practices](https://exp-platform.com/) — Microsoft's Experimentation Platform paper

---

**Last updated:** April 26, 2026  
**Status:** ✅ README complete — notebook + assets in progress
