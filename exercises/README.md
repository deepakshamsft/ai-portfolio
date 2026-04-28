# Production ML Exercises

> **Purpose:** Transform theoretical ML knowledge from `notes/` into production-ready portfolio projects.

Each exercise corresponds to a grand challenge from `notes/01-ml/` and requires implementing production patterns: clean code, testing, error handling, configuration management, and deployment.

---

## Philosophy

**`notes/` teaches concepts. `exercises/` builds production skills.**

- **Notebooks** (in `notes/`) = Learn *what* regularization does mathematically
- **Exercises** (here) = Implement Ridge/Lasso in production-ready, testable, deployable code

---

## How to Use

### 1. Complete Chapters First
Before attempting an exercise, finish the corresponding chapters in `notes/01-ml/`:
- **Example:** Read `notes/01-ml/01_regression/ch01-ch07/` before starting `exercises/01-ml/01_regression/`

### 2. Attempt the Exercise
Each exercise includes:
- ✅ **Scaffolding:** Partial code structure (varies by track)
- ⚠️ **Hints:** Coding guidelines with patterns to implement
- ❌ **TODOs:** Code you must write from scratch
- ✅ **Tests:** Unit tests to validate your solution

### 3. Validate Your Solution
```bash
cd exercises/01-ml/01_regression/
./setup.sh              # Create venv + install dependencies
make test               # Run unit tests
make lint               # Check code quality
make train              # Train model
```

### 4. Compare with Reference
After attempting the exercise, review `SOLUTION.md` to compare your approach.

---

## Scaffolding Levels

### 🟢 Heavy Scaffolding (Track 01 — Regression)
**What's provided:**
- ✅ Data loading fully implemented
- ✅ Complete test suite (learn TDD workflow)
- ✅ Evaluation framework implemented
- ⚠️ Feature engineering: function signatures + hints
- ❌ Model training: TODO comments only
- ❌ API: TODO comments only

**Goal:** Learn the production ML workflow without getting stuck on boilerplate.

---

### 🟡 Medium Scaffolding (Tracks 02-03)
**What's provided:**
- ✅ Data loading signatures provided
- ⚠️ Test suite: half complete, half TODO
- ⚠️ Feature engineering: hints only
- ❌ Model training: blank
- ❌ API: blank

**Goal:** Apply patterns learned in Track 01 to new problems.

---

### 🔴 Minimal Scaffolding (Tracks 04-08)
**What's provided:**
- ✅ Project structure (folders, config template)
- ⚠️ README with requirements
- ❌ Everything else: blank files with TODO

**Goal:** Demonstrate independence — design and implement from scratch.

---

## Success Criteria (All Exercises)

Your exercise is complete when:
- [ ] All tests pass (`make test`)
- [ ] Grand challenge constraint met (MAE, accuracy, HR@10, etc.)
- [ ] Code passes linting (`make lint`)
- [ ] README documents your approach
- [ ] Can explain every design decision in 1:1 interview

---

## Available Exercises

### Core Fundamentals

#### 📊 [01. Regression — SmartVal AI](01-ml/01_regression/README.md)
**Grand Challenge:** <$40k MAE on California Housing  
**Concepts:** Ridge, Lasso, polynomial features, cross-validation, hyperparameter tuning  
**Production:** Flask API, joblib persistence, config-driven training  
**Scaffolding:** 🟢 Heavy

#### 🎭 [02. Classification — FaceAI](01-ml/02_classification/README.md)
**Grand Challenge:** >90% avg accuracy across 40 facial attributes  
**Concepts:** Logistic regression, SVM, multi-label classification, ROC/PR curves  
**Production:** Batch inference, threshold optimization, class imbalance handling  
**Scaffolding:** 🟡 Medium

#### 🧠 [03. Neural Networks — UnifiedAI](01-ml/03_neural_networks/README.md)
**Grand Challenge:** $28k MAE + 95% accuracy with same architecture  
**Concepts:** Backpropagation, CNNs, RNNs, Transformers, multi-task learning  
**Production:** TensorBoard monitoring, model checkpointing, ONNX export  
**Scaffolding:** 🟡 Medium

---

### Specialized Paradigms

#### 🎬 [04. Recommender Systems — FlixAI](01-ml/04_recommender_systems/README.md)
**Grand Challenge:** >85% HR@10 on MovieLens  
**Concepts:** Collaborative filtering, matrix factorization, cold start handling  
**Production:** Two-stage serving, latency budgeting, A/B testing  
**Scaffolding:** 🔴 Minimal

#### 🚨 [05. Anomaly Detection — FraudShield](01-ml/05_anomaly_detection/README.md)
**Grand Challenge:** 80% recall @ 0.5% FPR on credit card fraud  
**Concepts:** Isolation Forest, Autoencoders, ensemble fusion, class imbalance  
**Production:** Feature store, ONNX runtime, PSI drift monitoring, SHAP  
**Scaffolding:** 🔴 Minimal

#### 🤖 [06. Reinforcement Learning — AgentAI](01-ml/06_reinforcement_learning/README.md)
**Grand Challenge:** Solve GridWorld + CartPole ≥195 steps  
**Concepts:** Q-learning, DQN, policy gradients, experience replay  
**Production:** Episode checkpointing, TensorBoard, environment wrappers  
**Scaffolding:** 🔴 Minimal

#### 📦 [07. Unsupervised Learning — SegmentAI](01-ml/07_unsupervised_learning/README.md)
**Grand Challenge:** 5 actionable segments, silhouette >0.5  
**Concepts:** K-means, DBSCAN, PCA, t-SNE, UMAP, cluster validation  
**Production:** Real-time assignment, cluster drift monitoring  
**Scaffolding:** 🔴 Minimal

---

### Capstone

#### 🏆 [08. Ensemble Methods — EnsembleAI](01-ml/08_ensemble_methods/README.md)
**Grand Challenge:** Beat all single models by >5%  
**Concepts:** Random Forest, XGBoost, LightGBM, stacking, SHAP  
**Production:** Parallel inference, PSI monitoring, shadow mode, retraining pipeline  
**Scaffolding:** 🔴 Minimal

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Terminal (Unix) or PowerShell (Windows)

### Quick Start (Any Exercise)

**Unix/macOS/WSL:**
```bash
cd exercises/01-ml/01_regression/
chmod +x setup.sh
./setup.sh
source venv/bin/activate
make test
```

**Windows PowerShell:**
```powershell
cd exercises\01-ml\01_regression\
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
make test
```

### Common Commands (All Exercises)

```bash
make train      # Train models, generate diagnostics
make test       # Run unit tests
make lint       # Check code quality (black, mypy)
make serve      # Start Flask API (if implemented)
make clean      # Remove generated files
make docker     # Build Docker image (if implemented)
```

---

## Portfolio Showcase

After completing an exercise, add to your portfolio:

### GitHub README Template
```markdown
## [Exercise Name] — Production ML System

**Grand Challenge:** [Constraint]  
**Result:** [Your metric] ✅  
**Tech Stack:** scikit-learn, XGBoost, Flask, Docker

**Key Features:**
- 🎯 [Feature 1]
- 📊 [Feature 2]
- 🚀 [Feature 3]

**Highlights:**
- [Specific production pattern you implemented]
- [Performance optimization you achieved]
- [Design decision you're proud of]

[Link to code] | [Demo GIF] | [API docs]
```

### LinkedIn Post Template
```
Just completed the [Exercise Name] production ML exercise! 🎉

Built a [brief description] achieving [metric]. Implemented [production pattern] for [business impact].

Key takeaway: [One-sentence insight]

Tech: [Stack]
GitHub: [Link]
#MachineLearning #MLOps #Portfolio
```

---

## FAQ

### Q: Do I need to complete exercises in order?
**A:** No, but we recommend starting with Track 01 (Regression) to learn the workflow, then choose based on interest.

### Q: Can I use different libraries (e.g., PyTorch instead of TensorFlow)?
**A:** Yes! The exercises specify constraints (MAE, accuracy), not implementation. If you can meet the constraint with PyTorch, go for it.

### Q: What if I get stuck?
**A:** 
1. Re-read the corresponding chapter in `notes/`
2. Check `coding_guidelines.md` for hints
3. Run the tests — they guide you to the solution
4. After genuine attempt, review `SOLUTION.md`

### Q: How long does each exercise take?
**A:**
- **Track 01 (Heavy scaffolding):** 8-12 hours
- **Tracks 02-03 (Medium):** 12-20 hours
- **Tracks 04-08 (Minimal):** 20-40 hours

### Q: Can I skip the tests?
**A:** No. Testing is a core production skill. The exercises teach TDD workflow — write test, watch it fail, implement code, watch it pass.

### Q: Are the SOLUTION.md files identical to what I should write?
**A:** No. They show *one* correct approach. Your solution may differ (and be equally valid) if it meets the success criteria.

---

## Contributing

Found a bug? Have a better solution? PRs welcome!

**Contribution guidelines:**
1. All exercises must pass `make test`
2. New scaffolding should include tests
3. SOLUTION.md must be tested and validated
4. Document any new production patterns

---

## See Also

- [notes/01-ml/README.md](../notes/01-ml/README.md) — Theoretical foundations
- [notes/01-ml/ml-engg-readiness.md](../notes/01-ml/ml-engg-readiness.md) — Gap analysis
- [TEMPLATE/](TEMPLATE/README.md) — Reusable project skeleton

---

**Status:** Phase 1 complete (infrastructure), Phase 2 in progress (Track 01)  
**Last Updated:** April 28, 2026
