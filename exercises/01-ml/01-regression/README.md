# Exercise 01: SmartVal AI — Production Regression System

> **Grand Challenge:** Build a production-grade house valuation API that achieves <$40k MAE on California Housing dataset while meeting 5 production constraints.

**Scaffolding Level:** 🟢 Heavy (learn the workflow)

---

## Objective

Implement a complete ML regression pipeline with production patterns:
- <$40k MAE on held-out test set
- <100ms inference latency (p99)
- Explainable predictions (feature importance)
- Error handling and input validation
- Configuration-driven training
- Automated diagnostics

---

## What You'll Learn

- Train/validation/test splitting (no data leakage)
- Feature engineering (polynomial features, scaling)
- Regularization (Ridge, Lasso, ElasticNet)
- Hyperparameter tuning (GridSearchCV, Optuna)
- Model persistence (joblib)
- REST API design (Flask)
- Unit testing for ML

---

## Setup

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

---

## Project Structure

```
01_regression/
├── requirements.txt          # Dependencies
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── coding_guidelines.md      # Production patterns & hints
├── SOLUTION.md               # Reference implementation
├── src/
│   ├── data.py               # ✅ Scaffolded
│   ├── features.py           # ⚠️ Hints provided
│   ├── models.py             # ❌ TODO
│   ├── evaluate.py           # ✅ Scaffolded
│   └── api.py                # ❌ TODO
├── tests/
│   ├── test_data.py          # ✅ Complete
│   ├── test_features.py      # ⚠️ Partial
│   └── test_models.py        # ❌ TODO
└── notebooks/
    └── exploratory.ipynb     # Optional EDA
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] MAE <$40k on test set
- [ ] API returns predictions in <100ms
- [ ] Code passes linting: `black . && mypy src/`
- [ ] Residual plot shows no patterns
- [ ] Cross-validation confirms generalization

---

## Resources

**Concept Review:**
- [notes/01-ml/01_regression/](../../notes/01-ml/01_regression/) — Complete track
- [notes/01-ml/01_regression/grand-challenge.md](../../notes/01-ml/01_regression/grand-challenge.md) — Constraints

**Implementation Guides:**
- [coding_guidelines.md](coding_guidelines.md) — Hints & patterns
- [SOLUTION.md](SOLUTION.md) — Reference (read AFTER attempting!)

---

## Quick Start

```bash
# Install dependencies
./setup.sh

# Activate venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\Activate.ps1  # Windows

# Run tests (will fail initially)
pytest tests/

# Implement features.py, models.py, api.py
# ...

# Train model
python -m src.models

# Start API
python -m src.api
```