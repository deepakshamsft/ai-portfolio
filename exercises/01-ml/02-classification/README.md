# Exercise 02: FaceAI — Production Classification System

> **Grand Challenge:** Build a production-grade face classification API that achieves >90% accuracy on Olivetti Faces dataset while meeting 5 production constraints.

**Scaffolding Level:** 🟡 Medium (apply patterns from Track 01)

---

## Objective

Implement a complete ML classification pipeline with production patterns:
- \>90% accuracy on held-out test set
- <100ms inference latency (p99)
- Multi-class prediction with confidence scores
- Error handling and input validation
- Configuration-driven training
- Automated evaluation

---

## What You'll Learn

- Train/validation/test splitting with stratification
- HOG (Histogram of Oriented Gradients) feature extraction
- Multi-class classification (Logistic Regression, SVM, Random Forest)
- PCA dimensionality reduction
- Model persistence (joblib)
- REST API design for image classification (Flask)
- Classification metrics (accuracy, precision, recall, F1, confusion matrix)
- Unit testing for classification systems

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
02_classification/
├── requirements.txt          # Dependencies (includes scikit-image)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── coding_guidelines.md      # Production patterns & hints
├── SOLUTION.md               # Reference implementation
├── src/
│   ├── data.py               # ✅ Scaffolded (Olivetti Faces loading)
│   ├── features.py           # ✅ Scaffolded (HOG + PCA)
│   ├── models.py             # ⚠️ Hints provided
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
- [ ] Accuracy >90% on test set
- [ ] API returns predictions in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] Confusion matrix shows balanced performance across classes
- [ ] Cross-validation confirms generalization

---

## Key Differences from Track 01 (Regression)

| Aspect | Track 01 (Regression) | Track 02 (Classification) |
|--------|----------------------|---------------------------|
| **Problem** | Predict continuous house prices | Predict discrete face classes |
| **Metrics** | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |
| **Features** | Polynomial expansion + scaling | HOG features + PCA |
| **Models** | Ridge, Lasso, XGBoost | LogisticRegression, SVM, RandomForest |
| **Evaluation** | Residual plots, learning curves | Confusion matrix, classification report |
| **Dataset** | California Housing (20k samples) | Olivetti Faces (400 samples, 40 classes) |

---

## Resources

**Concept Review:**
- [notes/01-ml/02_classification/](../../notes/01-ml/02_classification/) — Complete track
- [notes/01-ml/02_classification/grand-challenge.md](../../notes/01-ml/02_classification/grand-challenge.md) — Constraints

**Implementation Guides:**
- [coding_guidelines.md](coding_guidelines.md) — Hints & patterns
- [SOLUTION.md](SOLUTION.md) — Reference (read AFTER attempting!)

---

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Run tests to see what's expected:**
   ```bash
   make test
   ```

3. **Explore the data (optional):**
   ```python
   from src.data import load_and_split, load_dataset_info
   info = load_dataset_info()
   print(info)
   ```

4. **Train models:**
   ```python
   from src.data import load_and_split
   from src.features import FeatureEngineer
   from src.models import ModelRegistry
   
   # Load data
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
   
   # Engineer features
   engineer = FeatureEngineer(hog_orientations=9, pca_components=50)
   X_train_features = engineer.fit_transform(X_train)
   X_val_features = engineer.transform(X_val)
   
   # Train models
   registry = ModelRegistry()
   metrics = registry.train_logistic_regression(X_train_features, y_train)
   print(f"CV Accuracy: {metrics['cv_accuracy']:.3f}")
   ```

5. **Evaluate:**
   ```python
   from src.evaluate import AutoEvaluator
   
   evaluator = AutoEvaluator()
   test_metrics = evaluator.evaluate(
       registry.models["logistic_regression"],
       X_test_features,
       y_test
   )
   evaluator.plot_confusion_matrix(y_test, test_metrics["predictions"])
   ```

6. **Start API (after training):**
   ```bash
   make serve
   # Test: curl -X GET http://localhost:5000/health
   ```

---

## Production Constraints (from ml-engg-readiness.md)

This exercise addresses gaps identified in the ML engineering audit:

✅ **Addressed in scaffolding:**
- Structured logging with context
- Error handling for HOG extraction failures
- Input validation (image size, format)
- Model persistence patterns
- Configuration-driven hyperparameters

⚠️ **Your responsibility:**
- Comprehensive error handling in API
- Model comparison and selection
- Feature engineering decisions (PCA components, HOG parameters)
- Hyperparameter tuning (C for LogisticRegression, kernel for SVM)

---

## Tips & Hints

1. **Small dataset:** Olivetti Faces has only 400 samples (40 people × 10 images). Use stratified splits to ensure all classes represented.

2. **HOG features:** Already scaffolded. Extracts ~1296 features from 64×64 images. PCA recommended to reduce to 50-100 components.

3. **Model selection:** LogisticRegression often performs best for small datasets. SVM with RBF kernel can be competitive.

4. **Overfitting risk:** With 400 samples, easy to overfit. Use cross-validation and regularization.

5. **API input:** Expects flattened 64×64 grayscale image (4096 values in range [0, 1]).

---

## Challenge Extensions

Once core exercise is complete:

1. **Ensemble methods:** Stack LogisticRegression + SVM predictions
2. **Neural networks:** Add CNN classifier (requires more data augmentation)
3. **Deployment:** Deploy to AWS Lambda or Google Cloud Run
4. **Monitoring:** Build Grafana dashboard with per-class accuracy tracking

---

**Good luck! 🚀**

