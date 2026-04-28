# Exercise 05: FraudShield — Production Anomaly Detection System

> **Grand Challenge:** Build a production-grade fraud detection API that achieves >85% precision at 80% recall on imbalanced dataset while meeting 5 production constraints.

**Scaffolding Level:** 🟡 Medium (apply patterns from Tracks 01 & 02)

---

## Objective

Implement a complete ML anomaly detection pipeline with production patterns:
- \>85% precision at 80% recall on held-out test set
- <100ms inference latency (p99)
- Handle imbalanced data (10% anomalies)
- Explainable predictions (SHAP explanations)
- Configuration-driven training
- Automated evaluation with ROC curves and precision@K

---

## What You'll Learn

- Imbalanced data handling (stratified splits, contamination parameter)
- Anomaly detection algorithms (Isolation Forest, One-Class SVM, Autoencoder)
- Anomaly-specific metrics (precision@K, ROC-AUC, anomaly rate monitoring)
- Reconstruction-based detection (autoencoder approach)
- Model persistence and API deployment
- REST API for anomaly scoring
- Production monitoring for fraud detection systems

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
05_anomaly_detection/
├── requirements.txt          # Dependencies (includes scipy, tensorflow, shap)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── src/
│   ├── data.py               # Synthetic imbalanced data generation
│   ├── features.py           # StandardScaler (anomaly detection uses raw features)
│   ├── models.py             # IsolationForest, OneClassSVM, Autoencoder
│   ├── evaluate.py           # Binary metrics + precision@K + ROC curves
│   ├── monitoring.py         # Anomaly rate tracking
│   └── api.py                # POST /detect returns is_anomaly + score + explanation
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
└── Docker deployment files
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] Precision ≥85% at 80% recall on test set
- [ ] API returns predictions in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] ROC-AUC >0.90
- [ ] Confusion matrix shows acceptable false positive rate
- [ ] SHAP explanations available via API

---

## Key Adaptations from Track 01 (Regression)

| Aspect | Track 01 (Regression) | Track 05 (Anomaly Detection) |
|--------|----------------------|------------------------------|
| **Problem** | Predict continuous house prices | Detect binary anomalies (fraud) |
| **Data** | Balanced California Housing | Imbalanced synthetic (10% anomalies) |
| **Metrics** | MAE, RMSE, R² | Precision, Recall, F1, ROC-AUC, Precision@K |
| **Features** | Polynomial expansion + scaling | Scaling only (preserve anomaly patterns) |
| **Models** | Ridge, Lasso, XGBoost | IsolationForest, OneClassSVM, Autoencoder |
| **Evaluation** | Residual plots, learning curves | ROC curve, confusion matrix, score distribution |
| **API** | /predict returns price | /detect returns is_anomaly + score + explanation |
| **Monitoring** | Prediction value distribution | Anomaly rate gauge, alert thresholds |

---

## Anomaly Detection Models

### 1. Isolation Forest
- **Algorithm:** Tree-based outlier detection
- **Intuition:** Anomalies are easier to isolate (require fewer splits)
- **Hyperparameters:** `contamination`, `n_estimators`
- **Best for:** Fast training, handles high-dimensional data

### 2. One-Class SVM
- **Algorithm:** Density-based outlier detection
- **Intuition:** Learn a decision boundary around normal data
- **Hyperparameters:** `nu` (contamination upper bound), `kernel`
- **Best for:** Low-dimensional data, complex decision boundaries

### 3. Autoencoder
- **Algorithm:** Neural network reconstruction error
- **Intuition:** Normal data reconstructs well, anomalies have high error
- **Hyperparameters:** `encoding_dim`, `epochs`, `learning_rate`
- **Best for:** High-dimensional data, when normal patterns are learnable

---

## Key Metrics

### Precision@K
**What:** Fraction of top-K predictions that are true anomalies  
**Why:** In fraud detection, we investigate top suspicious cases  
**Target:** >90% precision at K=10

### ROC-AUC
**What:** Area under ROC curve (True Positive Rate vs False Positive Rate)  
**Why:** Measures model's ability to rank anomalies higher than normal  
**Target:** >0.90

### Anomaly Rate
**What:** Rolling window anomaly detection rate  
**Why:** Monitor for data drift or model degradation  
**Alert:** If rate exceeds 15% (above expected 10%)

---

## API Endpoints

### POST /detect
Detect if a single transaction is anomalous.

**Request:**
```json
{
  "features": [0.5, 1.2, -0.3, ...]  // 20 feature values
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "anomaly_score": 0.85,
  "confidence": "high",
  "model": "isolation_forest",
  "current_anomaly_rate": 0.12,
  "explanation": "Strong anomaly signal detected. This transaction deviates significantly from normal patterns."
}
```

### POST /batch_detect
Detect anomalies in a batch of transactions.

**Request:**
```json
{
  "samples": [
    [0.5, 1.2, ...],
    [0.3, -0.5, ...]
  ]
}
```

### GET /health
Health check endpoint.

### GET /metrics
Prometheus metrics endpoint.

---

## Development Workflow

1. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

2. **Train models:**
   ```python
   from src.data import load_and_split
   from src.features import FeatureEngineer
   from src.models import ModelRegistry
   
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
   engineer = FeatureEngineer()
   X_train = engineer.fit_transform(X_train)
   
   registry = ModelRegistry()
   registry.train_isolation_forest(X_train, y_train)
   registry.train_one_class_svm(X_train, y_train)
   ```

3. **Start API:**
   ```bash
   python -m src.api
   # Or with gunicorn:
   gunicorn --bind 0.0.0.0:5000 --workers 4 src.api:app
   ```

4. **Test API:**
   ```bash
   curl -X POST http://localhost:5000/detect \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 1.2, -0.3, 0.8, 1.1, -0.5, 0.9, 1.3, 0.2, -0.4, 0.7, 1.0, -0.2, 0.6, 0.4, -0.1, 0.8, 1.2, 0.3, -0.3]}'
   ```

---

## Docker Deployment

Build and run in Docker:
```bash
docker-compose up --build
```

Services:
- FraudShield API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Resources

**Concept Review:**
- [notes/01-ml/](../../notes/01-ml/) — ML fundamentals
- Anomaly detection resources (external)

**Implementation Patterns:**
- [exercises/01-ml/01_regression/](../01_regression/) — Production patterns
- [exercises/01-ml/02_classification/](../02_classification/) — Classification metrics

---

## Common Issues

### Issue: Low precision
**Solution:** Adjust `contamination` parameter or threshold

### Issue: High false positive rate
**Solution:** Use stricter threshold, ensemble multiple models

### Issue: Poor separation in ROC curve
**Solution:** Add more informative features, try different model

### Issue: Model overfits to training anomalies
**Solution:** Ensure training only on normal samples (for autoencoder)

---

## Production Checklist

- [ ] Model achieves >85% precision at 80% recall
- [ ] API latency <100ms (p99)
- [ ] All tests pass with >80% coverage
- [ ] Docker image builds successfully
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set for anomaly rate
- [ ] SHAP explanations working
- [ ] Documentation complete

---

**Ready to detect fraud? Start with `python -m src.data` to verify data generation!** 🚀

