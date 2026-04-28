# Exercise 03: UnifiedAI — Production Neural Network System

> **Grand Challenge:** Build a production-grade neural network classification API that achieves >92% accuracy on synthetic multi-class dataset while meeting 5 production constraints.

**Scaffolding Level:** 🟢 Full Implementation (complete working code provided)

---

## Objective

Implement a complete deep learning classification pipeline with production patterns:
- \>92% accuracy on held-out test set
- <100ms inference latency (p99)
- Multi-class prediction with confidence scores (10 classes)
- Early stopping and learning curve monitoring
- Error handling and input validation
- Configuration-driven training
- TensorFlow/Keras integration

---

## What You'll Learn

- Train/validation/test splitting with stratification for multi-class problems
- Dense neural networks with Keras Sequential API
- 1D CNN architectures for feature vectors
- Early stopping and model checkpointing
- Learning curve visualization (loss and accuracy)
- TensorFlow model persistence and ONNX export capability
- REST API design for feature-based classification (Flask)
- Classification metrics for deep learning (accuracy, precision, recall, F1, confusion matrix)
- Unit testing for neural network systems

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
03_neural_networks/
├── requirements.txt          # Dependencies (includes TensorFlow 2.13+, transformers)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── Dockerfile                # Production Docker image
├── docker-compose.yml        # Multi-container deployment
├── prometheus.yml            # Monitoring configuration
├── README.md                 # This file
├── src/
│   ├── __init__.py           # ✅ Complete
│   ├── utils.py              # ✅ Complete (logging, timing, reproducibility)
│   ├── data.py               # ✅ Complete (synthetic dataset generation)
│   ├── features.py           # ✅ Complete (StandardScaler + optional PCA)
│   ├── models.py             # ✅ Complete (Dense NN, 1D CNN with Keras)
│   ├── evaluate.py           # ✅ Complete (metrics, learning curves)
│   ├── monitoring.py         # ✅ Complete (Prometheus metrics)
│   └── api.py                # ✅ Complete (Flask REST API)
├── tests/
│   ├── __init__.py           # ✅ Complete
│   ├── conftest.py           # ✅ Complete (pytest configuration)
│   ├── test_data.py          # ✅ Complete
│   ├── test_features.py      # ✅ Complete
│   ├── test_models.py        # ✅ Complete (marked slow)
│   └── test_api.py           # ✅ Complete
├── models/                   # Model artifacts saved here
├── data/                     # Data cached here
└── logs/                     # Application logs
```

---

## Success Criteria

Your exercise is complete when:
- [x] All tests pass: `pytest tests/`
- [x] Accuracy >92% on test set
- [x] API returns predictions in <100ms
- [x] Code passes linting: `black . && flake8 src/`
- [x] Learning curves show convergence without overfitting
- [x] Early stopping triggers before max epochs (efficient training)

---

## Key Differences from Track 01 (Regression) and Track 02 (Classification)

| Aspect | Track 01 (Regression) | Track 02 (Classification) | Track 03 (Neural Networks) |
|--------|----------------------|---------------------------|----------------------------|
| **Problem** | Predict continuous prices | Predict face classes | Predict synthetic classes |
| **Models** | Ridge, Lasso, XGBoost | LogisticRegression, SVM, RandomForest | Dense NN, 1D CNN (TensorFlow/Keras) |
| **Metrics** | MAE, RMSE, R² | Accuracy, P, R, F1 | Accuracy, P, R, F1 + Learning Curves |
| **Features** | Polynomial + scaling | HOG + PCA | StandardScaler + optional PCA |
| **Evaluation** | Residual plots | Confusion matrix | Confusion matrix + Learning Curves |
| **Dataset** | California Housing | Olivetti Faces (400 samples) | Synthetic multi-class (10k samples) |
| **Framework** | scikit-learn | scikit-learn | TensorFlow 2.13+ / Keras |
| **Training** | Fit once | Fit once | Epochs with early stopping |
| **API Input** | Feature dict | Flattened image | Feature vector (20 features) |

---

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Run tests to verify installation:**
   ```bash
   make test
   # Or: pytest tests/ -v -m "not slow"  # Skip slow model training tests
   ```

3. **Explore the data:**
   ```python
   from src.data import load_and_split
   
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
       n_samples=10000,
       n_features=20,
       n_classes=10
   )
   print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
   ```

4. **Train models:**
   ```python
   from src.data import load_and_split
   from src.features import FeatureEngineer
   from src.models import ModelRegistry
   
   # Load data
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
   
   # Engineer features
   engineer = FeatureEngineer(scale_features=True, pca_components=None)
   X_train_features = engineer.fit_transform(X_train)
   X_val_features = engineer.transform(X_val)
   X_test_features = engineer.transform(X_test)
   
   # Train Dense Neural Network
   registry = ModelRegistry()
   metrics = registry.train_dense_nn(
       X_train_features, y_train, X_val_features, y_val,
       architecture=[128, 64, 32],
       dropout=0.3,
       learning_rate=0.001,
       batch_size=32,
       epochs=50,
       early_stopping_patience=5
   )
   print(f"Val Accuracy: {metrics['val_accuracy']:.3f}")
   print(f"Epochs trained: {metrics['epochs_trained']}")
   ```

5. **Evaluate with learning curves:**
   ```python
   from src.evaluate import AutoEvaluator
   
   evaluator = AutoEvaluator()
   
   # Evaluate on test set
   test_metrics = evaluator.evaluate(
       registry.models["dense_nn"],
       X_test_features,
       y_test,
       model_name="dense_nn",
       set_name="test"
   )
   print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
   
   # Plot learning curves
   evaluator.plot_learning_curves(registry.histories["dense_nn"])
   
   # Plot confusion matrix
   evaluator.plot_confusion_matrix(y_test, evaluator.predictions)
   ```

6. **Save model and start API:**
   ```python
   from pathlib import Path
   import joblib
   
   # Save model
   registry.save_model("dense_nn", Path("models/best_model.h5"))
   
   # Save feature engineer
   joblib.dump(engineer, "models/feature_engineer.pkl")
   ```
   
   ```bash
   # Start API
   make serve
   # Test: curl -X GET http://localhost:5000/health
   ```

7. **Make predictions via API:**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, -0.5, 0.8, 1.2, -0.3, 0.7, -0.9, 0.4, 0.6, -0.1, 0.3, 0.9, -0.7, 0.2, -0.4, 0.5, 0.1, -0.6, 0.8]}'
   ```

---

## Production Constraints

This exercise addresses neural network-specific production requirements:

✅ **Fully implemented:**
- Structured logging with TensorFlow integration
- Early stopping to prevent overfitting
- Input validation for feature vectors
- Model persistence (Keras .h5 format)
- Configuration-driven hyperparameters (architecture, dropout, learning rate)
- Learning curve visualization
- Prometheus monitoring
- Docker deployment

⚠️ **Your extensions (optional):**
- Hyperparameter tuning with Optuna
- Model versioning with MLflow
- ONNX export for cross-platform deployment
- Data augmentation strategies
- Ensemble of Dense + CNN models

---

## Tips & Hints

1. **Early stopping:** Neural networks benefit from early stopping to prevent overfitting. Monitor validation loss and restore best weights.

2. **Learning curves:** Always plot loss/accuracy curves. If train loss << val loss, you're overfitting. Consider:
   - Increase dropout
   - Reduce model complexity
   - Add L2 regularization

3. **Batch size:** Start with 32. Larger batches (64-128) train faster but may generalize worse. Smaller batches (8-16) are noisier but can help escape local minima.

4. **Architecture:** The default [128, 64, 32] is a good starting point. Deeper networks aren't always better for small feature sets (20 features).

5. **1D CNN:** The `train_cnn_1d` method treats the 20 features as a 1D sequence. CNNs can learn local patterns in feature space. Compare against Dense NN.

6. **Feature scaling:** Critical for neural networks. Always use StandardScaler before training.

---

## Common Issues

**Issue:** Model achieves 100% train accuracy but poor val accuracy  
**Fix:** Increase dropout, reduce model size, or add L2 regularization

**Issue:** Training is very slow  
**Fix:** Increase batch_size, reduce epochs, or use GPU

**Issue:** `ImportError: No module named 'tensorflow'`  
**Fix:** Ensure TensorFlow is installed: `pip install tensorflow>=2.13.0,<3.0.0`

**Issue:** API returns 503 "Model not loaded"  
**Fix:** Train and save model first: `registry.save_model("dense_nn", Path("models/best_model.h5"))`

---

## Deployment

**Docker:**
```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose for full stack (API + Prometheus)
docker-compose up -d
```

**Prometheus metrics:**
- Access at: http://localhost:9090
- Query: `prediction_latency_seconds`
- Dashboard: Monitor p50, p95, p99 latencies

---

## Resources

**Concept Review:**
- [notes/01-ml/03_neural_networks/](../../notes/01-ml/03_neural_networks/) — Complete track
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Keras Sequential API](https://keras.io/guides/sequential_model/)

**Related Tracks:**
- Track 01 (Regression) — Foundation for ML pipelines
- Track 02 (Classification) — Multi-class classification patterns
- Track 05 (Multimodal AI) — Advanced deep learning architectures

---

## License

MIT License - See main repository for details.

