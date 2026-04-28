# Exercise 04: FlixAI — Production Recommender System

> **Grand Challenge:** Build a production-grade movie recommendation API that achieves RMSE <0.9 and precision@10 >0.3 on MovieLens dataset while meeting 5 production constraints.

**Scaffolding Level:** 🟢 Heavy (learn the workflow)

---

## Objective

Implement a complete recommender system pipeline with production patterns:
- RMSE <0.9 on held-out test set
- Precision@10 >0.3 for top-k recommendations
- <100ms inference latency (p99)
- Explainable recommendations (similarity scores)
- Error handling and input validation
- Configuration-driven training
- Automated diagnostics

---

## What You'll Learn

- Collaborative filtering (user-based, item-based)
- Matrix factorization (SVD, ALS)
- Neural collaborative filtering (TensorFlow)
- Similarity search with FAISS
- Ranking metrics (precision@k, recall@k, NDCG)
- Cold start handling
- REST API design for recommendations
- Unit testing for recommender systems

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
04_recommender_systems/
├── requirements.txt          # Dependencies (includes TensorFlow, FAISS)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── src/
│   ├── __init__.py           # Package exports
│   ├── utils.py              # ✅ Scaffolded
│   ├── data.py               # ✅ Scaffolded (user-item matrix loading)
│   ├── features.py           # ⚠️ Hints provided (embeddings)
│   ├── models.py             # ❌ TODO (matrix factorization, neural CF)
│   ├── evaluate.py           # ⚠️ Partial (ranking metrics)
│   ├── monitoring.py         # ✅ Complete
│   └── api.py                # ❌ TODO (recommendation endpoint)
├── tests/
│   ├── conftest.py           # ✅ Complete
│   ├── test_data.py          # ⚠️ Partial
│   ├── test_features.py      # ❌ TODO
│   ├── test_models.py        # ❌ TODO
│   └── test_api.py           # ❌ TODO
└── notebooks/
    └── exploratory.ipynb     # Optional EDA
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] RMSE <0.9 on test set
- [ ] Precision@10 >0.3 on test set
- [ ] API returns recommendations in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] Cold start handling implemented
- [ ] Diversity in top-k recommendations

---

## Dataset

**MovieLens 100K**: 100,000 ratings from 943 users on 1,682 movies
- Rating scale: 1-5 stars
- Sparsity: ~6.3% (highly sparse)
- Format: user_id, item_id, rating, timestamp

**Download**: Automatically fetched by `src/data.py` using sklearn or direct download

---

## Model Architecture

### 1. Matrix Factorization
- Decompose user-item matrix into user and item latent factors
- Optimize using Alternating Least Squares (ALS) or SGD
- Predict rating as dot product of user and item vectors

### 2. Neural Collaborative Filtering
- User and item embeddings
- Multi-layer perceptron for interaction modeling
- Binary cross-entropy loss (implicit feedback)

### 3. FAISS Similarity Search
- Build item similarity index
- Retrieve top-k similar items efficiently
- Used for item-to-item recommendations

---

## API Endpoints

### `POST /recommend`
Request body:
```json
{
  "user_id": 123,
  "k": 10,
  "exclude_seen": true
}
```

Response:
```json
{
  "user_id": 123,
  "recommendations": [
    {"item_id": 456, "score": 4.2},
    {"item_id": 789, "score": 3.9},
    ...
  ],
  "model": "matrix_factorization",
  "latency_ms": 45
}
```

---

## Resources

**Concept Review:**
- [notes/01-ml/04_recommender_systems/](../../notes/01-ml/04_recommender_systems/) — Complete track
- Matrix factorization tutorial: sklearn.decomposition.TruncatedSVD
- Neural CF paper: He et al. (2017)

**Implementation Guides:**
- Follow Track 01 production patterns (logging, error handling, monitoring)
- Use FAISS for efficient similarity search
- Implement cold start handling (popular items fallback)

---

## Production Constraints

1. **Latency**: <100ms p99 for recommendations
2. **Memory**: Model size <500MB
3. **Cold start**: Handle new users/items gracefully
4. **Diversity**: Avoid echo chamber (>50% genre diversity in top-10)
5. **Explainability**: Return similarity scores for transparency

---

## Troubleshooting

**Issue**: TensorFlow import errors
- **Fix**: Ensure Python 3.8-3.11 (TensorFlow 2.13 compatibility)

**Issue**: FAISS not found
- **Fix**: `pip install faiss-cpu` (not `faiss`)

**Issue**: Low precision@k
- **Fix**: Tune regularization, increase n_factors, filter unpopular items

---

## Next Steps

After completing this exercise:
1. Experiment with hybrid models (content + collaborative)
2. Implement session-based recommendations (RNNs)
3. Add A/B testing framework
4. Deploy to cloud with auto-scaling

---

**Questions?** Check [notes/01-ml/04_recommender_systems/faq.md](../../notes/01-ml/04_recommender_systems/faq.md)

