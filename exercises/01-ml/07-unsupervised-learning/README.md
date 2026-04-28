# Exercise 07: SegmentAI — Production Clustering System

> **Grand Challenge:** Build a production-grade customer segmentation API that achieves >0.5 silhouette score using unsupervised learning while meeting 5 production constraints.

**Scaffolding Level:** 🟢 Heavy (learn the workflow)

---

## Objective

Implement a complete unsupervised ML pipeline with production patterns:
- >0.5 silhouette score on clustering quality
- <100ms inference latency (p99)
- Automatic optimal cluster discovery (elbow method)
- No labels needed for training
- Cluster drift monitoring
- Configuration-driven training

---

## What You'll Learn

- Unsupervised learning (clustering without labels)
- Distance-based clustering (KMeans, Hierarchical)
- Density-based clustering (DBSCAN)
- Probabilistic clustering (Gaussian Mixture Models)
- Dimensionality reduction (PCA, UMAP)
- Cluster quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
- Elbow method for optimal k
- REST API for clustering
- Unit testing for unsupervised ML

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
07-unsupervised-learning/
├── requirements.txt          # Dependencies (includes umap-learn, hdbscan)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── Dockerfile                # Production container
├── docker-compose.yml        # Multi-service deployment
├── prometheus.yml            # Monitoring config
├── src/
│   ├── __init__.py           # Package exports
│   ├── utils.py              # Logging, timing, validation
│   ├── data.py               # Load Iris/Wine/synthetic blobs
│   ├── features.py           # StandardScaler + PCA + UMAP
│   ├── models.py             # KMeans, DBSCAN, Hierarchical, GMM
│   ├── evaluate.py           # Silhouette, elbow, dendrogram plots
│   ├── monitoring.py         # Prometheus metrics + cluster distribution
│   └── api.py                # POST /cluster endpoint
├── tests/
│   ├── conftest.py           # Pytest configuration
│   ├── test_data.py          # Data loading tests
│   ├── test_features.py      # Feature engineering tests
│   ├── test_models.py        # Clustering algorithm tests
│   └── test_api.py           # API endpoint tests
├── models/                   # Saved models (gitignored)
├── logs/                     # Application logs (gitignored)
└── data/                     # Downloaded datasets (gitignored)
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] Silhouette score >0.5 on Iris dataset
- [ ] API returns cluster assignments in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] Elbow plot shows optimal k
- [ ] Cluster distribution is balanced (no single-sample clusters)

---

## Key Differences from Supervised Learning

1. **No Labels:** Training uses only features (X), not labels (y)
2. **Metrics:** Silhouette score, not accuracy/MAE
3. **Goal:** Discover natural groupings, not predict known targets
4. **Evaluation:** Ground truth labels (if available) used ONLY for validation, NOT training
5. **API Response:** Returns cluster_id + centroid_distance, not continuous value

---

## Clustering Algorithms Included

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **KMeans** | Spherical clusters | Fast, scalable | Requires k, sensitive to outliers |
| **DBSCAN** | Arbitrary shapes | Finds outliers, no k needed | Sensitive to eps/min_samples |
| **Hierarchical** | Dendrogram analysis | No k needed, visual hierarchy | Slow on large data |
| **GMM** | Probabilistic membership | Soft clustering, covariance modeling | Assumes Gaussian distribution |

---

## Unsupervised Metrics

- **Silhouette Score** (-1 to 1): Measures cluster cohesion and separation (higher is better)
- **Davies-Bouldin Index** (0 to ∞): Measures cluster similarity (lower is better)
- **Calinski-Harabasz Score** (0 to ∞): Ratio of between-cluster to within-cluster variance (higher is better)
- **Elbow Method:** Plot inertia vs k to find "elbow" where improvement diminishes

---

## API Endpoints

### POST /cluster
Assign a single sample to a cluster.

**Request:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**
```json
{
  "cluster_id": 0,
  "centroid_distance": 0.45,
  "cluster_size": 50,
  "model": "kmeans"
}
```

### POST /cluster/batch
Assign multiple samples to clusters.

**Request:**
```json
{
  "samples": [
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [6.7, 3.1, 4.7, 1.5]}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"cluster_id": 0, "centroid_distance": 0.45},
    {"cluster_id": 1, "centroid_distance": 0.32}
  ],
  "cluster_distribution": {"0": 1, "1": 1},
  "n_samples": 2
}
```

### GET /health
Health check endpoint.

### GET /metrics
Prometheus metrics (clustering latency, cluster distribution, error count).

### GET /info
Model metadata and configuration.

---

## Docker Deployment

**Build image:**
```bash
docker build -t segmentai:latest .
```

**Run with docker-compose (includes Prometheus):**
```bash
docker-compose up -d
```

**Test endpoints:**
```bash
# Health check
curl http://localhost:5000/health

# Cluster a sample
curl -X POST http://localhost:5000/cluster \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# View metrics
curl http://localhost:5000/metrics
```

---

## Resources

**Concept Review:**
- [notes/01-ml/07_unsupervised_learning/](../../notes/01-ml/07_unsupervised_learning/) — Complete track (when available)
- Scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html

**Implementation Guides:**
- [exercises/01-ml/01-regression/](../01-regression/) — Reference supervised learning patterns
- UMAP documentation: https://umap-learn.readthedocs.io/
- HDBSCAN documentation: https://hdbscan.readthedocs.io/

---

## Common Commands

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Run API server
make run

# Build Docker image
make docker-build

# Start services
make docker-up
```

---

## Production Patterns Applied

✅ **Configuration-driven:** All hyperparameters in `config.yaml`  
✅ **Logging:** Structured logs to `logs/api.log`  
✅ **Monitoring:** Prometheus metrics for latency, cluster distribution, errors  
✅ **Error handling:** Comprehensive validation and exception handling  
✅ **Testing:** Unit tests for all modules  
✅ **Containerization:** Multi-stage Dockerfile for optimized image size  
✅ **API validation:** Pydantic schemas for request validation  
✅ **Drift monitoring:** Track cluster distribution over time  

---

## Next Steps

After completing this exercise, try:
1. **Experiment with datasets:** Try Wine dataset or synthetic blobs with different cluster counts
2. **Tune hyperparameters:** Optimize eps/min_samples for DBSCAN, linkage method for Hierarchical
3. **Add anomaly detection:** Use DBSCAN noise points (-1) as outliers
4. **Implement elbow automation:** Automatically select optimal k from elbow plot
5. **Add dimensionality reduction:** Use UMAP for 2D visualization of high-dimensional clusters

---

## Troubleshooting

**Issue:** Silhouette score is negative  
**Fix:** Try different number of clusters (elbow method) or different algorithm (DBSCAN for non-spherical shapes)

**Issue:** All samples assigned to one cluster  
**Fix:** Check feature scaling (must standardize!), reduce eps for DBSCAN, or increase n_clusters for KMeans

**Issue:** DBSCAN finds too much noise  
**Fix:** Increase eps (neighborhood size) or decrease min_samples

**Issue:** API returns 503 (model not loaded)  
**Fix:** Train a model first and save to `models/best_model.pkl`

---

## License

MIT License - See repository root for details.


