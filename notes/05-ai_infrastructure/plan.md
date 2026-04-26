# AI Infrastructure Track — Implementation Plan

> **Status:** Ch.1-10 complete! 🎉  
> **Last updated:** April 26, 2026  
> **Focus:** AI/ML-specific infrastructure (GPU optimization, model serving, experiment tracking, feature engineering infrastructure)  
> **Note:** For generic DevOps (Docker, Kubernetes, CI/CD, monitoring), see `notes/06-devops_fundamentals/`

---

## Current Status

| Chapter | Status | README | Notebook | Supplement (Azure) |
|---------|--------|--------|----------|-------------------|
| Ch.1: GPU Architecture | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.2: Memory & Compute Budgets | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.3: Quantization & Precision | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.4: Parallelism & Distributed Training | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.5: Inference Optimization | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.6: Model Serving Frameworks | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.7: AI-Specific Networking | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.8: Feature Stores & Data Infrastructure | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.9: ML Experiment Tracking & Model Registry | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.10: Production ML Monitoring & A/B Testing | ✅ Complete | ✅ | ✅ | ✅ |

---

## Completed Chapters

### Ch.7 — AI-Specific Networking ✅

**Location:** `notes/05-ai_infrastructure/ch07_ai_specific_networking/`

**Content:**
- README.md: Challenge → Core Idea → Running Example (Llama-2-70B multi-GPU with NVLink)
- notebook.ipynb: 10 cells covering NVLink detection, single-GPU baseline, PCIe bottleneck, NVLink speedup, topology detection, latency comparison
- notebook_supplement.ipynb: Azure ND-series VMs (InfiniBand), multi-node inference, NCCL bandwidth tests, cost analysis
- 4 diagrams: network topology (PCIe vs NVLink), bandwidth comparison, latency heatmap, decision tree
- Animation: communication patterns showing PCIe bottleneck vs NVLink efficiency

**Tech stack:** PyTorch distributed (NCCL), nvidia-smi topology, Azure ND-series (InfiniBand + NVLink)

### Ch.8 — Feature Stores & Data Infrastructure ✅

**Location:** `notes/05-ai_infrastructure/ch08_feature_stores/`

**Content:**
- README.md: Challenge → Core Idea → Running Example (Recommendation system for document extraction)
- notebook.ipynb: 10 cells covering Feast setup, feature definitions, materialization, online/offline serving
- notebook_supplement.ipynb: 8 cells covering Azure ML Feature Store, Azure Cache for Redis, cost analysis
- 4 diagrams: feature store architecture, training-serving flow, latency comparison, versioning timeline
- Animation: latency reduction (380ms → 8ms with Redis)
- Gen scripts: `gen_ch08_architecture.py`, `gen_ch08_training_serving_flow.py`, `gen_ch08_latency_comparison.py`, `gen_ch08_versioning_timeline.py`, `gen_ch08_latency_animation.py`

**Tech stack:** Feast (local), Redis (online store), Parquet (offline store), Azure ML Feature Store + Azure Cache for Redis (supplement)

### Ch.9 — ML Experiment Tracking & Model Registry ✅

**Location:** `notes/05-ai_infrastructure/ch09_ml_experiment_tracking/`

**Content:**
- README.md: Challenge → Core Idea → Running Example (BERT fine-tuning sweep)
- notebook.ipynb: 10 cells covering MLflow tracking, model registry, and DVC versioning
- notebook_supplement.ipynb: Azure ML HyperDrive and cost tracking
- 4 diagrams: experiment lifecycle, MLflow UI, model registry stages, DVC pipeline
- Animation: debugging time reduction (2 days → 30 min)
- Helper scripts: `start-mlflow-server.sh`, `init-dvc-repo.sh`

**Tech stack:** MLflow (local), DVC, Weights & Biases (optional)

### Ch.10 — Production ML Monitoring & A/B Testing ✅

**Location:** `notes/05-ai_infrastructure/ch10_production_ml_monitoring/`

**Content:**
- README.md: Challenge → Core Idea → Running Example (BERT sentiment classifier monitoring)
- notebook.ipynb: 10 cells covering drift detection, A/B testing, and automated rollback
- notebook_supplement.ipynb: Azure ML endpoints, Application Insights, and Monitor alerts
- 4 diagrams: drift types, A/B flow, monitoring dashboard, gradual rollout
- Animation: detection time reduction (2 weeks → 2 hours)
- Helper scripts: `start-evidently-server.sh`, `generate-drift-report.py`, `ab-test-controller.py`

**Tech stack:** Evidently AI, MLflow Model Serving, custom A/B framework (Python + SQLite)

---

## Infrastructure Setup (Complete) ✅

### Azure Supplements (Ch.1-5)

All chapters now have Azure ML supplements with credential templates:
- Ch.1: Azure GPU VM SKUs comparison and cost calculator
- Ch.2: Azure ML compute cluster memory profiling and VRAM estimation
- Ch.3: Quantized model deployment and FP16 vs INT8 cost comparison
- Ch.4: Multi-node PyTorch DDP and Azure ML distributed training
- Ch.5: vLLM on AKS with Application Gateway load balancing

**Template pattern:** All supplements follow the same structure (credentials → SDK setup → Azure-specific content)

### Security & Git Hooks

**Pre-push secret removal hooks installed:**
- `scripts/hooks/pre-push-remove-secrets.sh` (Unix/Linux/macOS)
- `scripts/hooks/pre-push-remove-secrets.ps1` (Windows)
- Registered in `scripts/install-hooks.sh` and `scripts/install-hooks.ps1`

**Patterns detected:** `API_KEY`, `SECRET`, `AZURE_*`, `OPENAI_*`, `AWS_*`, `WANDB_*`

### Dependencies

**AI Infrastructure packages added to setup scripts:**
- MLflow (experiment tracking + model registry)
- Evidently (drift detection + data quality)
- DVC (data version control)
- TensorBoard (alternative visualization)
- Weights & Biases (optional cloud tracking)

**Scripts updated:**
- `scripts/setup.ps1` (Windows)
- `scripts/setup.sh` (Unix/Linux/macOS)



---

## Next Steps (Ch.6-8 Implementation)

### Phase 5: Ch.6 — Model Serving Frameworks
- [ ] Write Ch.6 README.md
- [ ] Create Ch.6 notebook (vLLM + ONNX Runtime comparison)
- [ ] Create Ch.6 notebook_supplement (Azure ML deployment)
- [ ] Generate diagrams (serving architecture, throughput comparison)

### Phase 6: Ch.7 — AI-Specific Networking  
- [ ] Write Ch.7 README.md
- [ ] Create Ch.7 notebook (NVLink vs PCIe benchmarks)
- [ ] Create Ch.7 notebook_supplement (Azure InfiniBand setup)
- [ ] Generate diagrams (network topology, bandwidth comparison)

### Phase 7: Ch.8 — Feature Stores & Data Infrastructure
- [ ] Write Ch.8 README.md
- [ ] Create Ch.8 notebook (Feast local setup)
- [ ] Create Ch.8 notebook_supplement (Azure ML Feature Store)
- [ ] Generate diagrams (feature serving architecture, online/offline stores)

---

## Design Principles

- **AI/ML-specific focus:** GPU optimization, model serving, experiment tracking, feature stores (not generic DevOps)
- **DevOps prerequisite:** Students should complete `notes/06-devops_fundamentals/` first
- **Local-first philosophy:** Every concept must work on localhost before showing cloud alternative
- **Zero-cost teaching:** Free tier or local tools only; paid subscriptions optional for Azure supplements
- **Security by default:** Secrets hook is mandatory; supplements must have empty credential strings
- **Cross-platform:** All scripts must work on Windows (PowerShell) and Unix (Bash)

---

## Track Complete! 🎉

All 10 chapters of the AI Infrastructure track are now complete:
- ✅ Ch.1-5: Core infrastructure (GPU, memory, quantization, parallelism, inference optimization)
- ✅ Ch.6-8: Production serving (serving frameworks, networking, feature stores)
- ✅ Ch.9-10: MLOps (experiment tracking, production monitoring)

**What's next?**
- Students can now build production ML systems end-to-end
- All constraint targets met: Cost (<$15k/mo), Latency (≤2s), Throughput (≥10k req/day)
- Ready for advanced topics: AutoML, edge deployment, federated learning

---