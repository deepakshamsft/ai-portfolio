# AI Infrastructure Track — Implementation Plan

> **Status:** Ch.1-5 complete. Ch.6-10 planned.  
> **Last updated:** April 26, 2026  
> **Focus:** AI/ML-specific infrastructure (GPU optimization, model serving, experiment tracking, feature engineering infrastructure)  
> **Note:** For generic DevOps (Docker, Kubernetes, CI/CD, monitoring), see `notes/devops_fundamentals/`

---

## Current Status

| Chapter | Status | README | Notebook | Supplement (Azure) |
|---------|--------|--------|----------|-------------------|
| Ch.1: GPU Architecture | ✅ Complete | ✅ | ✅ | ❌ TODO |
| Ch.2: Memory & Compute Budgets | ✅ Complete | ✅ | ✅ | ❌ TODO |
| Ch.3: Quantization & Precision | ✅ Complete | ✅ | ✅ | ❌ TODO |
| Ch.4: Parallelism & Distributed Training | ✅ Complete | ✅ | ✅ | ❌ TODO |
| Ch.5: Inference Optimization | ✅ Complete | ✅ | ✅ | ❌ TODO |
| Ch.6: Model Serving Frameworks | ⏳ Planned | ❌ | ❌ | ❌ |
| Ch.7: AI-Specific Networking | ⏳ Planned | ❌ | ❌ | ❌ |
| Ch.8: Feature Stores & Data Infrastructure | ⏳ Planned | ❌ | ❌ | ❌ |
| Ch.9: ML Experiment Tracking & Model Registry | ⏳ Planned | ❌ | ❌ | ❌ |
| Ch.10: Production ML Monitoring & A/B Testing | ⏳ Planned | ❌ | ❌ | ❌ |

---

## Ch.9 — ML Experiment Tracking & Model Registry

### Overview

**Running example:** Track hyperparameter sweep for BERT fine-tuning (100+ experiments)  
**Constraint:** Must compare runs across 2 weeks, 3 team members, 5 GPUs  
**Tech stack (FREE):**
- MLflow (local tracking server + model registry)
- DVC (data version control)
- Weights & Biases Free Tier (optional cloud alternative)

**Prerequisites from DevOps Track:**
- Git basics (version control)
- Basic Python packaging (for reproducible environments)

### Implementation Tasks

#### ✅ Prerequisites

- [ ] **Install ML Experiment Tracking dependencies via root setup script**
  - File: `scripts/setup.ps1` and `scripts/setup.sh`
  - Dependencies:
    - `mlflow` (experiment tracking + model registry)
    - `dvc` (data version control)
    - `tensorboard` (alternative visualization)
    - `wandb` (optional: Weights & Biases)
  - Update: Add ML experiment tracking kernel registration to Jupyter

- [ ] **Create secrets removal pre-push hook**
  - File: `scripts/hooks/pre-push-remove-secrets.sh` and `.ps1`
  - Scan for patterns:
    - `AZURE_API_KEY = "..."`
    - `OPENAI_API_KEY = "..."`
    - `AWS_ACCESS_KEY = "..."`
    - `WANDB_API_KEY = "..."`
    - Any `XXX_API_KEY` or `XXX_SECRET` patterns
  - Strip secrets or fail push with warning
  - Register in `scripts/install-hooks.ps1` and `.sh`

#### 📝 Chapter Content

- [ ] **Ch.9 README.md**
  - Structure: Follow Ch.1-5 pattern (Challenge → Core Idea → Running Example → etc.)
  - Sections:
    1. **The Challenge:** Lost experiments, irreproducible results, "which hyperparameters gave 94% accuracy?"
    2. **Core Idea:** Every experiment = logged parameters + metrics + artifacts (model, plots)
    3. **Running Example:** BERT fine-tuning sweep (learning rate, batch size, warmup steps)
       - Step 1: Log first experiment (baseline)
       - Step 2: Hyperparameter sweep (grid search: 3 LR × 2 batch sizes × 2 warmup = 12 runs)
       - Step 3: Compare runs (MLflow UI)
       - Step 4: Register best model (staging → production)
       - Step 5: Version dataset with DVC
    4. **Mental Model:** Experiment = (code version, data version, hyperparameters, metrics, artifacts)
    5. **Code Skeleton:** MLflow tracking setup, model registry workflow, DVC pipeline
    6. **What Can Go Wrong:**
       - Forgot to log random seed → irreproducible
       - Logged metrics but not hyperparameters → can't debug
       - Deleted local model files → registry points to nothing
       - Dataset changed but not versioned → results meaningless
    7. **Progress Check:** Given MLflow UI screenshot, identify which run to deploy
    8. **Bridge to Ch.10:** After tracking experiments, deploy best model (A/B testing, drift monitoring)

- [ ] **Ch.9 notebook.ipynb (Local)**
  - Cell 1: Setup MLflow tracking (local server on localhost:5000)
  - Cell 2: Log first experiment (baseline BERT fine-tuning)
    - Log hyperparameters: `learning_rate=5e-5, batch_size=32, epochs=3`
    - Log metrics: `accuracy, f1_score, loss` (per epoch)
    - Log artifacts: `model.pt, confusion_matrix.png`
  - Cell 3: Hyperparameter sweep (12 experiments in loop)
  - Cell 4: MLflow UI walkthrough (compare runs, parallel coordinates plot)
  - Cell 5: Search API (find best run by metric: `max(accuracy)`)
  - Cell 6: Model registry (register best model, tag as "staging")
  - Cell 7: Load model from registry (inference on test set)
  - Cell 8: Promote model to "production" stage
  - Cell 9: DVC setup (version training dataset)
  - Cell 10: Reproduce experiment from run ID (fetch code, data, hyperparameters)

- [ ] **Ch.9 notebook_supplement.ipynb (Azure)**
  - Cell 1: Azure ML credentials setup
    ```python
    # USER: Replace with your Azure ML credentials
    AZURE_SUBSCRIPTION_ID = ""  # Will be stripped by pre-push hook
    AZURE_RESOURCE_GROUP = ""
    AZURE_WORKSPACE_NAME = ""
    ```
  - Cell 2: Azure ML workspace connection
  - Cell 3: Log experiment to Azure ML (`Run.start_logging()`)
  - Cell 4: Hyperparameter sweep with Azure ML HyperDrive
  - Cell 5: Azure ML model registry (register + version)
  - Cell 6: Download model from registry for inference
  - Cell 7: Compare Azure ML vs. MLflow UI
  - Cell 8: Cost tracking (Azure ML compute costs per experiment)

#### 🖼️ Diagrams & Assets

- [ ] **Generate diagrams**
  - `gen_ch09_experiment_lifecycle.py` → Code + data + params → run → metrics + artifacts
  - `gen_ch09_mlflow_ui_comparison.py` → Side-by-side runs table
  - `gen_ch09_model_registry_stages.py` → None → Staging → Production → Archived
  - `gen_ch09_dvc_pipeline.py` → Raw data → preprocess → train → evaluate (versioned)

- [ ] **Animation**
  - `gen_ch09_experiment_needle.py` → Progress animation (debugging time: 2 days → 30 min with tracking)

#### 🔧 Supporting Scripts

- [ ] **Setup helpers**
  - `scripts/ml_tracking/start-mlflow-server.sh` → Launch MLflow UI (localhost:5000)
  - `scripts/ml_tracking/init-dvc-repo.sh` → Initialize DVC for data versioning

---

## Ch.10 — Production ML Monitoring & A/B Testing

### Overview

**Running example:** Deploy 2 BERT model versions, measure performance in production  
**Constraint:** Detect degradation within 24 hours, roll back in <5 minutes  
**Tech stack (FREE):**
- Evidently AI (drift detection, data quality monitoring)
- MLflow Model Serving (local inference server)
- Custom A/B testing framework (Python + SQLite)

**Prerequisites from DevOps Track:**
- Docker basics (for model serving containers)
- Monitoring basics (Prometheus/Grafana from DevOps Ch.5)
- Networking basics (reverse proxy from DevOps Ch.7)

### Implementation Tasks

#### 📝 Chapter Content

- [ ] **Ch.10 README.md**
  - Structure: Follow Ch.1-5 pattern
  - Sections:
    1. **The Challenge:** Model accuracy drops in production but you don't notice for weeks
    2. **Core Idea:** Continuous monitoring = data drift + prediction drift + performance metrics
    3. **Running Example:** Sentiment classifier (BERT) in production
       - Step 1: Deploy model v1 (baseline)
       - Step 2: Monitor data drift (input text distribution changes)
       - Step 3: Monitor prediction drift (output label distribution changes)
       - Step 4: Deploy model v2 (A/B test: 10% traffic)
       - Step 5: Compare business metrics (v1 vs. v2: accuracy, latency, user satisfaction)
       - Step 6: Gradual rollout (10% → 50% → 100%) or rollback
    4. **Mental Model:** Train-time metrics ≠ production metrics (data drift, concept drift, adversarial inputs)
    5. **Code Skeleton:** Evidently monitoring dashboard, A/B test controller, rollback script
    6. **What Can Go Wrong:**
       - Silent degradation (no alerts configured)
       - False alarms (noisy metrics)
       - Rollback too slow (no automated cutover)
       - A/B test bias (traffic not random)
    7. **Progress Check:** Given drift report, decide: rollback, retrain, or do nothing?
    8. **Bridge to Future:** Scaling to multi-model systems (ensemble serving, cascading models)

- [ ] **Ch.10 notebook.ipynb (Local)**
  - Cell 1: Deploy model v1 with MLflow model serving (localhost:5001)
  - Cell 2: Generate synthetic production traffic (simulate user requests)
  - Cell 3: Log predictions + ground truth (for performance monitoring)
  - Cell 4: Evidently drift report (compare training data vs. production data)
  - Cell 5: Detect data drift (feature distribution shift)
  - Cell 6: Detect prediction drift (output class imbalance)
  - Cell 7: Deploy model v2 (A/B test: 10% traffic gets v2)
  - Cell 8: Compare metrics (v1 vs. v2: accuracy, F1, latency)
  - Cell 9: Gradual rollout script (increase v2 traffic to 100%)
  - Cell 10: Automated rollback (if accuracy drops below threshold)

- [ ] **Ch.10 notebook_supplement.ipynb (Azure)**
  - Cell 1: Azure ML credentials
  - Cell 2: Deploy model to Azure ML endpoint (v1 + v2)
  - Cell 3: Azure ML traffic splitting (A/B test configuration)
  - Cell 4: Monitor with Azure Application Insights (request rate, latency, errors)
  - Cell 5: Custom metrics (accuracy, F1) logged to Application Insights
  - Cell 6: Azure Monitor alerts (trigger on accuracy drop)
  - Cell 7: Automated rollback with Azure ML deployment config
  - Cell 8: Cost comparison (v1 vs. v2 inference costs)

#### 🖼️ Diagrams & Assets

- [ ] **Generate diagrams**
  - `gen_ch10_drift_types.py` → Data drift vs. prediction drift vs. concept drift
  - `gen_ch10_ab_test_flow.py` → Traffic split → metrics collection → decision → rollout/rollback
  - `gen_ch10_monitoring_dashboard.py` → Evidently UI mockup
  - `gen_ch10_gradual_rollout.py` → Timeline: 10% → 25% → 50% → 100% traffic migration

- [ ] **Animation**
  - `gen_ch10_monitoring_needle.py` → Progress animation (detection time: 2 weeks → 2 hours)

#### 🔧 Supporting Scripts

- [ ] **Setup helpers**
  - `scripts/ml_monitoring/start-evidently-server.sh` → Launch Evidently UI
  - `scripts/ml_monitoring/generate-drift-report.py` → Ad-hoc drift analysis
  - `scripts/ml_monitoring/ab-test-controller.py` → Traffic splitting logic

---

## Ch.6-8 Quick Summaries (Planned)

### Ch.6 — Model Serving Frameworks

**Focus:** vLLM, TensorRT, ONNX Runtime, TorchServe (AI-specific inference optimization)  
**Running example:** Deploy Llama-2-7B with vLLM (continuous batching, KV cache optimization)  
**Comparison:** vLLM vs. HuggingFace Transformers (throughput, latency)  
**Prerequisites from DevOps:** Docker (for containerized serving)

### Ch.7 — AI-Specific Networking

**Focus:** GPU-to-GPU communication (NVLink, InfiniBand, RDMA), model parallelism networking  
**Running example:** Multi-GPU inference with tensor parallelism (Llama-2-70B split across 4 GPUs)  
**Comparison:** PCIe vs. NVLink bandwidth impact on latency  
**Prerequisites from DevOps:** Networking basics (DevOps Ch.7)

### Ch.8 — Feature Stores & Data Infrastructure

**Focus:** Real-time feature serving, offline feature engineering, feature versioning  
**Running example:** Recommendation system (user features + item features → inference)  
**Tech stack:** Feast (local) or Azure ML Feature Store (cloud)  
**Prerequisites from DevOps:** Database basics, data pipelines

---

## Azure Supplement Implementation (Ch.1-5 Backfill)

### Template Structure

Every `*_supplement.ipynb` follows this pattern:

```python
# Cell 1: Credentials (stripped by pre-push hook)
# USER: Replace with your Azure credentials
AZURE_SUBSCRIPTION_ID = ""  # Leave empty before commit
AZURE_RESOURCE_GROUP = ""
AZURE_WORKSPACE_NAME = ""
AZURE_API_KEY = ""  # NEVER commit real keys

# Cell 2: Azure SDK setup
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(credential, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME)

# Cell 3+: Same content as main notebook, but using Azure APIs
```

### Tasks

- [ ] **Ch.1: GPU Architecture**
  - `gpu_architecture/notebook_supplement.ipynb`
  - Azure GPU VM SKUs comparison (NC-series, ND-series, NV-series)
  - Cost calculator using Azure Pricing API

- [ ] **Ch.2: Memory & Compute Budgets**
  - `memory_and_compute_budgets/notebook_supplement.ipynb`
  - Azure ML compute cluster memory profiling
  - VRAM estimation for Azure GPU instances

- [ ] **Ch.3: Quantization & Precision**
  - `quantization_and_precision/notebook_supplement.ipynb`
  - Deploy quantized model to Azure ML endpoint
  - Compare latency/cost (FP16 vs INT8 on Azure)

- [ ] **Ch.4: Parallelism & Distributed Training**
  - `parallelism_and_distributed_training/notebook_supplement.ipynb`
  - Azure ML distributed training (PyTorch DDP)
  - Multi-node GPU cluster setup on Azure

- [ ] **Ch.5: Inference Optimization**
  - `inference_optimization/notebook_supplement.ipynb`
  - Deploy vLLM to Azure Kubernetes Service (AKS)
  - Azure Application Gateway load balancing

---

## Security & Git Hooks

### Pre-Push Secret Removal Hook

**File:** `scripts/hooks/pre-push-remove-secrets.sh`

```bash
#!/bin/bash
# Pre-push hook: Strip secrets from notebooks before push

echo "🔍 Scanning notebooks for secrets..."

# Find all .ipynb files
notebooks=$(git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$')

for notebook in $notebooks; do
    # Check for API key patterns
    if grep -qE 'API_KEY.*=.*"[^"]+"' "$notebook"; then
        echo "❌ Found secrets in $notebook"
        echo "   Stripping secrets..."
        
        # Replace with empty strings
        sed -i.bak -E 's/(API_KEY.*=.*")([^"]+)(")/\1\3/g' "$notebook"
        sed -i.bak -E 's/(SECRET.*=.*")([^"]+)(")/\1\3/g' "$notebook"
        
        # Stage the cleaned file
        git add "$notebook"
        echo "✅ Cleaned $notebook"
    fi
done

echo "✅ Secret scan complete"
```

**File:** `scripts/hooks/pre-push-remove-secrets.ps1`

```powershell
# Pre-push hook: Strip secrets from notebooks before push (Windows)

Write-Host "🔍 Scanning notebooks for secrets..." -ForegroundColor Cyan

$notebooks = git diff --cached --name-only --diff-filter=ACM | Where-Object { $_ -match '\.ipynb$' }

foreach ($notebook in $notebooks) {
    $content = Get-Content $notebook -Raw
    
    if ($content -match 'API_KEY.*=.*"[^"]+"') {
        Write-Host "❌ Found secrets in $notebook" -ForegroundColor Red
        Write-Host "   Stripping secrets..." -ForegroundColor Yellow
        
        # Replace with empty strings
        $content = $content -replace '(API_KEY.*=.*")([^"]+)(")', '$1$3'
        $content = $content -replace '(SECRET.*=.*")([^"]+)(")', '$1$3'
        
        Set-Content $notebook -Value $content
        git add $notebook
        
        Write-Host "✅ Cleaned $notebook" -ForegroundColor Green
    }
}

Write-Host "✅ Secret scan complete" -ForegroundColor Green
```

### Installation Task

- [ ] **Update install-hooks scripts**
  - Add pre-push hook installation to `scripts/install-hooks.sh`
  - Add pre-push hook installation to `scripts/install-hooks.ps1`
  - Test on both Windows and Unix

---

## Dependency Management

### Root Setup Script Updates

**File:** `scripts/setup.ps1` (Windows)

```powershell
# Add AI Infrastructure (ML Experiment Tracking) dependencies section
Write-Host "📦 Installing AI Infrastructure dependencies..." -ForegroundColor Cyan

pip install mlflow evidently dvc tensorboard wandb

Write-Host "✅ AI Infrastructure setup complete" -ForegroundColor Green
Write-Host "   Verify: mlflow --version && dvc --version" -ForegroundColor Cyan
```

**File:** `scripts/setup.sh` (Unix)

```bash
# Add AI Infrastructure (ML Experiment Tracking) dependencies section
echo "📦 Installing AI Infrastructure dependencies..."

pip install mlflow evidently dvc tensorboard wandb

echo "✅ AI Infrastructure setup complete"
echo "   Verify: mlflow --version && dvc --version"
```

### Tasks

- [ ] **Update `scripts/setup.ps1`**
  - Add AI Infrastructure pip packages (mlflow, evidently, dvc, tensorboard, wandb)
  - Update Jupyter kernel registration for AI Infrastructure environment

- [ ] **Update `scripts/setup.sh`**
  - Add AI Infrastructure pip packages
  - Test on Linux/macOS

- [ ] **Update `requirements.txt`** (if exists)
  - Add AI Infrastructure packages with version pins
  - Separate section for Ch.9-10 dependencies

---

**Note:** For Docker, Kubernetes, CI/CD, and monitoring dependencies, see `notes/devops_fundamentals/plan.md`

---

## Timeline

### Phase 1: Infrastructure (Week 1)
- [ ] Install dependencies (scripts/setup)
- [ ] Create secrets removal hook
- [ ] Set up MLflow demo environment (local tracking server)

### Phase 2: Ch.9 Content (Week 2-3)
- [ ] Write Ch.9 README.md (Experiment Tracking & Model Registry)
- [ ] Create Ch.9 notebook (local)
- [ ] Create Ch.9 notebook_supplement (Azure)
- [ ] Generate diagrams and animations

### Phase 3: Ch.10 Content (Week 4)
- [ ] Write Ch.10 README.md (Production Monitoring & A/B Testing)
- [ ] Create Ch.10 notebook (local)
- [ ] Create Ch.10 notebook_supplement (Azure)
- [ ] Generate diagrams and animations

### Phase 4: Backfill Supplements (Week 5)
- [ ] Ch.1-5 Azure supplements
- [ ] Test all notebooks with dummy Azure credentials
- [ ] Verify secret removal hook on all supplements

### Phase 5: Ch.6-8 Planning (Week 6)
- [ ] Outline Ch.6 (Model Serving Frameworks)
- [ ] Outline Ch.7 (AI-Specific Networking)
- [ ] Outline Ch.8 (Feature Stores)

---

## Success Criteria

- ✅ All Ch.9-10 notebooks run 100% locally (no cloud required)
- ✅ All supplements run on Azure with user-provided credentials
- ✅ Secrets hook prevents accidental key commits
- ✅ Setup scripts install all ML experiment tracking dependencies on Windows + Unix
- ✅ ML experiment tracking taught with $0 cloud spend (MLflow local)
- ✅ Focus remains on AI/ML-specific infrastructure (not generic DevOps)

---

## Notes

- **AI/ML-specific focus:** This track covers GPU optimization, model serving, experiment tracking, feature stores (not Docker/K8s basics)
- **DevOps prerequisite:** Students should complete `notes/devops_fundamentals/` for Docker, Kubernetes, CI/CD, monitoring
- **Local-first philosophy:** Every concept must work on localhost before showing cloud alternative
- **Zero-cost teaching:** Free tier or local tools only; paid subscriptions optional for Azure supplements
- **Security by default:** Secrets hook is mandatory; supplements must have empty credential strings in repo
- **Cross-platform:** All scripts must work on Windows (PowerShell) and Unix (Bash)

---

## Questions / Blockers

- [ ] Should Ch.6 (Model Serving) include TorchServe, or focus only on vLLM + ONNX Runtime?
- [ ] Should Ch.8 (Feature Stores) use Feast (local) or also show Azure ML Feature Store?
- [ ] Do we need a separate chapter on batch inference optimization, or integrate into Ch.6?
- [ ] Should Ch.10 include retraining triggers, or is that a separate topic?
