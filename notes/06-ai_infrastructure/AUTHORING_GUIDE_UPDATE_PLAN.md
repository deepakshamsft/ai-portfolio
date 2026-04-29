# AI Infrastructure Track Authoring Guide Update — Implementation Plan

**Target:** `notes/06-ai_infrastructure/authoring-guide.md`  
**Effort:** 4-6 hours (expanded from ML pattern adaptation)  
**LLM Calls:** 3-4

---

## Context & Rationale

The ML track has developed sophisticated patterns for workflow-based chapters (procedural content like feature engineering, diagnostics, tuning). The AI Infrastructure track shares similar characteristics — many chapters are **operational workflows** (deployment pipelines, optimization procedures, monitoring setups) rather than pure concept introductions.

**Source patterns** (from `notes/01-ml/authoring-guide.md`):
1. Workflow-Based Chapter Pattern (when to use, template, decision checkpoints)
2. Code Snippet Guidelines (4 rules: executable, progressive, decision-annotated, copy-paste ready)
3. Industry Tools Integration (manual → production pattern)
4. Notebook Exercise Pattern (industry callouts, decision logic templates)

**Adaptation principle:** Infrastructure workflows differ from ML feature engineering — they involve **system configuration, performance profiling, and deployment decisions** rather than data transformations. This plan adapts patterns to infrastructure contexts while preserving the track's focus on **latency, throughput, cost, and reliability**.

---

## Part 1: Workflow-Based Infrastructure Chapter Pattern

### Identifying Workflow-Based Chapters in AI Infrastructure

A chapter is workflow-based if:
- ✅ It teaches a **sequence of operational decisions** (not just a technology overview)
- ✅ Practitioner asks "what should I check next?" or "which option fits my constraints?"
- ✅ Multiple tools/techniques chosen based on workload characteristics or performance metrics
- ✅ The chapter reads like a troubleshooting/optimization guide, not a concept introduction

**Infrastructure-specific workflow chapters:**

| Chapter | Workflow Type | Decision Sequence | Priority |
|---------|---------------|-------------------|----------|
| **ch05 Inference Optimization** | Performance tuning | Profile → Identify Bottleneck → Apply Optimization → Validate | **HIGH** |
| **ch06 Model Serving Frameworks** | Tool selection + deployment | Evaluate Constraints → Select Framework → Configure → Load Test | **HIGH** |
| **ch08 Feature Stores** | Infrastructure setup | Design Schema → Setup Store → Configure Sync → Query + Serve | **MEDIUM** |
| **ch09 ML Experiment Tracking** | Instrumentation workflow | Instrument Code → Log Artifacts → Compare Runs → Organize | **MEDIUM** |
| **ch10 Production ML Monitoring** | Observability pipeline | Instrument Metrics → Configure Alerts → Diagnose Drift → Mitigate | **HIGH** |
| **ch11 End-to-End Deployment** | Full deployment pipeline | Dockerize → Orchestrate (K8s) → Monitor → Scale | **CRITICAL** |

**Concept-based chapters (no workflow structure needed):**
- ch01 GPU Architecture — Hardware fundamentals
- ch02 Memory and Compute Budgets — Capacity planning theory
- ch03 Quantization and Precision — Mathematical foundations
- ch04 Parallelism and Distributed Training — Parallelism strategies overview
- ch07 AI-Specific Networking — Network topology concepts

### Modified Workflow Template for Infrastructure Chapters

```markdown
# Ch.N — [Topic Name]

[Same header: story, curriculum context, notation]

---

## 0 · The Challenge — Where We Are
[Infrastructure-specific challenge: latency spike, deployment complexity, monitoring gap]

## 1 · Core Idea
[Brief overview of the workflow purpose — 2-3 sentences]

## 1.5 · The Infrastructure Workflow — Your N-Phase Diagnostic

**Before diving into configuration details, understand the operational workflow you'll follow for every deployment:**

> 🏗️ **What you'll build by the end:** [Description of deployed system/dashboard/monitoring setup]

```
Phase 1: [ACTION]           Phase 2: [ACTION]           Phase 3: [ACTION]
──────────────────────────────────────────────────────────────────────────
[What you do]               [What you do]               [What you do]
[What you measure]          [What you measure]          [What you measure]

→ DECISION:                 → DECISION:                 → DECISION:
  [Choice criteria]           [Choice criteria]           [Choice criteria]
  [Metric thresholds]         [Metric thresholds]         [Metric thresholds]
```

**The workflow maps to this chapter:**
- **Phase 1 ([ACTION])** → §X Section Name
- **Phase 2 ([ACTION])** → §Y Section Name  
- **Phase 3 ([ACTION])** → §Z Section Name

> 💡 **Usage note:** [Brief note on phase dependencies, rollback strategies, or execution order]

---

## 2 · Running Example
[Infrastructure-specific example: deploying a vision model API, optimizing LLM inference, monitoring recommendation service]

## 3 · Implementation Details
[Configuration/code sections organized by phase]

### 3.X · [Phase Name] **[Phase N: ACTION]**

[Section content with phase marker in header]

[Code/config snippet showing phase implementation]

```python
# Phase N: [Brief description]
# Example: Profile inference latency to identify bottleneck

import time
import torch

model.eval()
latencies = []

with torch.no_grad():
    for batch in test_loader:
        start = time.perf_counter()
        output = model(batch)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        
        # DECISION LOGIC (inline annotation)
        if latency > 100:
            print(f"❌ SLOW - Batch latency: {latency:.1f}ms (target: <100ms)")
        elif latency > 50:
            print(f"⚠️ MODERATE - Batch latency: {latency:.1f}ms")
        else:
            print(f"✅ FAST - Batch latency: {latency:.1f}ms")

avg_latency = sum(latencies) / len(latencies)
p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
print(f"\nAverage: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")
```

> 💡 **Industry Standard:** `torch.profiler` or `py-spy`
> ```python
> from torch.profiler import profile, ProfilerActivity
> 
> with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
>     output = model(batch)
> 
> print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
> ```
> **When to use:** Always in production for detailed profiling. Manual timing shown above for learning only.
> **Common alternatives:** `cProfile` (Python-level), `nvprof` (GPU-level), `tensorboard-profiler` (visualization)
> **See also:** [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### 3.X.1 DECISION CHECKPOINT — Phase N Complete

**What you just measured:**
- Average latency: 127ms (target: <100ms)
- P95 latency: 203ms (unacceptable for real-time)
- GPU utilization: 34% (underutilized — CPU bottleneck?)

**What it means:**
- Latency exceeds SLA (100ms) by 27% → Users experience lag
- Low GPU utilization + high latency → Likely CPU preprocessing bottleneck or small batch size
- High P95/avg ratio (1.6x) → Inconsistent performance, possibly GC pauses or context switching

**What to do next:**
→ **Increase batch size:** Try batch_size=16→32 (may reduce per-sample overhead)
→ **Profile preprocessing:** Check if data loading/transforms dominate (use `cProfile`)
→ **Enable TorchScript:** JIT compile model to reduce Python overhead
→ **For our scenario:** Increase batch size to 32 first (simplest, often 2-3x speedup)

---

[Repeat pattern for all phases]

## N-1 · Putting It Together — The Complete Deployment Flow

[Mermaid flowchart showing all phases integrated with decision branches and rollback paths]

## N · Progress Check — What We Can Deploy Now
[Infrastructure-specific: what system is now production-ready, what metrics to monitor]

## N+1 · Bridge to the Next Chapter
[How this deployment/optimization feeds into next infrastructure concern]
```

### Key Differences: ML Workflow vs Infrastructure Workflow

| Element | ML Workflow (Feature Engineering) | Infrastructure Workflow (Deployment) |
|---------|-----------------------------------|-------------------------------------|
| **Phases** | Data-centric (inspect, transform, validate) | System-centric (profile, configure, deploy, monitor) |
| **Decision criteria** | Statistical thresholds (skew, VIF, correlation) | Performance metrics (latency, throughput, utilization) |
| **Code artifacts** | Python snippets (pandas, sklearn) | Config files (YAML, Dockerfiles, API code) |
| **Validation** | Model accuracy improvement | SLA compliance (latency < 100ms, uptime > 99.9%) |
| **Industry tools** | sklearn, pandas | Docker, K8s, Prometheus, FastAPI, TorchServe |
| **Rollback strategy** | N/A (data transformations are reversible) | Critical (deployment rollback, traffic shifting) |

---

## Part 2: Infrastructure-Specific Code Snippet Guidelines

Adapt ML's 4 code snippet rules to infrastructure contexts (configs, deployments, APIs):

### Rule 1: Each phase ends with executable code/config showing that phase's workflow

**ML example (data transformation):**
```python
for col in numeric_cols:
    if abs(df[col].skew()) > 1.0:
        print(f"{col}: Apply log transform")
```

**Infrastructure adaptation (deployment configuration):**
```yaml
# kubernetes/deployment.yaml
# Phase 2: Configure resource requests based on profiling

apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3  # DECISION: Start with 3 for 99.9% availability
  template:
    spec:
      containers:
      - name: api
        image: model-api:v1.2.0
        resources:
          requests:
            memory: "2Gi"    # DECISION: Based on profiling (1.8Gi peak + 10% buffer)
            cpu: "1000m"     # DECISION: 1 CPU per replica (avg 70% utilization)
          limits:
            memory: "4Gi"    # DECISION: 2x requests (allow bursts, trigger OOM kill if exceeded)
            cpu: "2000m"     # DECISION: 2x requests (prevent CPU throttling under load)
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30  # DECISION: Model loading takes ~20s
          periodSeconds: 10
```

**Annotation pattern:** Inline comments explain **why these values** (not just what they are).

### Rule 2: Decision logic appears in code/config comments, not just prose

**ML example (threshold-based branching):**
```python
if vif > 10:
    verdict = "❌ SEVERE - Drop feature"
```

**Infrastructure adaptation (latency-based optimization):**
```python
# Phase 1: Latency profiling with decision logic

def profile_and_recommend(model, test_loader, target_latency_ms=100):
    """Profile model inference and recommend optimization strategy."""
    
    latencies = []
    for batch in test_loader:
        start = time.perf_counter()
        _ = model(batch)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    
    # DECISION LOGIC: Optimization strategy selection
    if avg > target_latency_ms * 2:
        recommendation = "❌ CRITICAL - Apply quantization (INT8) + batch size tuning"
        priority = "P0"
    elif avg > target_latency_ms:
        recommendation = "⚠️ HIGH - Try TorchScript compilation + increase batch size"
        priority = "P1"
    elif p95 > target_latency_ms * 1.5:
        recommendation = "⚡ MODERATE - P95 high (inconsistent) — profile for outliers"
        priority = "P2"
    else:
        recommendation = "✅ MEETS SLA - Monitor for regressions"
        priority = "P3"
    
    return {
        "avg_latency_ms": avg,
        "p95_latency_ms": p95,
        "target_latency_ms": target_latency_ms,
        "recommendation": recommendation,
        "priority": priority
    }
```

### Rule 3: Code/configs should be copy-paste executable in infrastructure context

**Requirements:**
- Include all necessary imports (for Python scripts)
- Include all necessary fields (for YAML/JSON configs)
- Use realistic values (not `your-image-here` or `TODO: fill this`)
- Include validation/health checks where applicable
- Document prerequisites (e.g., "Requires model.pth in current directory")

**Example (FastAPI serving endpoint):**
```python
# api.py - Copy-paste ready model serving API
# Prerequisites: pip install fastapi uvicorn torch pillow
# Run: uvicorn api:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import io
import time

app = FastAPI(title="ResNet50 Classification API")

# Load model at startup (not per-request)
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify uploaded image and return top-5 predictions with latency."""
    
    start = time.perf_counter()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Load and preprocess image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    # ... (full preprocessing code here)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # DECISION LOGIC: Latency alerting
    if latency_ms > 100:
        print(f"⚠️ SLOW REQUEST - Latency: {latency_ms:.1f}ms (target: <100ms)")
    
    return {
        "predictions": top5_classes,
        "latency_ms": round(latency_ms, 1)
    }

@app.get("/health")
async def health():
    """Kubernetes liveness probe endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}
```

### Rule 4: Show progressive building, not isolated snippets

**ML example (references earlier data prep):**
```python
# Using X_train from Phase 1 inspection...
X_scaled = StandardScaler().fit_transform(X_train)
```

**Infrastructure adaptation (references earlier profiling):**
```python
# Phase 3: Apply optimization based on Phase 1 profiling results

# From Phase 1: avg_latency=127ms, recommendation="Try TorchScript + batch size increase"

# Step 3.1: Increase batch size (from profiling recommendation)
batch_size = 32  # Increased from 16 based on Phase 1 analysis

# Step 3.2: Apply TorchScript compilation
model_scripted = torch.jit.script(model)
model_scripted.save("model_scripted.pt")

# Step 3.3: Re-profile with optimizations applied
optimized_results = profile_and_recommend(model_scripted, test_loader_32)
print(f"Improvement: {127 - optimized_results['avg_latency_ms']:.1f}ms reduction")
```

---

## Part 3: Infrastructure Decision Checkpoint Format

Adapt ML's 3-part checkpoint structure to infrastructure performance/deployment decisions:

```markdown
### N.M DECISION CHECKPOINT — Phase K Complete

**What you just measured:**
- [Metric 1: specific number with units — latency, throughput, utilization, error rate]
- [Metric 2: specific number with units]
- [Metric 3: comparison to target/SLA]

**What it means:**
- [Interpretation: translate metrics into operational insight]
- [Impact: why this matters for reliability/cost/user experience]
- [Root cause hypothesis: likely bottleneck or constraint]

**What to do next:**
→ **Option 1 ([Optimization/Config Name]):** [Specific action with parameters/settings]
   - **Expected improvement:** [Quantified impact — "2-3x throughput", "50% latency reduction"]
   - **Risk:** [Potential downside — "Increased memory usage", "Requires redeployment"]
→ **Option 2 ([Alternative]):** [Alternative action]
   - **Expected improvement:** [Quantified impact]
   - **Risk:** [Potential downside]
→ **For [our scenario]:** Choose [option] because [reasoning based on constraints]
```

**Checkpoint placement (infrastructure-specific):**
- After profiling/measurement phase (before optimization)
- After configuration phase (before deployment)
- After deployment phase (before monitoring/scaling)
- After optimization phase (validate improvement)

**Example (Inference Optimization chapter):**

```markdown
### 5.2 DECISION CHECKPOINT — Profiling Complete

**What you just measured:**
- Average latency: 127ms per request (target: <100ms, 27% over SLA)
- P95 latency: 203ms (2x average — high variance)
- GPU utilization: 34% (underutilized — suggests CPU bottleneck)
- Requests/sec: 7.8 (target: 10+ for current traffic)

**What it means:**
- Missing SLA by 27% → 1 in 4 requests feels sluggish to users
- Low GPU util + high latency → CPU preprocessing is bottleneck (not model forward pass)
- High P95/avg ratio → Inconsistent performance, likely due to Python GIL or small batch size
- Throughput shortfall → Will need horizontal scaling OR optimization to handle growth

**What to do next:**
→ **Increase batch size (16→32):** Process more samples per forward pass
   - **Expected improvement:** 40-60% latency reduction (amortize overhead)
   - **Risk:** Increased memory usage (monitor for OOM), higher P95 if batches timeout
→ **Apply TorchScript JIT compilation:** Eliminate Python interpreter overhead
   - **Expected improvement:** 20-30% latency reduction (CPU-bound workloads)
   - **Risk:** Longer startup time (compilation), debugging harder (traced graphs)
→ **INT8 Quantization:** Reduce model size and memory bandwidth bottleneck
   - **Expected improvement:** 2-4x throughput, 50% memory reduction
   - **Risk:** 1-2% accuracy drop (validate on test set), requires calibration dataset
→ **For our scenario (ResNet50 classification):** Start with batch size increase to 32 (simplest, often biggest win). If still short of SLA, add TorchScript. Reserve quantization for next iteration.
```

---

## Part 4: Industry Tools Integration (Infrastructure-Specific)

Adapt ML's "manual implementation → industry standard" pattern to infrastructure tools.

**Core principle (same as ML):** Show manual/basic implementation first (build intuition about **what the system does**), then show industry-standard tool/framework.

### Required Callout Box Pattern (Infrastructure Adaptation)

```markdown
> 💡 **Industry Standard:** `Tool/Framework Name`
> 
> ```python
> # Production-ready one-liner/config
> from library import Tool
> result = Tool.configure(params).deploy()
> ```
> OR
> ```yaml
> # kubernetes/config.yaml - Production configuration
> apiVersion: v1
> kind: Service
> ...
> ```
> 
> **When to use:** [Always in production | For teams >5 | When [condition]]
> **Why it's better:** [Specific advantages — "Built-in HA", "Auto-scaling", "Managed upgrades"]
> **Common alternatives:** [Alternative 1] (tradeoff), [Alternative 2] (tradeoff)
> **See also:** [Official docs link]
```

### Infrastructure Industry Tools by Chapter

| Chapter | Manual/Basic Approach | Industry Standard Tool | When to Use Standard |
|---------|----------------------|------------------------|---------------------|
| **ch05 Inference Optimization** | Manual timing loops | `torch.profiler`, `py-spy`, TensorRT | Always for production profiling |
| **ch06 Model Serving** | Flask/FastAPI from scratch | **TorchServe**, BentoML, Ray Serve, Triton | Teams deploying >3 models |
| **ch08 Feature Stores** | Manual SQL queries + cache | **Feast**, Tecton, Hopsworks | When features reused across >2 models |
| **ch09 Experiment Tracking** | Manual CSV logging | **MLflow**, Weights & Biases, Neptune | Always (free tier sufficient) |
| **ch10 Monitoring** | Manual metric collection | **Prometheus + Grafana**, Datadog | Always for production systems |
| **ch11 Deployment** | Manual docker run + scripts | **Kubernetes**, Docker Swarm, ECS | When scaling >5 nodes or needing HA |

**Example callout (Model Serving chapter):**

```markdown
> 💡 **Industry Standard:** TorchServe
> 
> ```bash
> # Package model for TorchServe
> torch-model-archiver --model-name resnet50 \
>   --version 1.0 \
>   --model-file model.py \
>   --serialized-file resnet50.pth \
>   --handler image_classifier
> 
> # Serve with autoscaling and metrics
> torchserve --start --model-store model_store --models resnet50=resnet50.mar \
>   --ts-config config.properties
> ```
> 
> **When to use:** Production deployments requiring autoscaling, A/B testing, or multi-model serving
> **Why it's better:**
> - Built-in metrics (latency, throughput) → Prometheus export
> - Autoscaling based on queue depth or latency
> - Model versioning + zero-downtime updates
> - Batching + GPU optimization out-of-box
> 
> **Common alternatives:**
> - **BentoML:** Python-native, easier for ML engineers, excellent local dev experience
> - **Ray Serve:** Best for multi-model pipelines, distributed inference
> - **Triton Inference Server:** NVIDIA's solution, optimal for multi-framework (TF/PyTorch/ONNX)
> 
> **See also:** [TorchServe Docs](https://pytorch.org/serve/), [Performance Guide](https://pytorch.org/serve/performance_guide.html)
```

---

## Part 5: Notebook/Script Exercise Pattern (Infrastructure)

Adapt ML's notebook enhancement pattern to infrastructure exercises (deployment configs, load testing, monitoring setup).

### Exercise Format Differences: ML vs Infrastructure

| Aspect | ML Track | Infrastructure Track |
|--------|----------|---------------------|
| **Primary artifacts** | Jupyter notebooks (.ipynb) | Python scripts + config files (YAML, Dockerfile) |
| **Exercise goal** | Implement data transformation → train model | Configure deployment → validate performance |
| **Manual approach** | Feature engineering from scratch | Manual docker run + curl testing |
| **Industry approach** | sklearn transformers | Docker Compose, Kubernetes, TorchServe |
| **Success criteria** | Model accuracy improvement | Latency < SLA, uptime > 99%, cost < budget |

### Required Enhancements for Infrastructure Exercises

#### 1. Industry Standard Callout Boxes (Config Files)

**Pattern:** Show manual deployment first, then show production-ready orchestration.

```markdown
> 💡 **Industry Standard Pattern:** After testing locally with `docker run`, deploy to Kubernetes:
> 
> ```yaml
> # kubernetes/deployment.yaml
> apiVersion: apps/v1
> kind: Deployment
> metadata:
>   name: model-server
> spec:
>   replicas: 3
>   template:
>     spec:
>       containers:
>       - name: api
>         image: model-api:v1.0
>         resources:
>           requests:
>             memory: "2Gi"
>             cpu: "1000m"
>         livenessProbe:
>           httpGet:
>             path: /health
>             port: 8000
> ```
> 
> **When to use:** Production deployments requiring HA (3+ replicas) and autoscaling
> **Why it's better:** Automatic failover, rolling updates, resource management
> **Common alternatives:** Docker Compose (local dev), AWS ECS (managed, less config)
```

**Frequency:** 2-4 callouts per exercise (deployment config, monitoring setup, load testing, optimization)

#### 2. Decision Logic Templates (Performance-Based)

```markdown
**Decision Logic Template: Latency-Based Optimization**

When you profile your deployment, include threshold-based recommendations:

\```python
def profile_and_decide(api_url, num_requests=100):
    """Profile API latency and recommend next optimization."""
    
    latencies = []
    for _ in range(num_requests):
        start = time.time()
        response = requests.post(f"{api_url}/predict", files={"file": image_bytes})
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
    
    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    
    # DECISION LOGIC (add threshold-based branching)
    if avg > 200:
        action = "❌ CRITICAL - Apply INT8 quantization + increase batch size"
    elif avg > 100:
        action = "⚠️ HIGH - Enable TorchScript JIT compilation"
    elif p95 > avg * 1.5:
        action = "⚡ MODERATE - Investigate P95 outliers (GC pauses?)"
    else:
        action = "✅ MEETS SLA - Monitor for regressions"
    
    print(f"Avg: {avg:.1f}ms | P95: {p95:.1f}ms | Action: {action}")
\```

**Thresholds:**
- `avg > 200ms` → Critical (2x SLA) — Aggressive optimization needed
- `avg > 100ms` → Warning (above SLA) — Incremental optimization
- `p95 > 1.5*avg` → Inconsistent performance — Profile for outliers
```

**Frequency:** 2-3 templates per exercise (latency, cost, resource utilization)

#### 3. Visual Indicators (Infrastructure Metrics)

| Indicator | Meaning | Infrastructure Use Case |
|-----------|---------|-------------------------|
| ✅ | Healthy/Meets SLA | Latency < 100ms, uptime > 99.9%, cost < budget |
| ⚠️ | Warning/Monitor | Latency 100-150ms, uptime 99-99.9%, cost near budget |
| ⚡ | Action Needed | Latency 150-200ms, uptime 95-99%, cost over budget |
| ❌ | Critical/Failing | Latency > 200ms, uptime < 95%, cost >> budget |
| 💡 | Industry standard | Production tool/pattern |
| 🏗️ | Infrastructure note | Deployment/config consideration |
| 📊 | Metric/measurement | Performance data point |

### Implementation Checklist for Infrastructure Exercises

When creating/updating exercise scripts/notebooks for infrastructure chapters:

- [ ] **Industry callouts added** (2-4 locations: Docker → K8s, manual → TorchServe, local → cloud)
- [ ] **Decision logic templates added** (2-3 locations: latency thresholds, cost budgets, scale-up triggers)
- [ ] **Visual indicators consistent** (✅ ❌ ⚠️ ⚡ 💡 🏗️ used appropriately)
- [ ] **Thresholds documented** (specific numbers — "<100ms latency", ">99.9% uptime", not vague "fast")
- [ ] **Config files included** (Dockerfile, docker-compose.yml, kubernetes/*.yaml as needed)
- [ ] **Prerequisites documented** ("Requires Docker 20+, kubectl configured, model.pth in ./models/")
- [ ] **Validation commands included** (`curl` health checks, load testing commands, metrics queries)
- [ ] **Rollback instructions included** (for deployment chapters — how to revert if deployment fails)

### Anti-Patterns to Avoid (Infrastructure Context)

❌ **Don't:**
- Show only Kubernetes YAML without explaining local Docker testing first
- Use placeholder configs (`image: your-image-here`) — use realistic values
- Omit health checks and resource limits from K8s configs
- Show industry tools without explaining **when to use** them (not everyone needs K8s)
- Include configs that won't work copy-paste (missing required fields, wrong API versions)

✅ **Do:**
- Show progression: local testing → Docker → Docker Compose → Kubernetes
- Include full working configs with realistic values (memory: "2Gi", not "XXXMi")
- Explain tradeoffs ("K8s adds complexity — use Docker Compose if < 5 nodes")
- Document prerequisites clearly ("Requires model file at ./models/resnet50.pth")
- Include validation steps ("Test with: `curl http://localhost:8000/health`")

### Example: Ch11 End-to-End Deployment Exercise

**Structure:**
1. **Phase 1:** Dockerize FastAPI model server (manual `docker run`)
2. **Phase 2:** Multi-container setup with docker-compose (app + Redis + monitoring)
3. **Phase 3:** Kubernetes deployment (3 replicas, autoscaling, ingress)
4. **Phase 4:** Load testing + monitoring (Prometheus metrics, Grafana dashboard)

**Enhancements:**
- **4 industry callouts:**
  - After Dockerfile: "In production, use multi-stage builds to reduce image size (500MB → 200MB)"
  - After docker-compose: "For 5+ services, migrate to Kubernetes for better orchestration"
  - After K8s deployment: "Add Horizontal Pod Autoscaler (HPA) for traffic spikes"
  - After load testing: "Use k6 or Locust (not just curl) for realistic load patterns"
- **3 decision logic templates:**
  - Latency-based optimization (shown above)
  - Cost-based scaling ("If cost > $500/month → right-size resources or use spot instances")
  - Replica count selection ("If p95 latency > 150ms OR CPU > 70% → scale replicas")
- **Specific thresholds:**
  - Latency SLA: p95 < 100ms
  - Availability SLA: > 99.9% (max 43 min downtime/month)
  - Resource utilization: 60-80% (not too low = wasted $, not too high = risk)

**Result:** 15-script exercise (Dockerfile, docker-compose.yml, 3 K8s YAMLs, 2 Python scripts, load test config, Prometheus config, Grafana dashboard JSON, README.md)

---

## Part 6: Infrastructure Track Grand Challenge Integration

### Track Grand Challenge Themes

**AI Infrastructure track focus:**
1. **Latency optimization** — Getting models to respond in <100ms
2. **Throughput scaling** — Serving 1K+ requests/sec cost-effectively
3. **Resource efficiency** — Minimizing GPU hours and cloud spend
4. **Reliability** — Achieving 99.9%+ uptime with graceful degradation

**Grand challenge (hypothetical running example across chapters):**
> "Deploy a vision model (ResNet50) as a production API serving 1,000 requests/sec with p95 latency <100ms, 99.9% uptime, at <$2,000/month cloud cost."

### How Workflow Patterns Support Grand Challenge

| Chapter | Workflow Contribution to Grand Challenge | Decision Checkpoint Metric |
|---------|------------------------------------------|---------------------------|
| ch05 Inference Optimization | Reduce latency from 200ms → <100ms via quantization/batching | Latency (ms), throughput (req/s) |
| ch06 Model Serving | Choose TorchServe for autoscaling + batching (handle traffic spikes) | Requests/sec capacity, failover time |
| ch08 Feature Stores | Precompute features to reduce online latency (200ms → 50ms) | Feature fetch latency, cache hit rate |
| ch09 Experiment Tracking | Track optimization experiments (which config hits latency SLA?) | Experiment comparison (latency, cost) |
| ch10 Monitoring | Detect latency regressions early (alert if p95 > 120ms) | Alert firing time, false positive rate |
| ch11 End-to-End | Integrate all optimizations into K8s deployment with autoscaling | End-to-end SLA compliance (%) |

**Grand challenge progress tracking** (add to each workflow chapter):

```markdown
## N · Progress Check — Grand Challenge Update

**Grand Challenge:** Deploy ResNet50 API @ 1,000 req/s, p95 < 100ms, 99.9% uptime, <$2K/month

**Before this chapter:**
- Latency: 200ms p95 (2x over SLA)
- Throughput: 50 req/s (20x short of target)
- Cost: N/A (not deployed)
- Uptime: N/A (not deployed)

**After this chapter (Inference Optimization):**
- ✅ Latency: **85ms p95** (15% under SLA) — INT8 quantization + batch size 32
- ✅ Throughput: **120 req/s** (2.4x improvement, still 8x short) — Need serving framework (Ch.6)
- ⚠️ Cost: Projected $3,200/month (1.6x over budget) — Need autoscaling (Ch.11)
- ❌ Uptime: Not deployed yet (Ch.11)

**What's next:** Ch.6 Model Serving — TorchServe autoscaling to handle 1,000 req/s bursts
```

---

## Part 7: Implementation Checklist

### Files to Modify

1. **`notes/06-ai_infrastructure/authoring-guide.md`** (main authoring guide)
   - Add "Workflow-Based Infrastructure Chapter Pattern" section (adapted from ML pattern)
   - Add "Infrastructure Code Snippet Guidelines" section (4 rules adapted)
   - Add "Decision Checkpoint Format" section (infrastructure metrics)
   - Add "Industry Tools Integration" section (TorchServe, K8s, Prometheus, etc.)
   - Add "Infrastructure Exercise Pattern" section (script + config exercises)
   - Add "Grand Challenge Progress Tracking" section (per-chapter SLA tracking)

2. **Individual chapter authoring (future work):**
   - ch05, ch06, ch08, ch09, ch10, ch11 (high-priority workflow chapters)
   - Add §1.5 Workflow overview diagrams
   - Add decision checkpoints after each phase
   - Add industry tool callouts (2-4 per chapter)
   - Add grand challenge progress tracking

### Success Criteria (for this plan)

✅ **Part 1 complete:** Workflow pattern documented with infrastructure-specific phases  
✅ **Part 2 complete:** Code snippet guidelines adapted (Python + YAML/Dockerfile)  
✅ **Part 3 complete:** Decision checkpoint format includes performance metrics (latency, throughput, cost)  
✅ **Part 4 complete:** Industry tools catalog (Docker → K8s, Flask → TorchServe, manual → Prometheus)  
✅ **Part 5 complete:** Exercise pattern adapted (scripts + configs, not just notebooks)  
✅ **Part 6 complete:** Grand challenge integration (per-chapter SLA tracking)  

---

## Appendix: Infrastructure Industry Tools Reference

Quick reference for industry standard tools to mention in callouts:

### Model Serving Frameworks
- **TorchServe** (PyTorch native, AWS-backed) — Primary recommendation
- **BentoML** (Python-first, ML engineer friendly) — Alternative for teams <10
- **Ray Serve** (multi-model pipelines, distributed) — For complex inference graphs
- **Triton Inference Server** (NVIDIA, multi-framework) — For TensorRT optimization
- **FastAPI** (manual, educational) — Show first, then migrate to TorchServe

### Orchestration
- **Kubernetes** (industry standard) — Required for production (>5 nodes)
- **Docker Compose** (local dev) — Show first for learning
- **AWS ECS/Fargate** (managed) — Alternative for AWS-native stacks

### Monitoring & Observability
- **Prometheus + Grafana** (open-source standard) — Always show this
- **Datadog** (SaaS, full-stack) — Mention for enterprises
- **New Relic, AppDynamics** (alternatives) — Brief mention

### Feature Stores
- **Feast** (open-source, flexible) — Primary recommendation
- **Tecton** (enterprise, managed Feast) — Mention for scale
- **Hopsworks** (full ML platform) — Brief mention

### Experiment Tracking
- **MLflow** (open-source, comprehensive) — Primary recommendation
- **Weights & Biases** (best UI/UX) — Alternative
- **Neptune.ai** (collaboration focus) — Brief mention

### Optimization Tools
- **TensorRT** (NVIDIA, GPU optimization) — For NVIDIA GPUs
- **ONNX Runtime** (cross-platform) — For portability
- **torch.jit** (TorchScript) — Show as first optimization
- **OpenVINO** (Intel) — For Intel hardware

### Load Testing
- **Locust** (Python, ML-friendly) — Primary for exercises
- **k6** (Go, performant) — Alternative
- **Apache JMeter** (Java, legacy) — Brief mention
