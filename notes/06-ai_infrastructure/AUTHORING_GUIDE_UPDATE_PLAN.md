# AI Infrastructure Track Authoring Guide Update — Implementation Plan

**Target:** `notes/06-ai_infrastructure/authoring-guide.md`  
**Effort:** 2 hours  
**LLM Calls:** 1

---

## Quick Context

Add workflow pattern for deployment chapters (ch08 feature stores, ch11 end-to-end deployment). Track uses Docker/K8s/Feast but lacks workflow organization guidance.

---

## Single Addition Required

**Location:** After chapter template section

**Content:** Workflow pattern for deployment/infrastructure chapters

**Procedural Chapters:**
- ch08 Feature Stores: Setup → Configure → Deploy → Query
- ch11 End-to-End Deployment: Dockerize → Orchestrate → Monitor

---

## Implementation

**Call 1:** Insert workflow pattern section

**Content template:**
```markdown
## Workflow-Based Chapter Pattern (Deployment Chapters)

### When to Use
- Multi-step deployment procedures
- Infrastructure setup sequences
- Feature store configuration workflows

### Standard Infrastructure Workflow
1. Local Setup (Docker/docker-compose)
2. Configuration (environment, secrets, resources)
3. Deployment (Kubernetes/cloud)
4. Validation (health checks, metrics)
5. Monitoring (Prometheus, Grafana, alerts)

### Industry Tools Integration
- Feature Stores: Feast (primary), Tecton (mention)
- Orchestration: Kubernetes (required), Docker Compose (local dev)
- Monitoring: Prometheus + Grafana (required)
- Serving: FastAPI (primary), Flask (alternative)

### Decision Checkpoints
**Deployment target:** Local (docker-compose) vs Cloud (K8s)
**Storage:** Redis (fast) vs PostgreSQL (persistent) vs S3 (offline)
**Scaling:** Vertical (bigger pods) vs Horizontal (more replicas)

### Code Pattern
```yaml
# kubernetes/deployment.yaml with annotations
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3  # Horizontal scaling
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: api
        image: model-api:latest
        resources:
          requests:
            memory: "256Mi"  # Why these values?
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```
```

---

## TODO: Notebook Audit

- [ ] ch08: Show Feast feature store setup → query → serve
- [ ] ch11: Show Docker → K8s → monitoring full pipeline
- [ ] All deployment chapters: Add decision checkpoints for local vs cloud

---

## Success Criteria

1. ✅ Workflow pattern for infrastructure chapters
2. ✅ Decision checkpoints for deployment targets
3. ✅ Industry tools (Feast, K8s, Prometheus) integration pattern
4. ✅ YAML/config file annotation pattern

**Files Modified:** 1 (`authoring-guide.md`)
