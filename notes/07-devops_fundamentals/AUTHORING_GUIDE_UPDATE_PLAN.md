# DevOps Track Authoring Guide Update — Implementation Plan

**Target:** `notes/07-devops_fundamentals/authoring-guide.md`  
**Effort:** 2 hours  
**LLM Calls:** 1

---

## Quick Context

DevOps track is 100% procedural (all chapters are workflows) but lacks explicit authoring guidance for workflow organization. All chapters already use industry tools (Docker, Kubernetes, Terraform).

---

## Single Addition Required

**Location:** After chapter template section

**Content:** Codify existing workflow patterns into explicit authoring rules

**Key difference from ML track:** 
- DevOps has NO concept-based chapters
- All chapters follow setup → configure → deploy → validate
- Already uses industry tools (no from-scratch implementations)

---

## Implementation

**Call 1:** Insert workflow codification section

**Content template:**
```markdown
## Workflow-Based Chapter Pattern (ALL DevOps Chapters)

### Standard DevOps Workflow Structure
All chapters follow: Setup → Configure → Deploy → Validate → Troubleshoot

### Decision Checkpoints
- **Local vs Cloud:** When to use Minikube vs cloud Kubernetes
- **Tool Selection:** Docker Compose vs Kubernetes vs serverless
- **Monitoring:** What metrics to track per deployment type

### Industry Tools (Required)
- Container: Docker (required), Podman (alternative)
- Orchestration: Kubernetes (required), Docker Swarm (mention)
- IaC: Terraform (required), Pulumi (alternative)
- CI/CD: GitHub Actions (primary), GitLab CI (alternative)
- Monitoring: Prometheus + Grafana (required)

### Code Pattern
Show tool configuration → explain options → execute → validate

Example:
```dockerfile
# Dockerfile best practices
FROM python:3.11-slim  # Why slim?
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # Why --no-cache-dir?
```
```

---

## TODO: Consistency Audit

Verify all 8 chapters follow:
- [ ] Setup phase with tool installation
- [ ] Configuration phase with decision checkpoints
- [ ] Deployment phase with validation
- [ ] Troubleshooting section with common issues
- [ ] Industry best practices highlighted

---

## Success Criteria

1. ✅ Workflow pattern explicitly documented (not just implied)
2. ✅ Decision checkpoints codified (local vs cloud, tool selection)
3. ✅ Industry tools list maintained
4. ✅ Troubleshooting pattern standardized

**Files Modified:** 1 (`authoring-guide.md`)
