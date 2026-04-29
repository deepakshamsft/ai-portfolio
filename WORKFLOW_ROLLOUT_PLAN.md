# Workflow Pattern Rollout Plan

**Created**: April 29, 2026  
**Status**: Planning Phase  
**Scope**: 40 workflow candidate chapters across 8 tracks (out of 129 total chapters audited)

---

## Executive Summary

### Current State
- ✅ **1 chapter complete**: ML Ch.3 Feature Importance (workflow pattern fully implemented)
- ✅ **8 authoring guides updated**: All tracks have workflow pattern documentation
- ❌ **40 chapters need workflow retrofit**: 31% of all chapters are workflow candidates but still use concept-based structure

### Audit Results by Track

| Track | Total Chapters | Workflow Candidates | Already Complete | Concept Chapters | Adherence % |
|-------|----------------|---------------------|------------------|------------------|-------------|
| **ML** | 61 | 7 (HIGH/MED) | 1 | 53 | **2%** |
| **Advanced DL** | 10 | 0 | 0 | 10 | **100%** ✅ |
| **AI** | 12 | 8 | 0 | 4 | **0%** |
| **Multi-Agent AI** | 7 | 5 | 0 | 2 | **0%** |
| **Multimodal AI** | 13 | 3 | 0 | 10 | **0%** |
| **AI Infrastructure** | 11 | 7 | 0 | 4 | **0%** |
| **DevOps** | 8 | 8 | 0 | 0 | **0%** |
| **Math** | 7 | 2 | 0 | 5 | **0%** |
| **TOTAL** | **129** | **40** | **1** | **88** | **2.5%** |

### Key Findings

1. **DevOps track is 100% workflow-based** - All 8 chapters describe procedures (Docker setup, K8s deployment, CI/CD pipelines)
2. **Advanced Deep Learning is 100% concept-based** - All 10 chapters correctly use concept structure (architectures like ResNet, YOLO, Mask R-CNN)
3. **AI track has highest workflow density** - 67% of chapters (8/12) are workflow candidates (RAG pipelines, evaluation, fine-tuning)
4. **Multimodal AI is mostly architecture-focused** - Only 3/13 chapters are workflow candidates (text-to-image, evaluation, deployment)
5. **ML track has only 7 candidates** - Most chapters correctly use concept structure for algorithms

---

## Workflow Candidates (40 Total)

### Track 1: ML (7 chapters)

#### HIGH Priority (5 chapters)
1. **ch00_data_prep** (01_regression) — MEDIUM complexity
   - Procedure: EDA → detect outliers → imputation → validate correlations → transform
   - 4 phases: Inspect, Audit, Transform, Validate
   
2. **ch08_data_validation** (01_regression) — MEDIUM complexity
   - Procedure: Schema contracts → drift monitoring (PSI/KS) → retraining triggers
   - 4 phases: Define, Monitor, Detect, Respond
   
3. **ch00b_class_imbalance** (01_regression) — LOW complexity
   - Procedure: Detect imbalance → choose strategy (SMOTE/weights) → tune thresholds → validate
   - 3 phases: Detect, Rebalance, Validate
   
4. **ch06_cold_start_production** (04_recommender_systems) — HIGH complexity
   - Procedure: Cold start detection → initialization → two-stage retrieval → A/B test
   - 5 phases: Detect, Initialize, Serve, Test, Cutover
   
5. **ch06_production** (05_anomaly_detection) — MEDIUM complexity
   - Procedure: Latency budget → drift detection → retraining → blue-green deployment
   - 4 phases: Deploy, Monitor, Detect, Retrain
   
6. **ch06_production** (08_ensemble_methods) — MEDIUM complexity
   - Procedure: Parallel inference → latency budget → drift monitoring → A/B testing
   - 4 phases: Deploy, Monitor, Validate, Cutover

#### LOW Priority (Borderline)
7. **ch08_tensorboard** (03_neural_networks) — LOW complexity
   - Procedure: Scalar logging → diagnose loss curves → histogram tracking → gradient checks
   - 4 phases: Instrument, Diagnose, Tune, Validate

---

### Track 2: Advanced Deep Learning (0 chapters)
✅ **All chapters correctly use concept-based structure** - No workflow retrofits needed

---

### Track 3: AI (8 chapters)

#### HIGH Complexity (6 chapters)
1. **Ch04 - RAG and Embeddings** 
   - Procedure: Ingestion (chunk → embed → store) + Query (embed → retrieve → generate)
   - 5 phases: Chunk, Embed, Store, Retrieve, Generate

2. **Ch06 - ReAct and Semantic Kernel**
   - Procedure: Agent development loop (Thought → Action → Observation)
   - 4 phases: Design, Implement, Test, Iterate

3. **Ch07 - Safety and Hallucination**
   - Procedure: Input validation → output validation → monitoring → red-team testing
   - 3 phases: Input Filters, Output Validation, Monitoring

4. **Ch08 - Evaluating AI Systems**
   - Procedure: Component eval → pipeline eval (RAGAS) → user eval (A/B)
   - 3 phases: Component, Pipeline, User

5. **Ch10 - Fine-Tuning**
   - Procedure: Decision tree (RAG vs FT) → dataset prep → LoRA training → eval
   - 4 phases: Decide, Prepare, Train, Validate

6. **Ch12 - Testing AI Systems**
   - Procedure: Unit tests → integration tests → property-based tests → CI/CD
   - 3 phases: Unit, Integration, Adversarial

#### MEDIUM Complexity (2 chapters)
7. **Ch02 - Prompt Engineering**
   - Procedure: System prompt → few-shot examples → structured output → CoT elicitation
   - 4 phases: System, Examples, Structure, Reasoning

8. **Ch09 - Cost and Latency**
   - Procedure: Model tier selection → prompt caching → streaming → batching/quantization
   - 4 phases: Select, Cache, Stream, Optimize

---

### Track 4: Multi-Agent AI (5 chapters)

#### HIGH Complexity (2 chapters)
1. **Ch04 - Event-Driven Agent Messaging**
   - Procedure: Choose message bus → define topics → implement pub/sub → idempotency → DLQ
   - 7 sub-phases across 4 major phases

2. **Ch06 - Trust, Sandboxing & Authentication**
   - Procedure: Mark trust boundaries → structured validation → HMAC signing → sandboxed execution → approval thresholds
   - 5 phases: Boundary, Validate, Auth, Isolate, Enforce

#### MEDIUM Complexity (3 chapters)
3. **Ch02 - Model Context Protocol (MCP)**
   - Procedure: Handshake → tool discovery → tool calling → transport handling
   - 4 phases: Initialize, Discover, Call, Handle

4. **Ch03 - Agent-to-Agent Protocol (A2A)**
   - Procedure: Publish agent card → submit task → track lifecycle → stream progress
   - 4 phases: Publish, Submit, Track, Stream

5. **Ch05 - Shared Memory & Blackboard**
   - Procedure: Design key schema → section-based writes → optimistic locking → event sourcing → TTL
   - 5 phases: Design, Write, Lock, Audit, Expire

---

### Track 5: Multimodal AI (3 chapters)

#### HIGH Complexity (1 chapter)
1. **ch08_text_to_image**
   - Procedure: Prompt engineering → negative prompts → tool selection (txt2img/img2img/inpainting) → ControlNet conditioning
   - 4 phases: Prompt, Negative, Select, Condition

#### MEDIUM Complexity (2 chapters)
2. **ch12_generative_evaluation**
   - Procedure: Identify metrics (fidelity/diversity/alignment) → select metrics (FID/CLIP/HPSv2) → compute → interpret
   - 4 phases: Identify, Select, Compute, Interpret

3. **ch13_local_diffusion_lab**
   - Procedure: Component selection → optimization tuning → integration testing → performance validation
   - 4 phases: Select, Optimize, Integrate, Validate

---

### Track 6: AI Infrastructure (7 chapters)

#### HIGH Complexity (4 chapters)
1. **Ch05 - Inference Optimization**
   - Procedure: Profile latency → identify bottleneck → apply optimization → benchmark → iterate
   - 4 phases: Profile, Identify, Optimize, Validate

2. **Ch08 - Feature Stores**
   - Procedure: Feature definition → offline store setup → online store setup → materialization → monitor
   - 5 phases: Define, Offline, Online, Materialize, Monitor

3. **Ch10 - Production ML Monitoring**
   - Procedure: Deploy monitoring → detect drift (data/prediction/performance) → root cause → A/B test fix → rollback/promote
   - 5 phases: Deploy, Detect, Triage, Remediate, Validate

4. **Ch11 - End-to-End Deployment**
   - Procedure: Docker image → K8s manifests → deploy → monitoring → validate → autoscaling
   - 6 phases: Containerize, K8s Setup, Deploy, Monitor, Validate, Scale

#### MEDIUM Complexity (3 chapters)
5. **Ch02 - Memory & Compute Budgets**
   - Procedure: Calculate parameter memory → KV cache → activations → check VRAM fit → adjust batch
   - 3 phases: Calculate, Check, Optimize

6. **Ch06 - Model Serving Frameworks**
   - Procedure: Define requirements → benchmark candidates (vLLM/ONNX/TensorRT) → compare → deploy
   - 4 phases: Requirements, Benchmark, Compare, Deploy

7. **Ch09 - ML Experiment Tracking**
   - Procedure: Setup tracking → log experiments → compare runs → select best → register model
   - 4 phases: Setup, Track, Compare, Promote

---

### Track 7: DevOps Fundamentals (8 chapters - 100%)

#### HIGH Complexity (6 chapters)
1. **Ch01 - Docker Fundamentals**
   - Procedure: Write Dockerfile → build → run → debug (logs/exec) → persist (volumes)
   - 4 phases: Build, Run, Debug, Persist

2. **Ch02 - Container Orchestration (Compose)**
   - Procedure: Write compose.yml → define services → networks → health checks → dependencies
   - 4 phases: Define, Network, Persist, Dependencies

3. **Ch03 - Kubernetes Basics**
   - Procedure: Create cluster → write deployment → apply → expose service → validate self-healing → scale
   - 4 phases: Cluster, Deploy, Expose, Resilience

4. **Ch04 - CI/CD Pipelines**
   - Procedure: Define triggers → test job → build Docker → push → deploy to K8s → verify
   - 4 phases: Test, Build, Push, Deploy

5. **Ch05 - Monitoring & Observability**
   - Procedure: Instrument app → configure Prometheus → Grafana dashboards → alert rules → validate
   - 5 phases: Instrument, Scrape, Visualize, Alert, Validate

6. **Ch06 - Infrastructure as Code (Terraform)**
   - Procedure: Write .tf files → init → plan → apply → manage state → destroy
   - 5 phases: Define, Initialize, Preview, Apply, Manage

#### MEDIUM Complexity (2 chapters)
7. **Ch07 - Networking & Load Balancing**
   - Procedure: Deploy backends → configure Nginx upstream → health checks → test failover
   - 4 phases: Backends, Proxy, Health Checks, Resilience

8. **Ch08 - Security & Secrets Management**
   - Procedure: Audit hardcoded secrets → .env (local) → Docker/K8s secrets → rotation → pre-commit hooks
   - 4 phases: Audit, Local Dev, Production, Rotation

---

### Track 8: Math Under the Hood (2 chapters)

#### HIGH Complexity (1 chapter)
1. **Ch06 - Gradient + Chain Rule (Backpropagation)**
   - Procedure: Forward pass (layer-by-layer) → backward pass (chain Jacobians) → update weights
   - 2 major phases: Forward, Backward (each with 3-4 sub-steps)

#### MEDIUM Complexity (1 chapter)
2. **Ch04 - Small Steps (Gradient Descent)**
   - Procedure: Initialize → compute gradient → update parameters → check convergence → repeat
   - 4 phases: Initialize, Compute, Update, Converge

---

## Parallelization Strategy

### Objective
Retrofit 40 workflow candidate chapters while minimizing:
- **Context windows**: Each subagent works on single chapter, loads only authoring guide + chapter README
- **LLM calls**: Batch independent chapters by track, parallelize across tracks
- **Human review overhead**: Implement in priority waves, validate before next wave

### Implementation Waves

#### **Wave 1: Quick Wins (6 chapters, LOW complexity)**
*Estimated: 2-3 days with 6 parallel subagents*

Focus on smallest, clearest workflow patterns to validate process:

| Chapter | Track | Complexity | Phases | Estimated LOC Changes |
|---------|-------|-----------|--------|-----------------------|
| ch00b_class_imbalance | ML | LOW | 3 | +150 (workflow overlay) |
| ch08_tensorboard | ML | LOW | 4 | +200 (workflow overlay) |
| ch04_small_steps | Math | MEDIUM | 4 | +180 (workflow overlay) |
| ch02_prompt_engineering | AI | MEDIUM | 4 | +250 (workflow overlay) |
| ch02_memory_budgets | AI Infra | MEDIUM | 3 | +180 (workflow overlay) |
| ch08_security_secrets | DevOps | MEDIUM | 4 | +200 (workflow overlay) |

**Parallelization**: Launch 6 subagents simultaneously (one per chapter)

**Subagent Instructions Template**:
```
WORKFLOW RETROFIT: [Track] [Chapter Name]

Read authoring-guide.md workflow pattern section.
Read current chapter README.
Apply workflow overlay (Option A - preserve all existing content):

1. Add §1.5 Practitioner Workflow section with [N]-phase ASCII diagram
2. Add phase markers to section headers: [Phase X: VERB]
3. Add [N] decision checkpoints (3-part format: What you saw → What it means → What to do next)
4. Add [N] executable code snippets (one per phase)
5. Add [N] industry standard callout boxes (manual vs library patterns)

Return: Modified README.md only (no separate plan files)
```

**Success Criteria**: 
- All 6 chapters pass quality audit (Grade A: 85+/100)
- Pattern validated as reusable
- Average implementation time ≤ 4 hours per chapter

---

#### **Wave 2: Production Workflows (10 chapters, MEDIUM complexity)**
*Estimated: 1 week with 10 parallel subagents*

Focus on production-facing chapters (deployment, monitoring, serving):

| Chapter | Track | Complexity | Phases |
|---------|-------|-----------|--------|
| ch00_data_prep | ML | MEDIUM | 4 |
| ch08_data_validation | ML | MEDIUM | 4 |
| ch06_production (anomaly) | ML | MEDIUM | 4 |
| ch06_production (ensemble) | ML | MEDIUM | 4 |
| ch09_cost_and_latency | AI | MEDIUM | 4 |
| ch02_mcp | Multi-Agent | MEDIUM | 4 |
| ch03_a2a | Multi-Agent | MEDIUM | 4 |
| ch05_shared_memory | Multi-Agent | MEDIUM | 5 |
| ch12_generative_eval | Multimodal | MEDIUM | 4 |
| ch13_local_diffusion_lab | Multimodal | MEDIUM | 4 |
| ch06_model_serving | AI Infra | MEDIUM | 4 |
| ch09_experiment_tracking | AI Infra | MEDIUM | 4 |
| ch07_networking_lb | DevOps | MEDIUM | 4 |

**Parallelization**: 2 batches of 5-7 subagents (monitor quality before second batch)

---

#### **Wave 3: Complex Pipelines (10 chapters, HIGH complexity)**
*Estimated: 2 weeks with 5 parallel subagents (sequential batches)*

Focus on multi-phase pipelines (RAG, event-driven, deployment):

| Chapter | Track | Complexity | Phases |
|---------|-------|-----------|--------|
| ch06_cold_start_production | ML | HIGH | 5 |
| ch04_rag_embeddings | AI | HIGH | 5 |
| ch06_react_semantic_kernel | AI | HIGH | 4 |
| ch07_safety_hallucination | AI | HIGH | 3 |
| ch08_evaluating_ai | AI | HIGH | 3 |
| ch10_fine_tuning | AI | HIGH | 4 |
| ch12_testing_ai | AI | HIGH | 3 |
| ch04_event_driven | Multi-Agent | HIGH | 7 |
| ch06_trust_sandboxing | Multi-Agent | HIGH | 5 |
| ch08_text_to_image | Multimodal | HIGH | 4 |
| ch05_inference_optimization | AI Infra | HIGH | 4 |
| ch08_feature_stores | AI Infra | HIGH | 5 |
| ch10_production_monitoring | AI Infra | HIGH | 5 |
| ch11_end_to_end_deployment | AI Infra | HIGH | 6 |
| ch06_backpropagation | Math | HIGH | 2 (complex) |

**Parallelization**: 3 batches of 5 subagents (stagger by 2-3 days for quality validation)

---

#### **Wave 4: DevOps Track (6 remaining chapters)**
*Estimated: 1 week with 6 parallel subagents*

Focus on DevOps fundamentals (Docker, K8s, CI/CD, IaC):

| Chapter | Track | Complexity | Phases |
|---------|-------|-----------|--------|
| ch01_docker_fundamentals | DevOps | HIGH | 4 |
| ch02_container_orchestration | DevOps | HIGH | 4 |
| ch03_kubernetes_basics | DevOps | HIGH | 4 |
| ch04_cicd_pipelines | DevOps | HIGH | 4 |
| ch05_monitoring_observability | DevOps | HIGH | 5 |
| ch06_infrastructure_as_code | DevOps | HIGH | 5 |

**Parallelization**: Single batch of 6 subagents (all chapters have similar structure)

---

### Context Window Optimization

#### Per-Subagent Context Budget
Each subagent loads:
1. **Authoring guide** (~2,000 tokens): Workflow pattern template, decision checkpoint format, code snippet rules
2. **Current chapter README** (~4,000-8,000 tokens): Existing content to preserve
3. **System prompt** (~1,500 tokens): Retrofit instructions, quality criteria
4. **Working memory** (~2,000 tokens): Tracking edits, section mapping

**Total context**: ~10,000-14,000 tokens per subagent (well under 128K limit)

#### Minimizing Redundant Loads
- ❌ **Don't load**: Full track authoring guide plans (only needed for new authoring guides, not chapter retrofits)
- ❌ **Don't load**: Other chapters in same track (no cross-chapter dependencies)
- ❌ **Don't load**: Solution/exercise notebooks (retrofit README only, notebooks updated separately)
- ✅ **Do load**: Only ML authoring guide workflow pattern section (~800 tokens)
- ✅ **Do load**: Chapter-specific README being retrofitted

---

### Quality Assurance

#### Automated Checks (Pre-Commit)
```bash
# Check for workflow pattern compliance
python scripts/audit_workflow_pattern.py --chapter notes/01-ml/.../README.md

# Validates:
# ✓ §1.5 Practitioner Workflow section exists
# ✓ Phase markers in section headers ([Phase 1: VERB])
# ✓ At least 3 decision checkpoints (### DECISION format)
# ✓ At least 3 executable code snippets (```python with real data)
# ✓ At least 3 industry callout boxes (> **Industry Standard:** format)
```

#### Human Review Criteria
- **Pedagogical flow**: Can practitioner follow workflow start-to-finish without jumping?
- **Decision clarity**: Are decision checkpoints clear about what triggers each choice?
- **Code executability**: Do code snippets run without modification?
- **Theory preservation**: Is all original technical content retained?
- **Industry alignment**: Do callout boxes show production-ready patterns?

#### Quality Grades (Same as Ch.3 Feature Engineering)
- **Grade A (85-100)**: Production-ready, minimal revisions
- **Grade B (70-84)**: Good, minor improvements needed
- **Grade C (50-69)**: Needs significant revision
- **Grade F (<50)**: Restart from original

**Target**: 90% of chapters achieve Grade A on first implementation

---

## Risk Mitigation

### Risk 1: Inconsistent Pattern Application
**Mitigation**: 
- Use standardized subagent prompt template (shown above)
- Validate Wave 1 (Quick Wins) before proceeding to Wave 2
- Create `scripts/audit_workflow_pattern.py` for automated compliance checks

### Risk 2: Breaking Existing Cross-References
**Mitigation**:
- Preserve all existing section numbers (§3.1, §3.2, etc.)
- Only ADD new sections (§1.5, decision checkpoints)
- Run `scripts/check_md_links.py` after each wave

### Risk 3: Context Window Overflow (Large Chapters)
**Mitigation**:
- For chapters >10K LOC (e.g., ch01_docker_fundamentals: 663 lines), use Option A (overlay) not Option B (reorganization)
- If chapter README >12K tokens, split subagent work into 2 passes:
  - Pass 1: Add §1.5 + phase markers
  - Pass 2: Add decision checkpoints + code snippets

### Risk 4: Workflow Pattern Not Applicable
**Mitigation**:
- If subagent determines chapter is actually concept-based (false positive from audit), return "NO WORKFLOW NEEDED" verdict
- Human reviews verdict and updates audit results
- Example: If a "production" chapter is actually architecture comparison (tool-focused, not procedure-focused)

---

## Success Metrics

### Quantitative Targets
- **40 chapters retrofitted** by end of Q2 2026
- **90% Grade A** quality on first implementation
- **100% automated compliance** (all chapters pass `audit_workflow_pattern.py`)
- **Zero broken links** after retrofit (all pass `check_md_links.py`)

### Qualitative Targets
- **Practitioner feedback**: "I can now follow a clear workflow" (survey score >4.5/5)
- **Reduced support questions**: 30% fewer "how do I approach this?" questions in discussion forums
- **Faster onboarding**: New learners complete workflow chapters 20% faster than concept chapters

### Completion Timeline
- **Wave 1** (6 chapters): Week 1-2 (May 1-12, 2026)
- **Wave 2** (13 chapters): Week 3-4 (May 13-26, 2026)
- **Wave 3** (15 chapters): Week 5-7 (May 27-Jun 16, 2026)
- **Wave 4** (6 chapters): Week 8 (Jun 17-23, 2026)
- **Final QA & Documentation**: Week 9-10 (Jun 24-Jul 7, 2026)

**Target Completion**: July 7, 2026 (10 weeks total)

---

## Next Steps

### Immediate Actions (This Week)
1. ✅ **Audit complete** - All 8 tracks audited via parallel subagents
2. ⏳ **Create WORKFLOW_ROLLOUT_PLAN.md** - This document
3. ⏳ **Create `scripts/audit_workflow_pattern.py`** - Automated compliance checker
4. ⏳ **Validate subagent prompt template** - Test on 1 chapter manually
5. ⏳ **Launch Wave 1** - 6 parallel subagents for quick wins

### Weekly Cadence
- **Monday**: Launch subagent batch (5-7 chapters)
- **Wednesday**: Mid-week quality check (review 2-3 completed chapters)
- **Friday**: Batch completion review, decide on next wave
- **Weekend**: Address any quality issues, prepare next batch

### Stakeholder Communication
- **Weekly update**: Progress dashboard (chapters complete, quality grades, timeline)
- **Monthly retrospective**: Lessons learned, pattern refinements, blocker resolution
- **Completion report**: Final metrics, before/after comparisons, practitioner testimonials

---

## Appendix A: Subagent Prompt Template

```markdown
# WORKFLOW RETROFIT TASK

## Chapter
**Path**: [notes/.../README.md]
**Track**: [ML/AI/DevOps/etc.]
**Complexity**: [LOW/MEDIUM/HIGH]
**Phases**: [N phases identified in audit]

## Instructions

Read the workflow pattern documentation:
1. [notes/01-ml/authoring-guide.md] - "Workflow-Based Chapter Pattern" section
2. Current chapter README at path above

Apply workflow overlay (Option A - preserve all content):

### Required Additions
1. **§1.5 Practitioner Workflow** (~150 lines)
   - ASCII diagram showing [N] phases
   - Warning box: "Two ways to read this chapter"
   - Brief description of each phase

2. **Phase Markers in Section Headers** ([N] sections)
   - Format: `## [Phase 1: VERB] Original Section Title`
   - Example: `## [Phase 1: INSPECT] Feature Distributions`

3. **Decision Checkpoints** ([N] checkpoints, one per phase)
   - Format: 3-part structure
     - **What you saw**: Observation from previous phase
     - **What it means**: Interpretation/diagnosis
     - **What to do next**: Actionable next step with tool/threshold
   - Place at END of each phase section

4. **Executable Code Snippets** ([N] snippets, one per phase)
   - Real data (California Housing for ML, realistic examples for other tracks)
   - Copy-paste ready (imports + execution)
   - Commented decision logic

5. **Industry Standard Callout Boxes** ([N] callouts)
   - Format: `> **Industry Standard:** [Library/Tool Name]`
   - Show manual implementation (for learning) vs library call (for production)
   - Include 1-2 line explanation of when to use each

### Quality Criteria
- ✓ All original content preserved (no deletions except fixing errors)
- ✓ Section numbers unchanged (§3.1 stays §3.1)
- ✓ All code snippets syntactically complete
- ✓ Decision checkpoints use consistent 3-part format
- ✓ Phase markers match audit-identified phases

### Output
Return ONLY the modified README.md content. Do not create separate plan files.

## Reference Example
See [notes/01-ml/01_regression/ch03_feature_importance/README.md] for complete workflow pattern implementation (Grade A: 90/100).
```

---

## Appendix B: Automated Compliance Checker

```python
# scripts/audit_workflow_pattern.py
"""
Validate chapter README against workflow pattern requirements.

Usage:
    python scripts/audit_workflow_pattern.py --chapter notes/01-ml/.../README.md

Checks:
    1. §1.5 Practitioner Workflow section exists
    2. At least N phase markers ([Phase 1: VERB]) where N >= 3
    3. At least 3 decision checkpoints (### DECISION format)
    4. At least 3 executable code snippets (```python with imports)
    5. At least 3 industry callout boxes (> **Industry Standard:**)
    6. No broken section references (§ 3.1 should be §3.1)

Returns:
    - EXIT 0: All checks pass
    - EXIT 1: One or more checks fail (prints violations)
"""

import re
import sys
from pathlib import Path

def audit_workflow_compliance(readme_path: Path) -> dict:
    content = readme_path.read_text(encoding='utf-8')
    
    results = {
        'has_workflow_section': bool(re.search(r'§1\.5.*Practitioner Workflow', content)),
        'phase_markers': len(re.findall(r'\[Phase \d+:', content)),
        'decision_checkpoints': len(re.findall(r'### .*DECISION', content)),
        'code_snippets': len(re.findall(r'```python', content)),
        'industry_callouts': len(re.findall(r'> \*\*Industry Standard:', content)),
        'broken_refs': len(re.findall(r'§ \d+\.', content)),  # Space after §
    }
    
    # Validation
    failures = []
    if not results['has_workflow_section']:
        failures.append("Missing §1.5 Practitioner Workflow section")
    if results['phase_markers'] < 3:
        failures.append(f"Insufficient phase markers: {results['phase_markers']} (need >= 3)")
    if results['decision_checkpoints'] < 3:
        failures.append(f"Insufficient decision checkpoints: {results['decision_checkpoints']} (need >= 3)")
    if results['code_snippets'] < 3:
        failures.append(f"Insufficient code snippets: {results['code_snippets']} (need >= 3)")
    if results['industry_callouts'] < 3:
        failures.append(f"Insufficient industry callouts: {results['industry_callouts']} (need >= 3)")
    if results['broken_refs'] > 0:
        failures.append(f"Found {results['broken_refs']} broken section references (§ N.M should be §N.M)")
    
    return {'results': results, 'failures': failures, 'passed': len(failures) == 0}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=Path, required=True)
    args = parser.parse_args()
    
    audit = audit_workflow_compliance(args.chapter)
    
    if audit['passed']:
        print(f"✅ PASS: {args.chapter}")
        sys.exit(0)
    else:
        print(f"❌ FAIL: {args.chapter}")
        for failure in audit['failures']:
            print(f"  - {failure}")
        sys.exit(1)
```

---

## Appendix C: Context Window Budget (Per Subagent)

| Component | Estimated Tokens | Notes |
|-----------|------------------|-------|
| **System Prompt** | 1,500 | Retrofit instructions, quality criteria, output format |
| **Authoring Guide Workflow Section** | 800 | Only workflow pattern section from ML authoring guide |
| **Current Chapter README** | 4,000-8,000 | Varies by chapter size (avg ~6,000) |
| **Working Memory** | 2,000 | Tracking edits, section mapping, decision checkpoints |
| **Reference Example (Ch.3)** | 1,000 | Key sections only (§1.5, sample checkpoint, sample callout) |
| **Output Buffer** | 8,000-12,000 | Modified README content |
| **TOTAL** | **17,300-25,300** | Well under 128K context limit |

**Optimization Notes**:
- For chapters >10K tokens, load in 2 passes (structure first, then content)
- Don't load full authoring guide plans (only pattern template needed)
- Don't load other track examples (Ch.3 from ML track is sufficient reference)

---

## Appendix D: Complexity Estimates

| Complexity | Chapter Count | Avg Implementation Time | Avg LOC Changes |
|------------|---------------|-------------------------|-----------------|
| **LOW** | 2 | 3-4 hours | +150-200 |
| **MEDIUM** | 23 | 4-6 hours | +200-300 |
| **HIGH** | 15 | 8-12 hours | +350-500 |

**Total Estimated Effort**: 
- LOW: 2 × 4h = 8h
- MEDIUM: 23 × 5h = 115h
- HIGH: 15 × 10h = 150h
- **TOTAL**: 273 hours (~7 weeks with 40h/week, or 5 weeks with parallel subagents)

**With 6 Parallel Subagents**: ~45 hours elapsed (1.5 weeks) + QA overhead = **2-3 weeks total**

---

**Status**: Ready for Wave 1 execution  
**Next Action**: Launch 6 parallel subagents for Quick Wins batch  
**Owner**: AI Agent Coordinator  
**Review Cadence**: Weekly progress updates
