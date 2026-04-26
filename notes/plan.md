# AI Portfolio — Master Learning Plan

> **Purpose:** Complete pedagogical roadmap across all topics with numbered sequencing, prerequisites, and learning paths  
> **Last updated:** April 26, 2026  
> **Status:** ✅ Folder renaming complete

---

## Overview

This portfolio contains **5 core learning tracks** plus supporting material (interview guides, archives):

| # | Track | Folder | Status | Pedagogical Focus |
|---|-------|--------|--------|-------------------|
| **0** | Math Foundations | `00-math_under_the_hood/` | ✅ Complete | Prerequisites: calculus, linear algebra, optimization (7 chapters) |
| **1** | Machine Learning | `01-ml/` | ✅ Complete | Supervised, unsupervised, deep learning (8 topics, 50+ chapters) |
| **2** | Agentic AI | `02-ai/` | ✅ Complete (Ch.1-11) | Single-agent systems: RAG, tools, orchestration, advanced patterns (11 chapters) |
| **3** | Multi-Agent AI | `03-multi_agent_ai/` | ✅ Complete | Multi-agent protocols and coordination (7 chapters) |
| **4** | Multimodal AI | `04-multimodal_ai/` | ✅ Complete | Generative image/video/audio models (13 chapters) |
| **5** | AI Infrastructure | `05-ai_infrastructure/` | ✅ Complete (Ch.1-10) | GPU architecture through feature stores — all 10 chapters complete |
| **6** | DevOps Fundamentals | `06-devops_fundamentals/` | ✅ Complete (Ch.1-8) | Docker through security — all 8 chapters complete |

---

## Folder Renaming Complete ✅

**Completed actions:**
- ✅ Renamed `notes/ml/` → `notes/01-ml/`
- ✅ Renamed `notes/math_under_the_hood/` → `notes/00-math_under_the_hood/`
- ✅ Renamed `notes/ai/` → `notes/02-ai/`
- ✅ Renamed `notes/multi_agent_ai/` → `notes/03-multi_agent_ai/`
- ✅ Renamed `notes/multimodal_ai/` → `notes/04-multimodal_ai/`
- ✅ Renamed `notes/ai_infrastructure/` → `notes/05-ai_infrastructure/`
- ✅ Renamed `notes/devops_fundamentals/` → `notes/06-devops_fundamentals/`
- ✅ Updated all cross-references in READMEs and notebooks
- ✅ Updated root README.md with new folder names
- ✅ Updated scripts with hardcoded paths

**Rationale:**
- **Track 0 (Math):** Prerequisite material — foundations before ML
- **Track 1 (ML):** Core supervised/unsupervised learning — the foundation everything else builds on
- **Track 2 (Agentic AI):** Single-agent LLM systems — reasoning, retrieval, tool use
- **Track 3 (Multi-Agent AI):** Agent coordination & protocols — requires understanding single agents first
- **Track 4 (Multimodal AI):** Image/video/audio generation — extends transformers to multimodal
- **Track 5 (AI Infrastructure):** GPU optimization, serving, monitoring — production layer for all AI systems
- **Track 6 (DevOps Fundamentals):** Containers, orchestration, CI/CD — prerequisite for deploying AI systems

Numerical prefixes enforce pedagogical sequence and make the intended learning path explicit.

---

## Track 0 — Math Foundations (`00-math_under_the_hood/`)

**Pedagogical goal:** Build intuition for ML math through a single physical metaphor (knuckleball free-kick)

**Prerequisites:** High school algebra, basic calculus concepts (derivative = slope)

**Time estimate:** 8-12 hours (if math is rusty)

### Chapter Sequence

| Ch | Title | What it unlocks | ML Track Prerequisites |
|----|-------|-----------------|------------------------|
| 1 | Linear Algebra | Vectors, dot products, matrix multiplication | ML Ch.1-2 (Regression, Multiple Regression) |
| 2 | Non-Linear Algebra | Exponential, logarithm, sigmoid | ML Ch.3 (Classification) |
| 3 | Calculus Foundations | Derivatives, chain rule | ML Ch.4 (Gradient Descent) |
| 4 | 1-D Optimization | Gradient descent on single parameter | ML Ch.4 (Optimization) |
| 5 | Matrix Operations | Broadcasting, reshaping, transpose | ML Ch.5 (Neural Networks) |
| 6 | Gradients & Chain Rule | Multi-parameter optimization, backprop intuition | ML Ch.5 (Backpropagation) |
| 7 | Probability & Statistics | Expectation, variance, Bayes' theorem | ML Ch.6 (Evaluation Metrics) |

**Completion criteria:**
- ✅ Can derive gradient of $f(x) = wx + b$ by hand
- ✅ Can multiply matrices and explain broadcasting
- ✅ Can compute dot product and explain geometric meaning
- ✅ Can apply chain rule to nested functions

**Skip conditions:**
- Already comfortable with gradient descent math → Skip to Track 1
- Need quick refresher only → Read READMEs without notebooks

---

## Track 1 — Machine Learning (`01-ml/`)

**Pedagogical goal:** Ground-up understanding from linear regression to transformers

**Prerequisites:** Track 0 (Ch.1-6) or equivalent math background

**Time estimate:** 40-50 hours (full track), 20-25 hours (core topics only)

### Topic 1: Regression (6 chapters)

**Running example:** California Housing dataset — predict home prices

**Goal:** Master supervised learning foundations

| Ch | Title | What it unlocks | Key Concepts | Status |
|----|-------|-----------------|--------------|--------|
| 1 | Linear Regression | Single-variable prediction | OLS, loss functions, analytical solution | ✅ |
| 2 | Multiple Regression | Multi-variable prediction | Feature scaling, matrix form | ✅ |
| 3 | Polynomial Features | Non-linear relationships | Feature engineering, overfitting intro | ✅ |
| 4 | Gradient Descent | Iterative optimization | Learning rate, convergence, batch vs SGD | ✅ |
| 5 | Regularization (Ridge/Lasso) | Combat overfitting | L1/L2 penalties, hyperparameter λ | ✅ |
| 6 | Evaluation Metrics | Model assessment | MSE, MAE, R², cross-validation | ✅ |

**Completion criteria:**
- ✅ Can implement linear regression from scratch (no sklearn)
- ✅ Can explain why gradient descent converges
- ✅ Can debug overfitting with regularization
- ✅ Can choose appropriate evaluation metric

### Topic 2: Classification (5 chapters)

**Running example:** Binary classification (TBD dataset)

**Goal:** Extend regression to discrete predictions

| Ch | Title | What it unlocks | Key Concepts | Status |
|----|-------|-----------------|--------------|--------|
| 1 | Logistic Regression | Binary classification | Sigmoid, log-loss, decision boundary | ✅ |
| 2 | Softmax Regression | Multi-class classification | Softmax, cross-entropy | ✅ |
| 3 | Classification Metrics | Model assessment | Accuracy, precision, recall, F1, ROC-AUC | ✅ |
| 4 | Support Vector Machines | Max-margin classifiers | Kernel trick, RBF kernel | ✅ |
| 5 | Hyperparameter Tuning | Model selection | Grid search, random search, validation curves | ✅ |

**Completion criteria:**
- ✅ Can explain sigmoid vs softmax
- ✅ Can compute precision/recall from confusion matrix
- ✅ Can choose SVM kernel based on data shape
- ✅ Can tune hyperparameters without overfitting

### Topic 3: Neural Networks (18 chapters)

**Running example:** MNIST, ImageNet, sequence modeling

**Goal:** Deep learning foundations from XOR to transformers

| Ch | Title | What it unlocks | Key Concepts | Status |
|----|-------|-----------------|--------------|--------|
| 1 | XOR Problem | Why neural networks exist | Perceptron limitations, hidden layers | ✅ |
| 2 | Feedforward Networks | Multi-layer perceptrons | Activation functions, forward pass | ✅ |
| 3 | Backpropagation | How training works | Chain rule, gradient flow | ✅ |
| 4 | Neural Network Regularization | Prevent overfitting | Dropout, batch norm, early stopping | ✅ |
| 5 | Convolutional Neural Networks | Image recognition | Convolution, pooling, spatial hierarchies | ✅ |
| 6 | Recurrent Neural Networks | Sequence modeling | LSTM, GRU, vanishing gradients | ✅ |
| 7 | Maximum Likelihood Estimation | Training objective theory | MLE, cross-entropy derivation | ✅ |
| 8 | TensorBoard | Training visualization | Loss curves, histograms, embeddings | ✅ |
| 9 | Attention Mechanisms | Seq2seq with focus | Query-key-value, attention weights | ✅ |
| 10 | Transformers | Self-attention, GPT/BERT | Multi-head attention, positional encoding | ✅ |
| 11-18 | Advanced Topics | Production neural networks | (TBD) | ⏳ |

**Completion criteria:**
- ✅ Can implement backprop from scratch (simple 2-layer network)
- ✅ Can explain why CNNs work for images
- ✅ Can derive attention mechanism math
- ✅ Can explain transformer encoder vs decoder

### Topics 4-8: Specialized ML Domains

| Topic | Chapters | Key Use Case | Prerequisites | Status |
|-------|----------|--------------|---------------|--------|
| **04: Recommender Systems** | 6 | Personalization, collaborative filtering | Regression, Classification | ✅ |
| **05: Anomaly Detection** | 6 | Fraud detection, outlier detection | Classification, Neural Networks | ✅ |
| **06: Reinforcement Learning** | 6 | MDPs, Q-learning, policy gradients | Neural Networks | ✅ |
| **07: Unsupervised Learning** | 4 | Clustering, dimensionality reduction | Linear Algebra, Statistics | ✅ |
| **08: Ensemble Methods** | 3 | Bagging, boosting, XGBoost | Classification, Decision Trees | ✅ |

**Learning path:**
- **Core foundations:** Topics 1-3 (Regression → Classification → Neural Networks)
- **Specializations:** Pick based on career goal:
  - **Applied ML engineer:** Topics 4, 7, 8
  - **AI safety researcher:** Topics 5, 6
  - **Research engineer:** All topics

---

## Track 2 — Agentic AI (`02-ai/`)

**Pedagogical goal:** Transform LLMs from text predictors into production agents

**Prerequisites:** 
- Track 1 Topic 3 Ch.10 (Transformers) — understand what LLMs are
- Basic API usage (REST, JSON)

**Time estimate:** 10-14 hours

**Running example:** Mamma Rosa's PizzaBot — AI ordering system beating human phone staff

### Chapter Sequence

| Ch | Title | What it unlocks | Constraint Unlocked | Status |
|----|-------|-----------------|---------------------|--------|
| 1 | LLM Fundamentals | Token prediction, context windows | Foundation knowledge | ✅ |
| 2 | Prompt Engineering | System prompts, structured output | Accuracy (partial) | ✅ |
| 3 | Chain-of-Thought Reasoning | Multi-step reasoning | Logic (not facts) | ✅ |
| 4 | RAG & Embeddings | Grounded retrieval | ✅ Accuracy <5% error | ✅ |
| 5 | Vector Databases | Scale retrieval | Latency (partial) | ✅ |
| 6 | ReAct & Semantic Kernel | Tool orchestration | ✅ Business value >25% conv | ✅ |
| 7 | Safety & Hallucination | Defense against attacks | ✅ Safety (zero attacks) | ✅ |
| 8 | Evaluating AI Systems | Metrics, evals, A/B testing | Reliability (partial) | ✅ |
| 9 | Cost & Latency Optimization | Production performance | ✅ Cost <$0.08/conv, Latency <3s | ✅ |
| 10 | Fine-Tuning | Domain adaptation | ✅ Reliability >99% uptime | ✅ |
| **11** | **Advanced Agentic Patterns** | **Reflection, debate, orchestration** | **Edge case handling <1% error** | ✅ **Complete** |

### Ch.11 Details — Advanced Agentic Patterns

**NEW CHAPTER (Planned):**

**What's missing:** Current track covers single-pass agent loops. Ch.11 adds iterative refinement patterns:

1. **Reflection** (self-critique and iterative refinement)
   - Generate → Critique → Revise loop
   - Use case: Ambiguous inputs, complex reasoning
   - Cost: 3× tokens vs. single-pass
   - Animation: Draft (red) → Critique (yellow) → Revised (green)

2. **Debate & Consensus** (multi-agent reasoning)
   - Propose → Challenge → Defend → Vote loop
   - Use case: High-stakes decisions (medical, legal, fraud)
   - Agents: Proposer 1, Proposer 2, Judge
   - Animation: Speech bubbles, policy doc popup, consensus Venn diagram

3. **Hierarchical Orchestration** (planner → workers → verifier)
   - Decompose → Execute → Verify loop
   - Use case: Complex multi-step tasks (research, catering orders)
   - Agents: Planner, Workers (N), Verifier
   - Animation: Tree structure, Gantt chart, checkmarks

4. **Tool Selection Strategies**
   - Rule-based, cost-based, LLM-based meta-agent
   - Error recovery: Fallback chains
   - Animation: Decision tree, waterfall diagram

**Animations (8 total):**
- [ ] Reflection loop (generate → critique → revise)
- [ ] Reflection convergence graph (error rate vs. loops)
- [ ] Debate flow (multi-agent speech bubbles)
- [ ] Debate consensus Venn diagram
- [ ] Hierarchical orchestration tree
- [ ] Hierarchical coordination Gantt chart
- [ ] Tool selection decision tree
- [ ] Tool fallback chain waterfall
- [ ] Pattern comparison table (heatmap)
- [ ] Pattern needle movement (error rate reduction)

**Bridge to Track 3:** These single-agent patterns scale to multi-agent systems (MCP, A2A, event-driven)

**Completion criteria:**
- ✅ All 6 constraints met (business value, accuracy, latency, cost, safety, reliability)
- ✅ Can build production RAG pipeline
- ✅ Can orchestrate multi-tool agent
- ✅ Can defend against prompt injection
- ✅ Can optimize for cost and latency
- ✅ Can handle edge cases with reflection/debate patterns

---

## Track 3 — Multi-Agent AI (`03-multi_agent_ai/`)

**Pedagogical goal:** Scale single-agent patterns to multi-agent systems

**Prerequisites:**
- Track 2 Ch.6 (ReAct & Semantic Kernel) — understand agent loops
- Track 2 Ch.11 (Advanced Agentic Patterns) — understand coordination patterns

**Time estimate:** 8-10 hours

**Running example:** OrderFlow — B2B purchase order automation

### Chapter Sequence

| Ch | Title | What it unlocks | Pattern Extended | Status |
|----|-------|-----------------|------------------|--------|
| 1 | Message Formats | Agent communication primitives | Reflection (cross-agent critique) | ✅ |
| 2 | MCP (Model Context Protocol) | Standardized tool access | Tool selection (MCP servers) | ✅ |
| 3 | A2A (Agent-to-Agent) | Task delegation | Hierarchical orchestration (specialist agents) | ✅ |
| 4 | Event-Driven Agents | Async pub/sub coordination | Debate (async negotiation) | ✅ |
| 5 | Shared Memory | Blackboard patterns | Hierarchical (shared context) | ✅ |
| 6 | Trust & Sandboxing | Multi-agent safety | Safety (untrusted agents) | ✅ |
| 7 | Agent Frameworks | LangGraph, AutoGen, Semantic Kernel | Production orchestration | ✅ |

**Completion criteria:**
- ✅ Can design agent message protocol
- ✅ Can expose tool as MCP server
- ✅ Can delegate tasks via A2A
- ✅ Can build event-driven agent pipeline
- ✅ Can implement shared memory with blackboard
- ✅ Can sandbox untrusted agents

---

## Track 4 — Multimodal AI (`04-multimodal_ai/`)

**Pedagogical goal:** Generate images, video, audio with diffusion models

**Prerequisites:**
- Track 1 Topic 3 Ch.5 (CNNs) — understand convolutions
- Track 1 Topic 3 Ch.10 (Transformers) — understand attention
- Track 2 Ch.4 (RAG & Embeddings) — understand embeddings

**Time estimate:** 12-16 hours

**Running example:** PixelSmith — local AI creative studio (no cloud GPU)

### Chapter Sequence

| Ch | Title | What it unlocks | GPU Constraint | Status |
|----|-------|-----------------|----------------|--------|
| 1 | Multimodal Foundations | Cross-modal alignment | Conceptual | ✅ |
| 2 | Vision Transformers (ViT) | Patch embeddings, image classification | 8GB VRAM | ✅ |
| 3 | CLIP | Text-image retrieval, zero-shot | 8GB VRAM | ✅ |
| 4 | Diffusion Models (DDPM) | Forward/reverse process | 12GB VRAM | ✅ |
| 5 | Latent Diffusion (Stable Diffusion) | VAE latent space, CFG | 8GB VRAM (quantized) | ✅ |
| 6 | Schedulers | DDIM, DPM-Solver, Euler-a | Speed optimization | ✅ |
| 7 | Guidance & Conditioning | ControlNet, img2img, inpainting | Advanced control | ✅ |
| 8 | Text-to-Image | End-to-end pipeline | Production | ✅ |
| 9 | Text-to-Video | Temporal attention | 16GB VRAM | ✅ |
| 10 | Multimodal LLMs | Vision encoders in LLMs | 24GB VRAM | ✅ |
| 11 | Generative Evaluation | FID, IS, CLIP score | Quality metrics | ✅ |
| 12 | Local Diffusion Lab | Memory optimization, quantization | 6GB VRAM | ✅ |
| 13 | Audio Generation | Text-to-speech, vocoder | CPU-first | ✅ |

**Completion criteria:**
- ✅ Can run Stable Diffusion locally (no cloud)
- ✅ Can explain diffusion forward/reverse process
- ✅ Can use ControlNet for guided generation
- ✅ Can evaluate generated images (FID, CLIP score)
- ✅ Can optimize for consumer GPU (quantization, attention slicing)

---

## Track 5 — AI Infrastructure (`05-ai_infrastructure/`)

**Pedagogical goal:** GPU hardware to production ML systems

**Prerequisites:**
- Track 1 Topic 3 Ch.10 (Transformers) — understand model architecture
- Basic Linux command line
- Python package management

**Time estimate:** 10-14 hours (Ch.1-5 complete), 20-25 hours (full track)

**Running example:** InferenceBase — self-host Llama-3-8B vs. $80k/month API bills

### Chapter Sequence

| Ch | Title | What it unlocks | Business Impact | Status |
|----|-------|-----------------|-----------------|--------|
| 1 | GPU Architecture | CUDA cores, tensor cores, VRAM | Hardware understanding | ✅ |
| 2 | Memory & Compute Budgets | VRAM estimation, optimizer states | "Will this fit?" | ✅ |
| 3 | Quantization & Precision | FP16/INT8/INT4, GPTQ, AWQ | 4× memory savings | ✅ |
| 4 | Parallelism & Distributed Training | Data/tensor/pipeline, ZeRO, FSDP | Multi-GPU training | ✅ |
| 5 | Inference Optimization | KV cache, flash attention, speculative decoding | 3× throughput | ✅ |
| 6 | Model Serving Frameworks | vLLM, TensorRT-LLM, TGI | Production serving | ✅ Complete |
| 7 | AI-Specific Networking | NVLink, InfiniBand, RDMA | Multi-GPU inference | ✅ Complete |
| 8 | Feature Stores | Feast, real-time features | ML data infrastructure | ✅ Complete |
| 9 | ML Experiment Tracking | MLflow, DVC, model registry | Reproducibility | ✅ Complete |
| 10 | Production Monitoring | Evidently, drift detection, A/B testing | Production reliability | ✅ Complete |

**Note:** Generic DevOps (Docker, Kubernetes, CI/CD, Prometheus) moved to Track 6

**Completion criteria (Ch.1-5):**
- ✅ Can estimate VRAM for any model
- ✅ Can quantize model to fit consumer GPU
- ✅ Can set up multi-GPU training
- ✅ Can optimize inference latency 3×

**Completion criteria (Ch.6-10 — now complete):**
- ✅ Can deploy vLLM serving endpoint
- ✅ Can configure NVLink/InfiniBand for multi-GPU
- ✅ Can implement feature stores for real-time ML
- ✅ Can track experiments with MLflow
- ✅ Can monitor model drift in production
- ✅ Can run A/B tests on model versions

**Track completion notes:**
- All 10 chapters complete (GPU architecture through production monitoring)
- Covers full stack: hardware → optimization → serving → operations
- Running example (InferenceBase) achieves $80k/month → $7.3k/month cost reduction

---

## Track 6 — DevOps Fundamentals (`06-devops_fundamentals/`)

**Pedagogical goal:** Generic infrastructure skills (not AI-specific)

**Prerequisites:**
- Python programming
- Command line basics
- HTTP/REST API concepts

**Time estimate:** 16-20 hours

**Focus:** Docker, Kubernetes, CI/CD, monitoring (applicable to ANY application)

### Chapter Sequence

| Ch | Title | What it unlocks | Applies To | Status |
|----|-------|-----------------|------------|--------|
| 1 | Docker Fundamentals | Containers, images, Dockerfiles | All apps | ✅ Complete |
| 2 | Container Orchestration (Docker Compose) | Multi-container apps | 3-tier apps | ✅ Complete |
| 3 | Kubernetes Basics (Kind) | Pods, services, deployments | Cloud-native apps | ✅ Complete |
| 4 | CI/CD Pipelines (GitHub Actions) | Automated testing, deployment | All projects | ✅ Complete |
| 5 | Monitoring & Observability (Prometheus + Grafana) | Metrics, logs, traces | Production systems | ✅ Complete |
| 6 | Infrastructure as Code (Terraform) | Reproducible infrastructure | Cloud deployments | ✅ Complete |
| 7 | Networking & Load Balancing (Nginx) | Reverse proxy, failover | Web services | ✅ Complete |
| 8 | Security & Secrets Management | Secrets rotation, RBAC | Production security | ✅ Complete |

**Bridge to Track 5:** After Track 6, apply DevOps to ML deployments (Track 5 Ch.6-10)

**Completion criteria:**
- ✅ Can containerize any application
- ✅ Can deploy to local Kubernetes (Kind)
- ✅ Can set up CI/CD pipeline
- ✅ Can monitor with Prometheus/Grafana
- ✅ Can write Terraform configurations
- ✅ Can secure secrets in production

---

## Recommended Learning Paths

### Path 1: AI Engineer (RAG + Agents) — 25-30 hours

**Goal:** Build production agentic systems

```
Track 0 (optional refresh) → Track 1 Topics 1-2 → Track 1 Topic 3 Ch.1-10 → 
Track 2 (all) → Track 3 (all)
```

**Milestones:**
1. After Track 1: Understand ML foundations
2. After Track 2 Ch.4: Build RAG pipeline
3. After Track 2 Ch.6: Build tool-calling agent
4. After Track 2 Ch.11: Handle edge cases with reflection/debate
5. After Track 3: Scale to multi-agent systems

**Portfolio project:** Multi-agent RAG system (e.g., research assistant, customer support)

### Path 2: ML Engineer (Classical + Deep Learning) — 40-50 hours

**Goal:** Train and deploy ML models

```
Track 0 (if needed) → Track 1 (all topics) → Track 5 Ch.1-5 → 
Track 6 (Docker, CI/CD)
```

**Milestones:**
1. After Track 1 Topics 1-2: Implement regression/classification from scratch
2. After Track 1 Topic 3: Train CNN on MNIST
3. After Track 1 Topic 8: Build XGBoost ensemble
4. After Track 5: Optimize inference for production
5. After Track 6: Deploy with Docker + CI/CD

**Portfolio project:** End-to-end ML pipeline (training → evaluation → deployment)

### Path 3: Generative AI Engineer (Multimodal) — 30-35 hours

**Goal:** Build text-to-image, text-to-video systems

```
Track 0 (if needed) → Track 1 Topic 3 Ch.1-10 → Track 2 Ch.1-4 → 
Track 4 (all) → Track 5 Ch.1-5
```

**Milestones:**
1. After Track 1 Topic 3: Understand transformers
2. After Track 2 Ch.4: Understand embeddings
3. After Track 4 Ch.5: Run Stable Diffusion locally
4. After Track 4 Ch.7: Use ControlNet for guided generation
5. After Track 5: Optimize for consumer GPU

**Portfolio project:** Local creative studio (text-to-image + img2img + inpainting)

### Path 4: AI Infrastructure Engineer — 35-40 hours

**Goal:** Optimize and deploy ML at scale

```
Track 0 (if needed) → Track 1 Topic 3 Ch.1-10 → Track 5 (all) → 
Track 6 (all) → Track 2 Ch.9 (cost/latency)
```

**Milestones:**
1. After Track 1 Topic 3: Understand model architecture
2. After Track 5 Ch.1-3: Estimate VRAM, quantize models
3. After Track 5 Ch.4-5: Multi-GPU training + inference optimization
4. After Track 6: Deploy with Kubernetes + monitoring
5. After Track 2 Ch.9: Optimize for production SLAs

**Portfolio project:** Self-hosted LLM serving platform (vLLM + K8s + monitoring)

### Path 5: Interview Prep (Breadth-First) — 10-15 hours

**Goal:** Pass AI/ML interviews

```
Track 1 Topics 1-2 (fast read) → Track 1 Topic 3 Ch.1-10 → 
Track 2 Ch.1-6 → Track 2 Ch.11 → Interview Guides (all)
```

**Milestones:**
1. After Track 1: Answer "Explain gradient descent"
2. After Track 2 Ch.4: Answer "Design a RAG system"
3. After Track 2 Ch.6: Answer "How do agents use tools?"
4. After Track 2 Ch.11: Answer "When would you use reflection vs. single-pass?"
5. After Interview Guides: Complete mock interview

**Focus:** Interview Guides for rapid-fire Q&A + per-chapter Interview Checklists

---

## Cross-Track Dependencies

### Dependency Graph

```
Track 0 (Math) ───────────────────────────────────┐
                                                   ↓
Track 1 (ML) ────────────────┬───────────────────┴─→ Track 2 (Agentic AI)
      │                      │                              │
      │                      │                              ↓
      │                      ├──────────→ Track 4      Track 3 (Multi-Agent)
      │                      │         (Multimodal)
      │                      │
      ↓                      ↓
Track 5 (AI Infra) ← Track 6 (DevOps)
```

**Key prerequisites:**
- Track 2 requires Track 1 Topic 3 Ch.10 (Transformers)
- Track 3 requires Track 2 Ch.6 (ReAct) and Track 2 Ch.11 (Advanced Patterns)
- Track 4 requires Track 1 Topic 3 Ch.5 (CNNs) + Ch.10 (Transformers)
- Track 5 requires Track 1 Topic 3 Ch.10 (Transformers)
- Track 6 is independent (can be learned anytime)

### Suggested Parallel Learning

**Can be learned in parallel:**
- Track 1 Topics 1-2 + Track 0 (math refresher)
- Track 1 Topic 3 + Track 2 Ch.1-3 (after transformers)
- Track 5 Ch.1-3 + Track 1 Topic 3 (hardware + software)
- Track 6 + any other track (DevOps is orthogonal)

**Should be learned sequentially:**
- Track 2 Ch.1-10 → Track 2 Ch.11 (foundations → advanced patterns)
- Track 2 → Track 3 (single-agent → multi-agent)
- Track 5 Ch.1-5 → Track 5 Ch.6-10 (hardware → software)

---

## Time Estimates by Role

| Role | Core Tracks | Time | Portfolio Projects |
|------|-------------|------|-------------------|
| **AI Engineer** | 0, 1 (partial), 2, 3 | 25-30h | Multi-agent RAG system |
| **ML Engineer** | 0, 1, 5 (partial), 6 (partial) | 40-50h | End-to-end ML pipeline |
| **Generative AI Engineer** | 0, 1 (partial), 2 (partial), 4, 5 (partial) | 30-35h | Local creative studio |
| **AI Infrastructure Engineer** | 0, 1 (partial), 5, 6, 2 (partial) | 35-40h | Self-hosted LLM platform |
| **Interview Prep** | 1 (core), 2 (core), Interview Guides | 10-15h | Mock interviews |
| **Full Curriculum** | All tracks | 120-150h | Multiple projects |

---

## Progress Tracking

### Per-Track Checklists

**Track 0 (Math):**
- [ ] Ch.1-7 complete (all notebooks run successfully)
- [ ] Can derive gradient by hand
- [ ] Can multiply matrices without calculator

**Track 1 (ML):**
- [ ] Topics 1-3 complete (regression, classification, neural networks)
- [ ] Can implement linear regression from scratch
- [ ] Can train CNN on MNIST
- [ ] Can explain transformer architecture

**Track 2 (Agentic AI):**
- [x] Ch.1-10 complete
- [x] Ch.11 (Advanced Patterns) complete
- [x] Can build RAG pipeline
- [x] Can build tool-calling agent
- [x] Can implement reflection pattern

**Track 3 (Multi-Agent AI):**
- [x] Ch.1-7 complete
- [x] Can design agent message protocol
- [x] Can expose MCP server
- [x] Can delegate via A2A

**Track 4 (Multimodal AI):**
- [x] Ch.1-13 complete
- [x] Can run Stable Diffusion locally
- [x] Can use ControlNet

**Track 5 (AI Infrastructure):**
- [x] Ch.1-10 complete (all chapters finished)
- [x] Can estimate VRAM
- [x] Can quantize models
- [x] Can deploy with vLLM
- [x] Can configure NVLink/InfiniBand
- [x] Can implement feature stores
- [x] Can track experiments with MLflow
- [x] Can monitor production ML systems

**Track 6 (DevOps):**
- [x] Ch.1-8 complete (all chapters finished)
- [x] Can containerize applications
- [x] Can deploy to Kubernetes
- [x] Can set up CI/CD
- [x] Can implement monitoring with Prometheus/Grafana
- [x] Can write infrastructure as code with Terraform
- [x] Can secure secrets in production

---

## Next Actions

### Immediate (This Week)

1. **Folder renaming:**
   - ✅ Renamed `notes/ml/` → `notes/01-ml/`
   - ✅ Renamed `notes/math_under_the_hood/` → `notes/00-math_under_the_hood/`
   - ✅ Updated all cross-references

2. **Track completion notes:**
   - ✅ Track 2 (Agentic AI): All 11 chapters complete including advanced patterns
   - ✅ Track 5 (AI Infrastructure): All 10 chapters complete — GPU architecture through feature stores
   - ✅ Track 6 (DevOps Fundamentals): All 8 chapters complete — Docker through security

### Short-term (Next 2-4 Weeks)

**Focus:** Refinement and documentation updates

1. **Track review and polish:**
   - Review completed Track 2 Ch.11 content
   - Review completed Track 5 Ch.6-10 content
   - Review completed Track 6 Ch.1-8 content
   - Update cross-references and integration notes

2. **Documentation cleanup:**
   - ✅ Removed completed plan.md files (ai/, ai_infrastructure/, devops_fundamentals/)
   - Update track READMEs with completion status
   - Verify all cross-track dependencies documented

### Long-term (Next 2-3 Months)

**Focus:** Portfolio projects and cross-track integration

1. **Portfolio projects:**
   - Multi-agent RAG system (Track 2 + Track 3)
   - Local creative studio (Track 4 + Track 5)
   - Self-hosted LLM platform (Track 5 + Track 6)
   - End-to-end ML pipeline (Track 1 + Track 5 + Track 6)

2. **Cross-track integration examples:**
   - Deploy Track 2 agents with Track 6 infrastructure
   - Optimize Track 4 models with Track 5 techniques
   - Monitor Track 1 models with Track 6 observability

---

## Success Criteria (Portfolio-Wide)

**By completion, you can:**
- ✅ Implement any ML algorithm from scratch (Track 1)
- ✅ Build production RAG + tool-calling agents (Track 2)
- ✅ Design multi-agent systems with proper protocols (Track 3)
- ✅ Generate images/video/audio locally (Track 4)
- ✅ Optimize and deploy ML at scale (Track 5)
- ✅ Deploy and monitor any application (Track 6)
- ✅ Pass AI/ML technical interviews
- ✅ Build end-to-end AI products from scratch

**Portfolio projects to showcase:**
1. Multi-agent RAG system (Track 2 + Track 3)
2. Local creative studio (Track 4 + Track 5)
3. Self-hosted LLM platform (Track 5 + Track 6)
4. End-to-end ML pipeline (Track 1 + Track 5 + Track 6)

---

## Notes

- **Numerical prefixes:** `00-` = prerequisites, `01-` = first core track, etc.
- **Running examples:** Every track has a consistent running example (PizzaBot, OrderFlow, PixelSmith, InferenceBase)
- **Animation-first:** New content (Track 2 Ch.11) emphasizes rich animations
- **Local-first:** All tracks run on consumer hardware (no cloud GPU required for learning)
- **Production-ready:** Every track includes deployment, monitoring, optimization
- **Interview-aligned:** Every chapter has Interview Checklist

**Last updated:** April 26, 2026  
**Maintainer:** Portfolio owner  
**Next review:** After Track 2 Ch.11 + Track 5 Ch.9-10 + Track 6 Ch.1-4 completion
