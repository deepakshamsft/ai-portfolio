# AI Portfolio — Notes

A personal learning library covering machine learning foundations and modern AI engineering. Five core tracks take you from GPU silicon to deployed multi-agent systems, with additional supporting collections for math, interview prep, and archives.

---

## Who This Is For

**Target audience:** Software engineers and developers who write code for a living but have little or no prior ML or AI background. You are comfortable reading Python, you have used an API before, and you know what a matrix is — but you have never trained a model or built a RAG pipeline from scratch.

This is not a course for data scientists or academic researchers. It is a practitioner's curriculum designed to answer the question a working engineer actually asks: *"I know how to build software — now how do I understand and build AI systems?"*

### Prerequisites

| Prerequisite | Level needed |
|---|---|
| Python | Comfortable reading and writing Python — functions, classes, list comprehensions |
| Linear algebra | Know what a matrix multiplication is; you do not need to be fluent in proofs |
| Statistics | Mean, variance, probability — high-school level is enough to start |
| APIs / HTTP | Have called a REST API before; used JSON |
| Command line | Can navigate directories, run scripts, activate a virtual environment |

**You do not need:** prior ML experience, a GPU, a math degree, or familiarity with PyTorch/TensorFlow before you start.

### What Makes This Different

Every ML chapter derives the math from scratch before using it. All chapters in the Regression topic use the same California Housing dataset, so the delta between chapters is the concept, not a new dataset to understand. Every note ends with an **Interview Checklist** (Must Know / Likely Asked / Trap to Avoid). Every core AI note has a companion `_Supplement.md` for production depth. Every notebook runs on a stock developer laptop — no A100, no cloud GPU budget required.

---

## Nomenclature

| Term | Scope | Example |
|------|-------|---------|
| **Topic** | A folder directly under `notes/`. Covers a broad domain. | `ML/`, `AI/`, `MultiAgentAI/` |
| **Chapter** | A leaf-level folder inside a topic (excluding utility folders like `GenScripts/` and `img/`). Contains a `README.md` + `notebook.ipynb` covering one concept. | `AI/PromptEngineering/`, `ML/01-Regression/ch04-regularization/` |

Some topics have a flat chapter layout (e.g. `AI/PromptEngineering/`), while others group chapters under sub-topics (e.g. `ML/01-Regression/ch01-linear-regression/`). Either way, the leaf folder with a README + notebook is a **chapter**.

Utility folders that appear alongside chapters:

- `gen_scripts/` — standalone scripts that generate diagrams or data (not a chapter).
- `img/` — images produced by notebooks or `gen_scripts` (not a chapter).

---

## Repository Structure

```
notes/
├── AI/               ← Agentic AI: reasoning, retrieval, orchestration (+ notebooks)
├── AIInfrastructure/ ← GPU hardware to production serving platforms (+ notebooks)
├── Archived/         ← Historical HTML/PDF chronicles and archived storyboard assets
├── InterviewGuides/  ← Consolidated interview prep — rapid-fire Q&A + checklist index
├── ML/               ← Machine Learning: topics grouped by domain (Regression, Classification, …)
├── MultiAgentAI/     ← Multi-agent protocols and coordination patterns (+ notebooks)
├── MultimodalAI/     ← Diffusion, CLIP, vision transformers, text-to-video (+ notebooks)
├── MathUnderTheHood/ ← Math foundations: linear & non-linear algebra, calculus, 1-D optimisation, matrices, gradients & chain rule, probability
```

---

## Track 1 — Machine Learning (`ML/`)

A bottom-up curriculum organised by topic. Each topic groups chapters that build on each other. Every chapter has a technical README and a runnable Jupyter notebook.

> See [ML/AUTHORING_GUIDE.md](ML/AUTHORING_GUIDE.md) for the chapter authoring guide and build tracker.

### How We Got Here — A Short History of Machine Learning

The chapters below are not in arbitrary order. They follow the actual historical sequence in which each idea was invented, frustrated, and then rescued by the next one. **The detailed timeline now lives in each chapter's own prelude** — every ML chapter opens with a *"The story"* blockquote that names the people, dates, and tensions behind that specific idea. The big-picture arc across the portfolio tracks is summarised in the era table at the top of the [repo root README](../README.md#how-the-tracks-fit-together--the-historical-arc).

**The through-line in one paragraph:** Linear regression established the foundations (gradient descent, loss minimisation). Multiple features and polynomial expansion added expressiveness but risked overfitting. Regularisation (Ridge/Lasso) tamed complexity. Classification introduced the Perceptron, which failed at XOR (1969), motivating hidden layers and neural networks. Backprop made training possible; CNNs added spatial priors; RNNs added memory. When neural nets stalled in the 1990s, classical methods (SVMs, decision trees, ensembles) carried the field. Unsupervised learning (clustering, PCA) matured. Attention scaled into the Transformer — the foundation the rest of this portfolio stands on.

> Want to feel backprop in your hands? Spend ten minutes on the [**TensorFlow Playground**](https://playground.tensorflow.org/) — a browser-based neural network trainer that animates every weight, activation, and decision boundary as you tweak the architecture.

**Setup:** run the single uber-setup from the repo root — it installs everything (ML, AIInfrastructure, MultiAgentAI) and registers all Jupyter kernels:
```powershell
# Windows
.\scripts\setup.ps1
# macOS / Linux
bash scripts/setup.sh
```

| Topic | Chapters | Domain |
|-------|----------|--------|
| [01-Regression](ML/01-Regression/) | Linear → Multiple → Polynomial → Regularisation → Metrics → Hyperparameter Tuning | Continuous prediction, California Housing, $70k→$32k MAE |
| [02-Classification](ML/02-Classification/) | Logistic Regression → Classical Classifiers → Metrics → SVM → Hyperparameter Tuning | Discrete prediction, decision boundaries |
| [03-NeuralNetworks](ML/03-NeuralNetworks/) | XOR → Dense Nets → Backprop → Regularisation → CNNs → RNNs → MLE → TensorBoard → Attention → Transformers | Deep learning, from Perceptron to Transformer |
| [04-RecommenderSystems](ML/04-RecommenderSystems/) | Fundamentals → Collaborative Filtering → Matrix Factorization → Neural CF → Hybrid Systems → Cold Start/Production | Personalization and recommendation pipelines |
| [05-AnomalyDetection](ML/05-AnomalyDetection/) | Statistical Methods → Isolation Forest → Autoencoders → One-Class SVM → Ensemble Anomaly → Production | Imbalanced anomaly detection and fraud patterns |
| [06-ReinforcementLearning](ML/06-ReinforcementLearning/) | MDPs → Dynamic Programming → Q-Learning → DQN → Policy Gradients → Modern RL | Theory-first reinforcement learning fundamentals |
| [07-UnsupervisedLearning](ML/07-UnsupervisedLearning/) | Clustering → Dimensionality Reduction → Unsupervised Metrics | K-Means, PCA, t-SNE, silhouette |
| [08-EnsembleMethods](ML/08-EnsembleMethods/) | Ensembles | Bagging, boosting, XGBoost |

---

## Track 2 — Agentic AI (`AI/`)

Deep-dive notes explaining how LLMs become agents — from token prediction through tool use, retrieval, and orchestration. Running example: **Mamma Rosa's PizzaBot**.

| Document | What it covers |
|---|---|
| [AIPrimer.md](AI/AIPrimer.md) | Entry point — PizzaBot running example, conceptual arc, document map, and reading paths |
| [LLMFundamentals/](AI/LLMFundamentals/) | BPE tokenisation, pretraining → SFT → RLHF, temperature, context windows |
| [PromptEngineering/](AI/PromptEngineering/) | System prompts, few-shot, structured output, prompt injection |
| [CoTReasoning/](AI/CoTReasoning/) | Chain-of-Thought, hidden reasoning tokens, Self-Consistency, Tree of Thoughts |
| [RAGAndEmbeddings/](AI/RAGAndEmbeddings/) | Transformer encoders, contrastive training, chunking, full RAG pipeline |
| [VectorDBs/](AI/VectorDBs/) | ANN index types (HNSW, IVF, DiskANN), distance metrics, production architecture |
| [ReActAndSemanticKernel/](AI/ReActAndSemanticKernel/) | ReAct loop, LangChain vs Semantic Kernel, LangGraph, Plan-and-Execute |
| [EvaluatingAISystems/](AI/EvaluatingAISystems/) | RAGAS metrics, LLM-as-judge, hallucination detection, pipeline evaluation |
| [FineTuning/](AI/FineTuning/) | When to fine-tune vs RAG vs prompting, LoRA math, QLoRA |
| [SafetyAndHallucination/](AI/SafetyAndHallucination/) | Hallucination types, mitigation stack, jailbreaks, alignment failures |
| [CostAndLatency/](AI/CostAndLatency/) | Token budgets, model cost tiers, KV caching, streaming |
| [InterviewGuides/](InterviewGuides/) | Consolidated interview prep — rapid-fire Q&A plus index of every per-chapter Interview Checklist across all tracks |

Every core note has a companion `_Supplement.md` for production-depth details. Read the core note first.

---

## Track 3 — Multi-Agent AI (`MultiAgentAI/`)

7-chapter track on protocols and coordination patterns for multi-agent systems. Running scenario: **OrderFlow**, a B2B purchase-order automation platform.

> → [MultiAgentAI/README.md](MultiAgentAI/README.md) for the full reading map and setup instructions.

**Setup:** use the single uber-setup from the repo root — it already installs all MultiAgentAI dependencies and registers the `multi-agent-ai` kernel:
```powershell
# Windows
.\scripts\setup.ps1
# macOS / Linux
bash scripts/setup.sh
```

| Chapter | What it covers |
|---|---|
| [MessageFormats/](MultiAgentAI/MessageFormats/) | OpenAI message envelope, token counting, handoff strategies, context trimming |
| [MCP/](MultiAgentAI/MCP/) | Model Context Protocol — JSON-RPC 2.0, Resources/Tools/Prompts, transport options |
| [A2A/](MultiAgentAI/A2A/) | Agent-to-Agent protocol — Agent Cards, task lifecycle, SSE streaming, MCP+A2A composition |
| [EventDrivenAgents/](MultiAgentAI/EventDrivenAgents/) | Pub/sub bus, DLQ, correlation/causation IDs, idempotency, fan-out/fan-in |
| [SharedMemory/](MultiAgentAI/SharedMemory/) | Blackboard pattern, write-once guards, compare-and-swap, checkpoint/resume |
| [TrustAndSandboxing/](MultiAgentAI/TrustAndSandboxing/) | Prompt injection, output schema validation, HMAC signing, timing attacks |
| [AgentFrameworks/](MultiAgentAI/AgentFrameworks/) | LangGraph StateGraph, AutoGen multi-agent debate, Semantic Kernel, framework comparison |

---

## Track 4 — Multimodal AI (`MultimodalAI/`)

13-chapter track on generative image and video models, plus text-to-audio. Running example: **PixelSmith**, a local AI-powered creative studio that must run on a stock developer laptop.

> → [MultimodalAI/README.md](MultimodalAI/README.md) for the full reading map.

| Chapter | What it covers |
|---|---|
| [MultimodalFoundations/](MultimodalAI/MultimodalFoundations/) | Signals → tensors → tokens; patch embeddings; cross-modal alignment |
| [VisionTransformers/](MultimodalAI/VisionTransformers/) | ViT architecture, patch tokenisation, CLS token, attention maps |
| [CLIP/](MultimodalAI/CLIP/) | Contrastive pre-training, zero-shot classification, text-image retrieval |
| [DiffusionModels/](MultimodalAI/DiffusionModels/) | DDPM forward/reverse process, noise schedules, score matching |
| [LatentDiffusion/](MultimodalAI/LatentDiffusion/) | VAE latent space, Stable Diffusion architecture, CFG |
| [Schedulers/](MultimodalAI/Schedulers/) | DDIM, DPM-Solver, Euler-a — speed vs quality tradeoffs |
| [GuidanceConditioning/](MultimodalAI/GuidanceConditioning/) | Classifier-free guidance, ControlNet, img2img, inpainting |
| [TextToImage/](MultimodalAI/TextToImage/) | End-to-end prompt → pixel pipeline, prompt engineering for images |
| [TextToVideo/](MultimodalAI/TextToVideo/) | Temporal attention, video diffusion, consistency across frames |
| [MultimodalLLMs/](MultimodalAI/MultimodalLLMs/) | Vision encoders in LLMs, visual question answering, GPT-4V patterns |
| [GenerativeEvaluation/](MultimodalAI/GenerativeEvaluation/) | FID, IS, CLIP score, human preference alignment |
| [LocalDiffusionLab/](MultimodalAI/LocalDiffusionLab/) | Running Stable Diffusion locally — memory optimisation, quantisation |
| [AudioGeneration/](MultimodalAI/AudioGeneration/) | Text-to-speech, local synthesis, vocoder stacks — CPU-first quick wins |

---

## Track 5 — AI Infrastructure (`AIInfrastructure/`)

5 implemented chapters (with additional chapters planned) from GPU silicon through inference optimization. Running scenario: **InferenceBase**, a startup evaluating whether to self-host Llama-3-8B instead of paying $80k/month in API bills.

> → [AIInfrastructure/README.md](AIInfrastructure/README.md) for the full reading map.

| Chapter | What it covers |
|---|---|
| [GPUArchitecture/](AIInfrastructure/GPUArchitecture/) | CUDA cores, tensor cores, VRAM, memory bandwidth, roofline model |
| [MemoryAndComputeBudgets/](AIInfrastructure/MemoryAndComputeBudgets/) | VRAM estimation: parameters, KV cache, optimizer states, activations |
| [QuantizationAndPrecision/](AIInfrastructure/QuantizationAndPrecision/) | FP16/BF16/INT8/INT4, GPTQ, AWQ, perplexity vs compression tradeoffs |
| [ParallelismAndDistributedTraining/](AIInfrastructure/ParallelismAndDistributedTraining/) | Data/tensor/pipeline parallelism, ZeRO stages, FSDP |
| [InferenceOptimization/](AIInfrastructure/InferenceOptimization/) | KV cache, speculative decoding, flash attention, kernel fusion |
| *(planned, not yet in tree)* Serving Frameworks | vLLM, TensorRT-LLM, TGI — continuous batching, PagedAttention |
| *(planned, not yet in tree)* Networking & Cluster Architecture | InfiniBand, NVLink, RDMA, collective ops (AllReduce, AllGather) |
| *(planned, not yet in tree)* MLOps & Experiment Management | MLflow, W&B, experiment tracking, model registry, CI for ML |
| *(planned, not yet in tree)* Production AI Platform | SLOs, autoscaling, shadow deployment, cost monitoring |
| *(planned, not yet in tree)* Cloud AI Infrastructure | Azure/AWS/GCP GPU offerings, spot instances, cost vs throughput |

---

## Projects (`../projects/`)

Working Python experiments that accompany the theory.

| Project | What it does |
|---|---|
| [`projects/ml/linear-regression/`](../projects/ml/linear-regression/) | End-to-end linear regression pipeline: data loading, model fitting, evaluation metrics, sklearn and custom implementations |
| [`projects/ai/rag-pipeline/`](../projects/ai/rag-pipeline/) | RAG pipeline implementation — ingestion, embedding, retrieval, reranking |

---

## How to Consume This Content — Reading Paths

### Path A — Interview Prep (2–4 hours)

```
1. InterviewGuides/                     ← single consolidated interview prep entry point
2. AI/AIPrimer.md                      ← understand the agentic systems architecture (Part 2)
3. ML/AUTHORING_GUIDE.md               ← skim Chapter Summaries for ML concepts
4. MultiAgentAI/README.md              ← multi-agent protocol interview checklist
```

### Path B — AI Engineering Deep Dive (~10–14 hours)

```
Step 1 — Reasoning layer
  → AI/CoTReasoning/
  → AI/CoTReasoning/CoTReasoning_Supplement.md

Step 2 — Knowledge layer
  → AI/RAGAndEmbeddings/
  → AI/RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md
  → AI/VectorDBs/

Step 3 — Orchestration layer
  → AI/ReActAndSemanticKernel/
  → AI/ReActAndSemanticKernel/ReActAndSemanticKernel_Supplement.md

Step 4 — Multi-agent
  → MultiAgentAI/MessageFormats/ → MCP/ → A2A/ → AgentFrameworks/

Step 5 — Synthesis
  → InterviewGuides/              (now reads as a self-test)
```

### Path C — ML from Scratch (~40–50 hours)

```
0. Math on-ramp (skip if `ŷ = wx + b`, gradients, and matrix multiply already feel like tools):
   MathUnderTheHood/ ch01 → ch07 — knuckleball free-kick thread, README + notebook each
1. Run: .\scripts\setup.ps1  (Windows)  or  bash scripts/setup.sh  (macOS/Linux)
2. Work through ML topics in order: 01-Regression → 02-Classification → 03-NeuralNetworks → 07-UnsupervisedLearning → 08-EnsembleMethods (then 04/05/06 as specialization tracks)
3. After Regression + Classification fundamentals, you have enough ML to start Path B in parallel
```

### Path D — Multimodal & Generative AI (~12–16 hours)

```
Prerequisite: Path B Step 2 (transformers, embeddings)

MultimodalAI/MultimodalFoundations/
→ MultimodalAI/VisionTransformers/
→ MultimodalAI/CLIP/
→ MultimodalAI/DiffusionModels/
→ MultimodalAI/LatentDiffusion/
→ MultimodalAI/Schedulers/
→ MultimodalAI/TextToImage/
→ MultimodalAI/LocalDiffusionLab/
```

### Path E — Infrastructure & Production (~8–12 hours)

```
Prerequisite: any track above (context for why infrastructure decisions matter)

AIInfrastructure/GPUArchitecture/
→ AIInfrastructure/MemoryAndComputeBudgets/
→ AIInfrastructure/QuantizationAndPrecision/
→ AIInfrastructure/ParallelismAndDistributedTraining/
→ AIInfrastructure/InferenceOptimization/
→ planned: Serving Frameworks → Networking & Cluster Architecture → MLOps & Production Platform
```

---

### Cross-track connections

| From | To | Connection |
|---|---|---|
| ML Ch.4 Neural Networks | AI/RAGAndEmbeddings | Transformer encoders are neural networks — the same math |
| ML Ch.5 Backprop | AI/RAGAndEmbeddings | Contrastive learning (InfoNCE) is trained with the same gradient machinery |
| ML Ch.8 RNNs/LSTMs | ML Ch.17 Sequences to Attention | LSTMs motivate *why* attention was invented; Ch.17 introduces attention without transformers |
| ML Ch.17 Sequences to Attention | ML Ch.18 Transformers | Soft-lookup intuition → learned Q/K/V projections, multi-head, positional encoding |
| ML Ch.18 Transformers | AI track (all) | Load-bearing bridge — read before the AI track |
| ML Ch.12 Clustering | AI/VectorDBs | HDBSCAN discovers topic clusters in a vector index |
| AI/ReActAndSemanticKernel | MultiAgentAI/ | Multi-agent is an extension of single-agent — not a replacement |
| AI/RAGAndEmbeddings | MultimodalAI/CLIP | CLIP uses the same contrastive training as text embedding models |
| AIInfrastructure/InferenceOptimization | MLOps/Production | Inference-level throughput and latency constraints set the floor on what your SLOs can guarantee |

---

## Getting Started

```bash
# Clone
git clone <repo-url>
cd ai-portfolio

# Launch the full dev environment (installs everything into a .venv)
# Windows
.\scripts\setup.ps1
# Optional: add --enable-slm-assistant to install the Kilo Code + Ollama bundle
# Optional: add --enable-mkdocs-server to launch the local MkDocs docs server

# macOS / Linux
bash scripts/setup.sh
# Optional: add --enable-slm-assistant to install the Kilo Code + Ollama bundle
# Optional: add --enable-mkdocs-server to launch the local MkDocs docs server
```

The single uber setup script creates a `.venv` at the repo root, installs the full AI/ML package stack used across every track (ML, AI, MultiAgentAI, MultimodalAI, AIInfrastructure), registers all Jupyter kernels (`ai-ml-dev`, `ml-notes`, `ai-infrastructure`, `multi-agent-ai`), and starts Jupyter Lab. Pass `--enable-slm-assistant` if you also want VS Code + the **Kilo Code** extension wired to a local Ollama-served DeepSeek-R1 model. Pass `--enable-mkdocs-server` if you want the local MkDocs docs server.
