# AI Portfolio — Notes

A personal learning library covering machine learning foundations and modern AI engineering. Five tracks take you from GPU silicon to deployed multi-agent systems.

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

Every ML chapter derives the math from scratch before using it. All 19 ML chapters use the same California Housing dataset from Ch.1 through Ch.19, so the delta between chapters is the concept, not a new dataset to understand. Every note ends with an **Interview Checklist** (Must Know / Likely Asked / Trap to Avoid). Every core AI note has a companion `_Supplement.md` for production depth. Every notebook runs on a stock developer laptop — no A100, no cloud GPU budget required.

---

## Repository Structure

```
notes/
├── AI/               ← Agentic AI: reasoning, retrieval, orchestration (+ notebooks)
├── AIInfrastructure/ ← GPU hardware to production serving platforms (+ notebooks)
├── Chronicles/       ← Manga storyboard plan for the Pitch Chronicles arc
├── InterviewGuide/   ← Consolidated interview prep — rapid-fire Q&A + checklist index
├── ML/               ← Machine Learning: 19 chapters, each a README + notebook
├── MultiAgentAI/     ← Multi-agent protocols and coordination patterns (+ notebooks)
├── MultimodalAI/     ← Diffusion, CLIP, vision transformers, text-to-video (+ notebooks)
└── PreRequisites/    ← Math foundations: linear algebra, calculus, probability
```

---

## Track 1 — Machine Learning (`ML/`)

A 19-chapter bottom-up curriculum. Every chapter has a technical README and a runnable Jupyter notebook using the **California Housing dataset** throughout.

> See [ML/AUTHORING_GUIDE.md](ML/AUTHORING_GUIDE.md) for the chapter authoring guide and build tracker.

### How We Got Here — A Short History of Machine Learning

The chapters below are not in arbitrary order. They follow the actual historical sequence in which each idea was invented, frustrated, and then rescued by the next one. **The detailed timeline now lives in each chapter's own prelude** — every ML chapter opens with a *"The story"* blockquote that names the people, dates, and tensions behind that specific idea. The big-picture arc across all six tracks is summarised in the era table at the top of the [repo root README](../README.md#how-the-tracks-fit-together--the-historical-arc).

**The through-line in one paragraph:** the Perceptron failed at XOR (1969), so we needed hidden layers ([Ch.3](ML/ch03-xor-problem/) → [Ch.4](ML/ch04-neural-networks/)). Hidden layers couldn't train until backprop was rediscovered ([Ch.5](ML/ch05-backprop-optimisers/)). Deep nets overfit, so we needed regularisation ([Ch.6](ML/ch06-regularisation/)). Vision needed spatial priors ([Ch.7](ML/ch07-cnns/)); sequences needed memory ([Ch.8](ML/ch08-rnns-lstms/)). When neural nets stalled in the 1990s, the classical branch ([Ch.10](ML/ch10-classical-classifiers/)–[Ch.11](ML/ch11-svm-ensembles/)) carried the field. Unsupervised learning matured ([Ch.12](ML/ch12-clustering/)–[Ch.14](ML/ch14-unsupervised-metrics/)). MLE gave everything a probabilistic foundation ([Ch.15](ML/ch15-mle-loss-functions/)). Instrumentation ([Ch.16](ML/ch16-tensorboard/)) and tuning ([Ch.19](ML/ch19-hyperparameter-tuning/)) became first-class when models got serious. Finally, attention ([Ch.17](ML/ch17-sequences-to-attention/)) scaled into the Transformer ([Ch.18](ML/ch18-transformers/)) — the foundation the rest of this portfolio stands on.

> Want to feel backprop in your hands? Before opening Ch.5, spend ten minutes on the [**TensorFlow Playground**](https://playground.tensorflow.org/) — a browser-based neural network trainer that animates every weight, activation, and decision boundary as you tweak the architecture. It is the single best intuition-builder for Chapters 3–6. For Ch.18, the companion animation [`multihead_attention.gif`](ML/ch18-transformers/img/multihead_attention.gif) walks through scaled dot-product attention one query at a time and replays it across three heads that learn different relationship patterns.

**Depth labels** — a realism tag next to each chapter, so you can plan pacing honestly:

- **F — Foundational**: accessible with the stated prerequisites; core track.
- **D — Deep**: expect to spend ~2× foundational time and re-read.
- **R — Reference**: consult as needed, not meant cover-to-cover first pass.

**Setup:** run the single uber-setup from the repo root — it installs everything (ML, AIInfrastructure, MultiAgentAI) and registers all Jupyter kernels:
```powershell
# Windows
.\scripts\setup.ps1
# macOS / Linux
bash scripts/setup.sh
```

| # | Chapter | Depth | Core concept |
|---|---|---|---|
| 1 | [Linear Regression](ML/ch01-linear-regression/) | F | `ŷ = Wᵀx + b`, MSE, gradient descent, R² |
| 2 | [Logistic Regression](ML/ch02-logistic-regression/) | F | Sigmoid, binary cross-entropy, precision/recall |
| 3 | [The XOR Problem](ML/ch03-xor-problem/) | F | Why linear models fail, Universal Approximation Theorem |
| 4 | [Neural Networks](ML/ch04-neural-networks/) | F | Dense layers, activations (ReLU/Softmax), Xavier/He init |
| 5 | [Backprop & Optimisers](ML/ch05-backprop-optimisers/) | D | Chain rule, SGD → Momentum → Adam, LR schedules |
| 6 | [Regularisation](ML/ch06-regularisation/) | F | L1/L2, dropout, early stopping |
| 7 | [CNNs](ML/ch07-cnns/) | F | Convolution, pooling, feature hierarchies, ResNet idea |
| 8 | [RNNs / LSTMs / GRUs](ML/ch08-rnns-lstms/) | D | Hidden state, vanishing gradient, LSTM gates |
| 9 | [Metrics Deep Dive](ML/ch09-metrics/) | F | AUC-ROC, AUC-PR, confusion matrix, RMSE vs MAE |
| 10 | [Classical Classifiers](ML/ch10-classical-classifiers/) | F | Decision Trees, KNN, Gini impurity |
| 11 | [SVM & Ensembles](ML/ch11-svm-ensembles/) | F | Max-margin, kernel trick, bagging vs boosting, XGBoost |
| 12 | [Clustering](ML/ch12-clustering/) | F | K-Means, DBSCAN, HDBSCAN |
| 13 | [Dimensionality Reduction](ML/ch13-dimensionality-reduction/) | D | PCA, t-SNE, UMAP |
| 14 | [Unsupervised Metrics](ML/ch14-unsupervised-metrics/) | F | Silhouette, Davies-Bouldin, ARI |
| 15 | [MLE & Loss Functions](ML/ch15-mle-loss-functions/) | D | Derive MSE and Cross-Entropy from maximum likelihood |
| 16 | [TensorBoard](ML/ch16-tensorboard/) | R | Instrument training with scalars, histograms, and projector |
| 17 | [From Sequences to Attention](ML/ch17-sequences-to-attention/) | F | **Bridge chapter.** Soft dictionary lookup — dot product + softmax → attention, without transformers |
| 18 | [Transformers & Attention](ML/ch18-transformers/) | D | Scaled dot-product attention, multi-head, positional encoding |
| 19 | [Hyperparameter Tuning](ML/ch19-hyperparameter-tuning/) | R | Learning rate, batch size, optimiser, init, regularisation — tune order and search strategies |

---

## Track 2 — Agentic AI (`AI/`)

Deep-dive notes explaining how LLMs become agents — from token prediction through tool use, retrieval, and orchestration. Running example: **Mamma Rosa's PizzaBot**.

| Document | What it covers |
|---|---|
| [AgenticAI_ReadingMap.md](AI/AgenticAI_ReadingMap.md) | Entry point — conceptual arc and how all documents connect |
| [AIPrimer.md](AI/AIPrimer.md) | Running example — PizzaBot: system definition, RAG corpus, tools, full ReAct trace |
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
| [InterviewGuide/](InterviewGuide/) | Consolidated interview prep — rapid-fire Q&A plus index of every per-chapter Interview Checklist across all tracks |

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

12-chapter track on generative image and video models. Running example: **PixelSmith**, a local AI-powered creative studio that must run on a stock developer laptop.

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

---

## Track 5 — AI Infrastructure (`AIInfrastructure/`)

10-chapter track from GPU silicon to production serving platforms. Running scenario: **InferenceBase**, a startup evaluating whether to self-host Llama-3-8B instead of paying $80k/month in API bills.

> → [AIInfrastructure/README.md](AIInfrastructure/README.md) for the full reading map.

| Chapter | What it covers |
|---|---|
| [GPUArchitecture/](AIInfrastructure/GPUArchitecture/) | CUDA cores, tensor cores, VRAM, memory bandwidth, roofline model |
| [MemoryAndComputeBudgets/](AIInfrastructure/MemoryAndComputeBudgets/) | VRAM estimation: parameters, KV cache, optimizer states, activations |
| [QuantizationAndPrecision/](AIInfrastructure/QuantizationAndPrecision/) | FP16/BF16/INT8/INT4, GPTQ, AWQ, perplexity vs compression tradeoffs |
| [ParallelismAndDistributedTraining/](AIInfrastructure/ParallelismAndDistributedTraining/) | Data/tensor/pipeline parallelism, ZeRO stages, FSDP |
| [ServingFrameworks/](AIInfrastructure/ServingFrameworks/) | vLLM, TensorRT-LLM, TGI — continuous batching, PagedAttention |
| [InferenceOptimization/](AIInfrastructure/InferenceOptimization/) | KV cache, speculative decoding, flash attention, kernel fusion |
| [NetworkingAndClusterArchitecture/](AIInfrastructure/NetworkingAndClusterArchitecture/) | InfiniBand, NVLink, RDMA, collective ops (AllReduce, AllGather) |
| [MLOpsAndExperimentManagement/](AIInfrastructure/MLOpsAndExperimentManagement/) | MLflow, W&B, experiment tracking, model registry, CI for ML |
| [ProductionAIPlatform/](AIInfrastructure/ProductionAIPlatform/) | SLOs, autoscaling, shadow deployment, cost monitoring |
| [CloudAIInfrastructure/](AIInfrastructure/CloudAIInfrastructure/) | Azure/AWS/GCP GPU offerings, spot instances, cost vs throughput |

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
1. InterviewGuide/                     ← single consolidated interview prep entry point
2. AI/AgenticAI_ReadingMap.md          ← understand the agentic systems architecture
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
  → InterviewGuide/              (now reads as a self-test)
```

### Path C — ML from Scratch (~40–50 hours)

```
1. Run: .\scripts\setup.ps1  (Windows)  or  bash scripts/setup.sh  (macOS/Linux)
2. Work through ML/ ch01 → ch17 in order — README first, then notebook
3. After Ch.6 you have enough ML to start Path B in parallel
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
→ AIInfrastructure/ServingFrameworks/
→ AIInfrastructure/InferenceOptimization/
→ AIInfrastructure/ProductionAIPlatform/
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
| AIInfrastructure/ServingFrameworks | MLOps/Production | Serving decisions set the floor on what your SLOs can guarantee |

---

## Getting Started

```bash
# Clone
git clone <repo-url>
cd ai-portfolio

# Launch the full dev environment (installs everything into a .venv)
# Windows
.\scripts\setup.ps1

# macOS / Linux
bash scripts/setup.sh
```

The single uber setup script creates a `.venv` at the repo root, installs the full AI/ML package stack used across every track (ML, AI, MultiAgentAI, MultimodalAI, AIInfrastructure), registers all Jupyter kernels (`ai-ml-dev`, `ml-notes`, `ai-infrastructure`, `multi-agent-ai`), installs VS Code + the **Kilo Code** extension wired to a local Ollama-served DeepSeek-R1 model, and starts the Ollama server.
