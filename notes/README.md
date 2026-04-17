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

**You do not need:** prior ML experience, a GPU, a maths degree, or familiarity with PyTorch/TensorFlow before you start.

### What Makes This Different

Every ML chapter derives the maths from scratch before using it. All 19 ML chapters use the same California Housing dataset from Ch.1 through Ch.19, so the delta between chapters is the concept, not a new dataset to understand. Every note ends with an **Interview Checklist** (Must Know / Likely Asked / Trap to Avoid). Every core AI note has a companion `_Supplement.md` for production depth. Every notebook runs on a stock developer laptop — no A100, no cloud GPU budget required.

---

## Repository Structure

```
notes/
├── AI/               ← Agentic AI: reasoning, retrieval, orchestration (+ notebooks)
├── AIInfrastructure/ ← GPU hardware to production serving platforms (+ notebooks)
├── ML/               ← Machine Learning: 17 chapters, each a README + notebook
├── MultiAgentAI/     ← Multi-agent protocols and coordination patterns (+ notebooks)
├── MultimodalAI/     ← Diffusion, CLIP, vision transformers, text-to-video (+ notebooks)
├── Reference/        ← ML Chronicles + Neural Chronicles HTML/PDF reference books
└── scripts/          ← Cross-platform environment setup and notebook generation
```

---

## Track 1 — Machine Learning (`ML/`)

A 19-chapter bottom-up curriculum built from the **Neural Chronicles** reference book. Every chapter has a technical README and a runnable Jupyter notebook using the **California Housing dataset** throughout.

> See [ML/AUTHORING_GUIDE.md](ML/AUTHORING_GUIDE.md) for the chapter authoring guide and build tracker.

### How We Got Here — A Short History of Machine Learning

The chapters below are not in arbitrary order. They follow the actual historical sequence in which each idea was invented, frustrated, and then rescued by the next one. Reading the timeline once makes the curriculum feel inevitable.

| Era | Year | Breakthrough | Chapter it seeds |
|---|---|---|---|
| **Pre-ML statistics** | 1805 / 1809 | **Least squares** (Legendre, Gauss) | [Ch.1 Linear Regression](ML/ch01-linear-regression/) |
| | 1922 | **Maximum Likelihood Estimation** (R. A. Fisher) — the justification for our loss functions | [Ch.15 MLE & Loss Functions](ML/ch15-mle-loss-functions/) |
| | 1944 | **Logistic function for binary outcomes** (Berkson) | [Ch.2 Logistic Regression](ML/ch02-logistic-regression/) |
| **First neural era** | 1943 | **McCulloch–Pitts neuron** — first mathematical model of a neuron | [Ch.4 Neural Networks](ML/ch04-neural-networks/) |
| | 1958 | **Perceptron** (Rosenblatt) — first *learning* neural model | [Ch.4](ML/ch04-neural-networks/) |
| | 1969 | **"Perceptrons"** (Minsky & Papert) — proved single-layer nets cannot learn XOR | [Ch.3 The XOR Problem](ML/ch03-xor-problem/) |
| | 1969–1986 | **First AI winter** — funding collapses for over a decade | — |
| **Backprop revival** | 1974 / 1986 | **Backpropagation** (Werbos 1974; Rumelhart, Hinton & Williams 1986) | [Ch.5 Backprop & Optimisers](ML/ch05-backprop-optimisers/) |
| | 1989 | **LeNet** (LeCun) — CNN for handwritten digits | [Ch.7 CNNs](ML/ch07-cnns/) |
| | 1997 | **LSTM** (Hochreiter & Schmidhuber) — solves vanishing gradients in RNNs | [Ch.8 RNNs / LSTMs / GRUs](ML/ch08-rnns-lstms/) |
| **Statistical learning era** | 1984 | **CART decision trees** (Breiman) | [Ch.10 Classical Classifiers](ML/ch10-classical-classifiers/) |
| | 1995 | **Support Vector Machines** (Cortes & Vapnik) | [Ch.11 SVM & Ensembles](ML/ch11-svm-ensembles/) |
| | 1996 / 2001 | **AdaBoost / Random Forests** (Freund–Schapire; Breiman) | [Ch.11](ML/ch11-svm-ensembles/) |
| | 2014 | **XGBoost** (Chen & Guestrin) — gradient boosting wins Kaggle | [Ch.11](ML/ch11-svm-ensembles/) |
| **Clustering & geometry** | 1957 | **k-Means** (Lloyd, published 1982) | [Ch.12 Clustering](ML/ch12-clustering/) |
| | 1901 / 2008 | **PCA** (Pearson); **t-SNE** (van der Maaten & Hinton) | [Ch.13 Dimensionality Reduction](ML/ch13-dimensionality-reduction/) |
| | 1996 / 2018 | **DBSCAN** (Ester et al.); **UMAP** (McInnes et al.) | [Ch.12](ML/ch12-clustering/), [Ch.13](ML/ch13-dimensionality-reduction/) |
| **Second AI winter → recovery** | late 1990s | Neural nets fall out of favour; SVMs dominate | — |
| | 2006 | **Deep Belief Nets** (Hinton) — layer-wise pretraining revives deep learning | [Ch.4](ML/ch04-neural-networks/) |
| **Modern deep learning** | 2010 | **ReLU + Xavier init** make deep training stable | [Ch.4](ML/ch04-neural-networks/) |
| | 2012 | **AlexNet** wins ImageNet on two GPUs — deep learning goes mainstream | [Ch.7](ML/ch07-cnns/) |
| | 2014 | **Dropout** (Srivastava et al.) | [Ch.6 Regularisation](ML/ch06-regularisation/) |
| | 2014 | **Adam optimiser** (Kingma & Ba) | [Ch.5](ML/ch05-backprop-optimisers/) |
| | 2015 | **Batch Normalisation**; **ResNet** | [Ch.6](ML/ch06-regularisation/), [Ch.7](ML/ch07-cnns/) |
| | 2015 | **TensorBoard** ships with TensorFlow 0.6 — training becomes observable | [Ch.16 TensorBoard](ML/ch16-tensorboard/) |
| **Attention era** | 2014 | **Bahdanau attention** for seq2seq translation | [Ch.17 From Sequences to Attention](ML/ch17-sequences-to-attention/) |
| | 2015–2017 | AUC-ROC, AUC-PR become standard model-selection currency | [Ch.9 Metrics Deep Dive](ML/ch09-metrics/) |
| | 2017 | **"Attention Is All You Need"** (Vaswani et al.) — the Transformer | [Ch.18 Transformers & Attention](ML/ch18-transformers/) |
| | 2018 | **BERT / GPT** — the pretraining revolution we still live in | [Ch.18](ML/ch18-transformers/) |
| **Tuning as a discipline** | 2012 → 2019 | **Random Search** (Bergstra & Bengio); **Hyperband**; **Bayesian Optimization**; **Population-Based Training** | [Ch.19 Hyperparameter Tuning](ML/ch19-hyperparameter-tuning/) |

**The through-line:** the Perceptron failed at XOR (1969), so we needed hidden layers (Ch.3 → Ch.4). Hidden layers couldn't train until backprop was rediscovered (Ch.5). Deep nets overfit, so we needed regularisation (Ch.6). Vision needed spatial priors (Ch.7); sequences needed memory (Ch.8). When neural nets stalled in the 1990s, the classical branch (Ch.10–11) carried the field. Unsupervised learning matured (Ch.12–14). MLE gave everything a probabilistic foundation (Ch.15). Instrumentation (Ch.16) and tuning (Ch.19) became first-class when models got serious. Finally, attention (Ch.17) scaled into the Transformer (Ch.18) — the foundation the rest of this portfolio stands on.

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
| [AI_Interview_Primer/](AI/AI_Interview_Primer/) | Rapid-fire Q&A across all topics — designed for interview prep |

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
1. AI/AgenticAI_ReadingMap.md           ← understand the agentic systems architecture
2. AI/AI_Interview_Primer/              ← Q&A covering every topic in interview format
3. ML/AUTHORING_GUIDE.md                ← skim Chapter Summaries for ML concepts
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
  → AI/AI_Interview_Primer/      (now reads as a self-test)
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
| ML Ch.4 Neural Networks | AI/RAGAndEmbeddings | Transformer encoders are neural networks — the same maths |
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

The single uber setup script creates a `.venv` at the repo root, installs the full AI/ML package stack used across every track (ML, AI, MultiAgentAI, MultimodalAI, AIInfrastructure), registers all Jupyter kernels (`ai-ml-dev`, `ml-notes`, `ai-infrastructure`, `multi-agent-ai`), installs VS Code + Twinny, and starts a local Ollama server.
