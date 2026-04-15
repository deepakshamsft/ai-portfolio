# AI Portfolio

A personal learning library covering machine learning foundations and modern AI engineering. The repo is split into two tracks — **ML** (the maths and training mechanics behind models) and **AI** (how those models become thinking, acting agents) — plus working code projects.

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

The ML track (Path C) teaches neural networks from first principles — you will derive the maths as you go. The AI track (Path B) is accessible after you can confidently explain what a function is and what an API call does.

### What Makes This Different

Most ML/AI learning resources either state results without showing where they come from, or bury you in theory without connecting it to code you'd actually write. This repo tries to do neither. Every ML chapter derives the maths from scratch before using it — you see *why* the cross-entropy loss comes from maximum likelihood before you minimise it, not afterwards. All 17 ML chapters use the same California Housing dataset from Ch.1 through Ch.17, so you can watch a bare linear regression transform step by step into a regularised neural network evaluated with a full metrics suite — the delta between chapters is the concept, not a new dataset to understand. Every note ends with an **Interview Checklist** (Must Know / Likely Asked / Trap to Avoid) so the gap between reading and interview-ready is minimal. And every core AI note has a companion `_Supplement.md` for when you want the production-depth picture beyond the fundamentals — meaning you can go as shallow or as deep as your goal requires without the core note becoming overwhelming.

---

## Repository Structure

```
ai-portfolio/
├── notes/
│   ├── AI/               ← Agentic AI: reasoning, retrieval, orchestration
│   ├── ML/               ← Machine Learning: 17 chapters, each a README + notebook
│   └── Reference/        ← Source HTML/PDF books the notes were built from
├── projects/
│   └── ml/               ← Runnable Python experiments
│       └── linear-regression/
├── scripts/
│   ├── setup.ps1         ← Windows: install deps + launch Jupyter
│   └── setup.sh          ← macOS/Linux: install deps + launch Jupyter
```

---

## Track 1 — Machine Learning (`notes/ML/`)

A 17-chapter bottom-up curriculum built from the **Neural Chronicles** reference book. Every chapter lives in its own folder with two files: a technical README and a runnable Jupyter notebook. Both follow the same structure and use the same running example — the **California Housing dataset** (predicting and classifying home values in California).

> **Status:** In progress — see [notes/ML/ML_Chronicles_BuildPlan.md](notes/ML/ML_Chronicles_BuildPlan.md) for the chapter tracker.

| # | Chapter | Core concept |
|---|---|---|
| 1 | Linear Regression | `ŷ = Wᵀx + b`, MSE, gradient descent, R² |
| 2 | Logistic Regression | Sigmoid, binary cross-entropy, precision/recall, threshold tuning |
| 3 | The XOR Problem | Why linear models fail, Universal Approximation Theorem |
| 4 | Neural Networks | Dense layers, activations (ReLU/Softmax), Xavier/He init |
| 5 | Backprop & Optimisers | Chain rule, SGD → Momentum → Adam, LR schedules |
| 6 | Regularisation | L1/L2, dropout, early stopping |
| 7 | CNNs | Convolution, pooling, feature hierarchies, ResNet idea |
| 8 | RNNs / LSTMs / GRUs | Hidden state, vanishing gradient, LSTM gates |
| 9 | Metrics Deep Dive | AUC-ROC, AUC-PR, confusion matrix, RMSE vs MAE |
| 10 | Classical Classifiers | Decision Trees, KNN, Gini impurity |
| 11 | SVM & Ensembles | Max-margin, kernel trick, bagging vs boosting, XGBoost |
| 12 | Clustering | K-Means, DBSCAN, HDBSCAN |
| 13 | Dimensionality Reduction | PCA, t-SNE, UMAP |
| 14 | Unsupervised Metrics | Silhouette, Davies-Bouldin, ARI |
| 15 | MLE & Loss Functions | Derive MSE and Cross-Entropy from maximum likelihood |
| 16 | TensorBoard | Instrument training with scalars, histograms, and projector |
| 17 | Transformers & Attention | Scaled dot-product attention, multi-head attention, positional encoding, encoder vs decoder |

---

## Track 2 — Agentic AI (`notes/AI/`)

A tightly cross-referenced set of deep-dive notes explaining how LLMs become agents — from token prediction through tool use, retrieval, and multi-agent orchestration.

| File | What it covers |
|---|---|
| [AgenticAI_ReadingMap.md](notes/AI/AgenticAI_ReadingMap.md) | Entry point — explains the conceptual arc and how all documents connect |
| [AIPrimer.md](notes/AI/AIPrimer.md) | Running example — Mamma Rosa's PizzaBot: system definition, RAG corpus, tools, and full ReAct trace used across all AI notes |
| [LLMFundamentals.md](notes/AI/LLMFundamentals/LLMFundamentals.md) | What an LLM actually is: BPE tokenisation, pretraining → SFT → RLHF, temperature, context windows |
| [PromptEngineering.md](notes/AI/PromptEngineering/PromptEngineering.md) | System prompts, few-shot, structured output, prompt injection and mitigations |
| [CoTReasoning.md](notes/AI/CoTReasoning/CoTReasoning.md) | Chain-of-Thought prompting, hidden reasoning tokens, how "predict next token" becomes "call a tool" |
| [CoTReasoning_Supplement.md](notes/AI/CoTReasoning/CoTReasoning_Supplement.md) | Advanced patterns: Self-Consistency, Tree/Graph of Thoughts, Process Reward Models, production failure modes |
| [RAGAndEmbeddings.md](notes/AI/RAGAndEmbeddings/RAGAndEmbeddings.md) | Transformer encoders, contrastive training, pooling strategies, chunking, the full RAG ingestion + query pipeline |
| [RAGAndEmbeddings_Supplement.md](notes/AI/RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) | Hybrid search, reranking, late interaction, advanced chunking strategies |
| [VectorDBs.md](notes/AI/VectorDBs/VectorDBs.md) | Why exact search fails at scale, ANN index types (HNSW, IVF, DiskANN), distance metrics |
| [VectorDBs_Supplement.md](notes/AI/VectorDBs/VectorDBs_Supplement.md) | Production architecture, filtering, quantisation, database comparison |
| [ReActAndSemanticKernel.md](notes/AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md) | ReAct loop, LangChain vs Semantic Kernel, multi-agent patterns, the "detective agency" mental model |
| [ReActAndSemanticKernel_Supplement.md](notes/AI/ReActAndSemanticKernel/ReActAndSemanticKernel_Supplement.md) | Plan-and-Execute, LangGraph, memory types, production traps |
| [EvaluatingAISystems.md](notes/AI/EvaluatingAISystems/EvaluatingAISystems.md) | RAGAS metrics, LLM-as-judge, hallucination detection, component and pipeline evaluation |
| [FineTuning.md](notes/AI/FineTuning/FineTuning.md) | When to fine-tune vs. RAG vs. prompting, LoRA math, QLoRA, practical setup |
| [SafetyAndHallucination.md](notes/AI/SafetyAndHallucination/SafetyAndHallucination.md) | Hallucination types and causes, mitigation stack, jailbreaks, alignment failures |
| [CostAndLatency.md](notes/AI/CostAndLatency/CostAndLatency.md) | Token budgets, model cost tiers, KV caching, streaming, cost estimation patterns |
| [AI_Interview_Primer.md](notes/AI/AI_Interview_Primer/AI_Interview_Primer.md) | Rapid-fire Q&A across all topics — designed for interview prep |

Every core note has a companion `_Supplement.md` that goes deeper on advanced details and production gotchas. Read the core note first, then the supplement if you want the full picture.

---

## Projects (`projects/`)

Working Python experiments that accompany the theory.

| Project | What it does |
|---|---|
| `projects/ml/linear-regression/` | End-to-end linear regression pipeline: data loading, model fitting, evaluation metrics, comparison across sklearn and custom implementations |
| `projects/ml/linear-regression/football/` | Stub — planned experiment applying the same pipeline to a football dataset |

---

## How to Consume This Content — A Roadmap

There are three sensible paths through this repo depending on your goal.

---

### Path A — Interview Prep (fastest, 2–4 hours)

Start here if you have an interview coming up and need a fast refresh across both ML and AI.

```
1. notes/AI/AgenticAI_ReadingMap.md    ← understand the architecture of agentic systems
2. notes/AI/AI_Interview_Primer.md     ← Q&A format covering every topic fast
3. notes/ML/ML_Chronicles_BuildPlan.md ← skim the Chapter Summaries section for ML concepts
```

The Interview Primer is dense by design — it covers CoT, ReAct, RAG, vector databases, and Semantic Kernel in crisp Q&A form, exactly how interviewers probe them.

---

### Path B — AI Engineering Deep Dive (~10–14 hours, read order matters)

Start here if you want to deeply understand how agentic systems work from first principles.

```
Step 1 — Reasoning layer
  → CoTReasoning.md
  → CoTReasoning_Supplement.md

Step 2 — Knowledge layer
  → RAGAndEmbeddings.md
  → RAGAndEmbeddings_Supplement.md
  → VectorDBs.md
  → VectorDBs_Supplement.md

Step 3 — Orchestration layer
  → ReActAndSemanticKernel.md
  → ReActAndSemanticKernel_Supplement.md

Step 4 — Synthesis
  → AI_Interview_Primer.md   (now reads as a self-test, not a cram session)
```

This order mirrors the three layers of an agentic system: how the LLM thinks → how it retrieves knowledge → how the surrounding software orchestrates it all.

---

### Path C — ML from Scratch (~40–50 hours with labs, chapter by chapter)

Start here if you want to build solid ML foundations with runnable code.

```
1. Run the setup script to get Jupyter running locally
   Windows:      .\notes\scripts\setup.ps1
   macOS/Linux:  bash notes/scripts/setup.sh

2. Open notes/ML/ in Jupyter (the script does this automatically)

3. Work through chapters in order: ch01 → ch02 → ... → ch17
   Each chapter = read the README first, then run the notebook

4. After Ch.6 (Regularisation) you have enough ML to start Path B alongside
```

The 17 chapters build on each other. Ch.1–4 lay the model architecture foundations. Ch.5–6 cover how training actually works. Ch.7–8 extend to images and sequences. Ch.9–14 branch into classical methods, evaluation, and unsupervised learning. Ch.15 derives loss functions from first principles. Ch.16 covers training diagnostics with TensorBoard. Ch.17 builds the transformer from scratch — the bridge into the AI track.

---

### Combining the tracks

ML and AI are not independent. Here is where they connect:

| ML chapter | AI connection |
|---|---|
| Ch.4 Neural Networks | The transformer encoder in RAGAndEmbeddings.md is a neural network — reading both together builds real intuition |
| Ch.5 Backprop & Optimisers | Contrastive learning (InfoNCE) used to train embedding models is covered in RAGAndEmbeddings.md |
| Ch.8 RNNs / LSTMs | The predecessor to transformers — understanding LSTMs makes the "why attention" story in Ch.17 land better, before moving to the AI notes |
| Ch.17 Transformers | The load-bearing bridge: read this before starting the AI track — RAGAndEmbeddings.md assumes transformer encoders throughout |
| Ch.12 Clustering | HDBSCAN appears in VectorDBs_Supplement.md as a way to discover topic clusters in a vector index |

A natural combined path: **Ch.1–6 (ML) → CoTReasoning + RAGAndEmbeddings (AI) → Ch.7–8 (ML) → Ch.17 Transformers (ML) → ReActAndSemanticKernel (AI) → Ch.9–16 (ML) → VectorDBs (AI)**.

---

## Getting Started

```bash
# Clone
git clone <repo-url>
cd ai-portfolio

# Launch the ML notebooks (installs everything into a .venv)
# Windows
.\notes\scripts\setup.ps1

# macOS / Linux
bash notes/scripts/setup.sh
```

The setup script creates a `.venv` at the repo root, installs all required packages (`numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `scipy`, `notebook`), registers the kernel, and opens Jupyter rooted at `notes/ML/`.
