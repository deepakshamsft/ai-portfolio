# Interview Guide

> The single consolidated interview-prep entry point for everything in this portfolio. Start here the week before any AI / ML / infra interview.

This guide is an **index over interview-relevant material that already lives next to the source notes** (so the answers stay in sync with the long-form chapters), plus one rapid-fire Agentic AI Q&A primer hosted directly here.

---

## 1 · Rapid-Fire Agentic AI Q&A

The fastest single-file pass over the AI track.

| File | What it covers |
|---|---|
| [AgenticAI.md](./AgenticAI.md) | Crisp Q&A across CoT, ReAct, RAG, Vector DBs, Frameworks, Evaluation, Fine-tuning, Safety, Cost & Latency |
| [notebook.ipynb](./notebook.ipynb) | Companion notebook — runs a few of the primer's examples end-to-end |

---

## 2 · Per-Track Interview Checklists

Every chapter in the curriculum ends with a **3-column checklist** (Must Know / Likely Asked / Trap to Avoid) and, where applicable, a Q&A section. Use this table as a jump map.

### Machine Learning (`../ML/`)
The 19-chapter ML track has an `## 9 · Interview Checklist` section in every chapter README.

| # | Chapter | Checklist |
|---|---|---|
| 1 | [Linear Regression](../ML/ch01-linear-regression/README.md#9--interview-checklist) | MSE, R², gradient descent |
| 2 | [Logistic Regression](../ML/ch02-logistic-regression/README.md#9--interview-checklist) | Sigmoid, BCE, precision/recall |
| 3 | [The XOR Problem](../ML/ch03-xor-problem/README.md#9--interview-checklist) | Why linear models fail |
| 4 | [Neural Networks](../ML/ch04-neural-networks/README.md#9--interview-checklist) | Activations, init, UAT |
| 5 | [Backprop & Optimisers](../ML/ch05-backprop-optimisers/README.md#9--interview-checklist) | Chain rule, SGD → Adam |
| 6 | [Regularisation](../ML/ch06-regularisation/README.md#9--interview-checklist) | L1/L2, dropout, early stopping |
| 7 | [CNNs](../ML/ch07-cnns/README.md#9--interview-checklist) | Convolution, pooling, ResNet |
| 8 | [RNNs / LSTMs / GRUs](../ML/ch08-rnns-lstms/README.md#9--interview-checklist) | Vanishing gradients, gates |
| 9 | [Metrics](../ML/ch09-metrics/README.md#9--interview-checklist) | AUC-ROC vs AUC-PR, RMSE vs MAE |
| 10 | [Classical Classifiers](../ML/ch10-classical-classifiers/README.md#9--interview-checklist) | Trees, KNN, Gini |
| 11 | [SVM & Ensembles](../ML/ch11-svm-ensembles/README.md#9--interview-checklist) | Max-margin, kernel trick, XGBoost |
| 12 | [Clustering](../ML/ch12-clustering/README.md#9--interview-checklist) | K-Means, DBSCAN, HDBSCAN |
| 13 | [Dimensionality Reduction](../ML/ch13-dimensionality-reduction/README.md#9--interview-checklist) | PCA, t-SNE, UMAP |
| 14 | [Unsupervised Metrics](../ML/ch14-unsupervised-metrics/README.md#9--interview-checklist) | Silhouette, ARI |
| 15 | [MLE & Loss Functions](../ML/ch15-mle-loss-functions/README.md#9--interview-checklist) | Derive MSE/CE from MLE |
| 17 | [Sequences to Attention](../ML/ch17-sequences-to-attention/README.md#9--interview-checklist) | Soft dictionary lookup |

### AI Infrastructure (`../AIInfrastructure/`)
- [AI Infrastructure — Track Interview Checklist](../AIInfrastructure/README.md#interview-checklist--the-track-in-90-seconds)
- Each subsection (`GPUArchitecture/`, `MemoryAndComputeBudgets/`, `QuantizationAndPrecision/`, `ServingFrameworks/`, `InferenceOptimization/`, etc.) has its own checklist embedded in the topic note.

### Multi-Agent AI (`../MultiAgentAI/`)
- [Shared Memory — Interview Questions](../MultiAgentAI/SharedMemory/README.md#interview-questions)
- [Trust & Sandboxing — Interview Questions](../MultiAgentAI/TrustAndSandboxing/README.md#interview-questions)
- Other chapters (MessageFormats, MCP, A2A, EventDrivenAgents, AgentFrameworks) have inline "Interview Checklist" tables.

### Agentic AI (`../AI/`)
- [AgenticAI.md](./AgenticAI.md) is the consolidated Q&A for the whole AI track.
- Each core note also has a companion `_Supplement.md` whose final section contains domain-specific interview Q&A.

### Multimodal AI (`../MultimodalAI/`)
- Each chapter README ends with key-question prompts for diffusion, CLIP, ViT, schedulers, guidance, and evaluation metrics.

---

## 3 · Recommended Pre-Interview Path (2–4 hours)

```
1. AgenticAI.md                        ← rapid-fire pass; flag weak spots
2. ML chapter checklists 1–9, 17–18   ← skim the 3-column tables for any gap
3. AIInfrastructure track checklist    ← serving + quantisation + KV cache
4. MultiAgentAI MCP/A2A checklists     ← protocol-level questions
5. Re-read AgenticAI.md cold           ← second pass; answer aloud
```

If you only have one hour, do steps 1 and 5.
