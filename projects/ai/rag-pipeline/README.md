# RAG Pipeline — Runnable Project

> **Status: placeholder.** This document defines the scope and acceptance criteria. Implementation is pending.

---

## What This Project Does

End-to-end Retrieval-Augmented Generation pipeline in ~300 lines of Python. No hosted services required — everything runs locally.

**Pipeline stages:**

```
Document(s)
    └── Chunking  (sliding window, configurable size + overlap)
        └── Embedding  (sentence-transformers, local model)
            └── Vector index  (FAISS, in-memory)
                └── Query  (embed question → ANN search → retrieve top-k chunks)
                    └── Generation  (LLM call with retrieved context)
                        └── Evaluation  (RAGAS metrics: faithfulness, answer relevance, context precision, context recall)
```

---

## Files (to be created)

```
rag-pipeline/
├── README.md                ← this file
├── requirements.txt         ← dependencies
├── ingest.py                ← chunk + embed + index a document
├── query.py                 ← query the index and call the LLM
├── evaluate.py              ← run RAGAS on a test set of Q&A pairs
├── data/
│   └── sample.txt           ← sample document (public domain text)
└── eval/
    └── test_set.json        ← 20-question golden test set for RAGAS
```

---

## Acceptance Criteria

- [ ] `python ingest.py --file data/sample.txt` builds a FAISS index and saves it to disk
- [ ] `python query.py --question "..."` retrieves context and prints an LLM-generated answer
- [ ] `python evaluate.py` runs RAGAS on `eval/test_set.json` and prints a metric table
- [ ] Works with a free LLM API (Groq free tier — no credit card required — or local Ollama)
- [ ] No cloud vector DB required — FAISS local only
- [ ] Total runtime < 2 minutes on CPU for the 20-question eval set

---

## Dependencies (planned)

```
sentence-transformers   # local embedding model (no API key)
faiss-cpu               # local ANN index
langchain               # chunking utilities
langgraph               # agent orchestration — RAG modelled as a graph of nodes
langchain-groq          # Groq LLM backend (free tier, no credit card)
ragas                   # evaluation framework
# alternative LLM: ollama (fully local, zero API key)
```

---

## Learning Connection

This project makes the following AI notes executable:

| Note | What this project demonstrates |
|------|-------------------------------|
| [RAGAndEmbeddings.md](../../notes/AI/RAGAndEmbeddings.md) | Chunking, embedding, retrieval pipeline |
| [VectorDBs.md](../../notes/AI/VectorDBs.md) | FAISS as an IVF-flat index with flat brute-force fallback |
| [EvaluatingAISystems.md](../../notes/AI/EvaluatingAISystems.md) | RAGAS metrics running against a real pipeline |
| [CostAndLatency.md](../../notes/AI/CostAndLatency.md) | Input token count visible in `query.py` output |

---

## Framework Choice — LangGraph

The agent is built with **LangGraph** — the industry-standard framework for stateful, graph-based agent orchestration. The RAG pipeline maps cleanly to a graph:

```
ingest_node → retrieve_node → grade_node → generate_node
```

This structure makes it straightforward to add human-in-the-loop approval, streaming responses, and persistent memory without restructuring the codebase — which is exactly what `ReActAndSemanticKernel_Supplement.md` covers in theory. No proprietary model lock-in: swap the LLM backend by changing one line.

## Deployment Note

Once working locally, deploy to:
- **Hugging Face Spaces** (free CPU tier, `gradio` or `fastapi` runtime — good portfolio visibility, no credit card)
- **Fly.io** (free tier, better cold-start story than HF Spaces for REST API demos)
- **Railway / Render** (free hobby tier)
