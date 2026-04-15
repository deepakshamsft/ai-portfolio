# Agentic AI — How to Read This Collection

> This document is your **entry point and reading map**. It explains the conceptual arc across all notes, shows how each document connects to the others, and prescribes reading paths based on your goal.

---

## The Central Story in One Paragraph

Modern AI agents are built on a single powerful idea: **an LLM whose next-token prediction is constrained to choose from a menu of actions, and whose environment executes those actions and feeds results back as tokens — creating a planning loop**. To build that loop well, you need four layers of knowledge: (0) what an LLM actually is and how to prompt it reliably (LLMFundamentals + PromptEngineering), (1) how the LLM “thinks” internally (Chain-of-Thought and reasoning models), (2) how it sources knowledge it doesn’t have in its weights (embeddings + RAG + vector databases), and (3) how the surrounding software orchestrates the loop and scales it to production (ReAct, LangChain, Semantic Kernel, multi-agent patterns). The documents in this collection cover each layer in depth, and they deliberately cross-reference each other because the layers are not independent.

---

## The Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       AGENTIC AI SYSTEM                                  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ORCHESTRATION LAYER                            │    │
│  │  ReAct Loop · LangChain · Semantic Kernel · Multi-Agent Patterns │    │
│  │                  [ReActAndSemanticKernel.md]                      │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                               │                                           │
│          ┌────────────────────┴─────────────────────┐                   │
│          │                                           │                   │
│  ┌───────▼──────────────┐          ┌────────────────▼────────────┐      │
│  │    REASONING LAYER    │          │      KNOWLEDGE LAYER         │      │
│  │                       │          │                              │      │
│  │  How the LLM "thinks" │          │   How the agent retrieves    │      │
│  │  step by step before  │          │   facts it wasn't trained on │      │
│  │  choosing an action   │          │                              │      │
│  │                       │          │  ┌──────────────────────┐   │      │
│  │  [CoTReasoning.md]    │          │  │  Embeddings + RAG     │   │      │
│  │                       │          │  │  [RAGAndEmbeddings.md]│   │      │
│  └───────────────────────┘          │  └──────────────────────┘   │      │
│                                      │                              │      │
│                                      │  ┌──────────────────────┐   │      │
│                                      │  │  Vector Index Storage │   │      │
│                                      │  │  [VectorDBs.md]       │   │      │
│                                      │  └──────────────────────┘   │      │
│                                      └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Document Map: What Each File Covers

### Foundation Notes (read before the core notes)

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [AIPrimer.md](./AIPrimer.md) | Running example used across all notes — Mamma Rosa's PizzaBot: the system, tools, RAG corpus, and full ReAct trace | What concrete system does the AI track build toward? |
| [LLMFundamentals.md](./LLMFundamentals/LLMFundamentals.md) | What an LLM actually is: tokenisation, training stages, sampling, context windows | What is BPE? What does RLHF add? What is temperature? |
| [PromptEngineering.md](./PromptEngineering/PromptEngineering.md) | Getting reliable, structured output from LLMs | How do I write a system prompt? How do I prevent prompt injection? How do I guarantee JSON output? |

### Core Notes

| File | Layer | Purpose | Key Questions Answered |
|------|-------|---------|----------------------|
| [CoTReasoning.md](./CoTReasoning/CoTReasoning.md) | Reasoning | How LLMs plan step-by-step; the bridge from token prediction to action | How does "predict next token" become "call a tool"? What are reasoning tokens? |
| [ReActAndSemanticKernel.md](./ReActAndSemanticKernel/ReActAndSemanticKernel.md) | Orchestration | The ReAct loop; LangChain and Semantic Kernel frameworks | How do agents loop through thought/action/observe? LangChain vs. SK? |
| [RAGAndEmbeddings.md](./RAGAndEmbeddings/RAGAndEmbeddings.md) | Knowledge | How embeddings work; the full RAG ingestion and query pipeline | How is text turned into a vector? How does retrieval work at inference time? |
| [VectorDBs.md](./VectorDBs/VectorDBs.md) | Knowledge (Storage) | ANN index types; vector database architectures | How does HNSW search work? When to use DiskANN vs. IVF? |

### Enrichment Supplements (read after each core doc)

| File | Enriches | What It Adds |
|------|----------|-------------|
| [CoTReasoning_Supplement.md](./CoTReasoning/CoTReasoning_Supplement.md) | CoTReasoning.md | Advanced reasoning patterns (ToT, GoT, Reflexion, LATS), PRM vs. ORM, failure modes, budget calibration |
| [RAGAndEmbeddings_Supplement.md](./RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) | RAGAndEmbeddings.md | HyDE, FLARE, query decomposition, RAGAS evaluation, sparse vs. dense, lost-in-the-middle |
| [ReActAndSemanticKernel_Supplement.md](./ReActAndSemanticKernel/ReActAndSemanticKernel_Supplement.md) | ReActAndSemanticKernel.md | Multi-agent patterns, LangGraph, HITL, tool design, production failure modes |
| [VectorDBs_Supplement.md](./VectorDBs/VectorDBs_Supplement.md) | VectorDBs.md | Tuning recipes (HNSW/IVF params), quantization deep dive, filtering strategies, benchmarking protocol |

### Production & Operations Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [EvaluatingAISystems.md](./EvaluatingAISystems/EvaluatingAISystems.md) | How to measure RAG pipelines and agents — RAGAS, LLM-as-judge, hallucination detection | What is faithfulness? How do I evaluate a ReAct trace? What is RAGAS context precision? |
| [FineTuning.md](./FineTuning/FineTuning.md) | When and how to fine-tune with LoRA/QLoRA — vs. prompting and RAG | When does fine-tuning beat RAG? What is LoRA rank? What is QLoRA? |
| [SafetyAndHallucination.md](./SafetyAndHallucination/SafetyAndHallucination.md) | Hallucination types, mitigation stack, jailbreaks, alignment failures | How do I detect hallucination at scale? What is indirect prompt injection? |
| [CostAndLatency.md](./CostAndLatency/CostAndLatency.md) | Token budgets, model tiers, KV caching, streaming, cost estimation | How do I estimate monthly API cost? What is prefix caching? When is self-hosted cheaper? |

### Projects & Runnable Code

> **Status: coming soon.** The AI track currently has no runnable project. These are placeholders — the goal is to give the AI notes the same executable anchor that `projects/ml/linear-regression/` gives the ML track.

| Project | What it demonstrates | Status |
|---------|---------------------|--------|
| [projects/ai/rag-pipeline/](../../projects/ai/rag-pipeline/) | End-to-end RAG: chunk a document, embed it, store in a local vector index, query it, and evaluate the results with RAGAS metrics — the full pipeline from ingestion to evaluation in runnable Python | 🔴 Placeholder |

### Reference Documents

| File | Purpose |
|------|---------|
| [AI_Interview_Primer.md](./AI_Interview_Primer/AI_Interview_Primer.md) | Rapid-fire interview reference — crisp answers to every likely interview question across all topics |
| **This document** | Reading map and conceptual guide |

---

## The Story Arc: How the Concepts Chain Together

Read in this order to build the mental model from the ground up:

```
START HERE
    │
    ▼
Step 0: GROUND YOURSELF IN THE BASICS
        LLMFundamentals.md (complete)
        PromptEngineering.md (complete)
        AIPrimer.md (complete)

        Key insight: Before reasoning about agents, understand
        what an LLM actually is (tokenisation, sampling, RLHF,
        context windows), how to communicate with one reliably
        (prompt engineering, structured output, injection
        defense), and what concrete system the AI track builds
        toward (Mamma Rosa's PizzaBot — the running example
        that ties every subsequent document together).
    │
    ▼
Step 1: UNDERSTAND THE CORE MECHANISM
        CoTReasoning.md §1–3

        Key insight: An LLM is a next-token predictor.
        A ReAct agent is that same predictor, but its valid
        "next tokens" include structured tool calls. The host
        program executes those calls and feeds results back
        as tokens. This is the entire bridge from "language
        model" to "autonomous agent."
    │
    ▼
Step 2: SEE THE FULL LOOP IN ACTION
        ReActAndSemanticKernel.md §1–5

        Key insight: The ReAct Thought→Action→Observation
        loop is the practical embodiment of Step 1.
        CoTReasoning.md described why it works at the
        token level. This document shows what it looks
        like in code.
    │
    ▼
Step 3: UNDERSTAND HOW AGENTS GET EXTERNAL KNOWLEDGE
        RAGAndEmbeddings.md §1–6

        Key insight: The agent's tools may include a "retrieve"
        function backed by a RAG pipeline. For that retrieval
        to return useful chunks, the corpus must be embedded
        with the same model used at query time, and those
        embeddings must be stored and searched efficiently.
    │
    ▼
Step 4: UNDERSTAND THE STORAGE LAYER BENEATH RAG
        VectorDBs.md §1–5

        Key insight: The vector database is the engine behind
        Step 3's retrieval. The index type (HNSW, IVF, DiskANN)
        determines the speed, recall, and memory profile of every
        RAG query the agent makes.
    │
    ▼
Step 5: DEEPEN WITH FRAMEWORKS
        ReActAndSemanticKernel.md §7–13

        Key insight: LangChain and Semantic Kernel are software
        frameworks that automate the ReAct loop from Step 2,
        add memory management, and provide production-grade
        features. Semantic Kernel adds telemetry and filters;
        LangChain adds ecosystem breadth.
    │
    ▼
Step 6: ENRICH EACH LAYER WITH ITS SUPPLEMENT
        Read the four _Supplement.md files, one per core doc.
        Each adds: advanced patterns, failure modes, and the
        interview Q&A for its domain.
    │
    ▼
Step 7: APPLY PRODUCTION & OPERATIONS KNOWLEDGE
        EvaluatingAISystems.md → FineTuning.md →
        SafetyAndHallucination.md → CostAndLatency.md

        Key insight: An agent that works on your laptop is not
        an agent in production. This step closes the gap:
        measure what you built (RAGAS, LLM-as-judge), decide
        whether to fine-tune or keep prompting (LoRA/QLoRA),
        harden it against hallucination and injection, then
        size your infrastructure against a real cost model.
    │
    ▼
Step 8: CONSOLIDATE FOR INTERVIEWS
        AI_Interview_Primer.md

        A single file with crisp answers to every likely
        interview question. Best used as a spaced-repetition
        study tool the week before an interview.
```

---

## How the Documents Cross-Reference Each Other

### System-Level Running Example: Mamma Rosa's PizzaBot

[AIPrimer.md](./AIPrimer.md) defines the **primary running example** for the entire AI track: a RAG-backed chatbot for a pizza company, with a 6-file knowledge corpus and 3 external tools. Read it before the core notes. It maps every concept in every document to the concrete PizzaBot slice it teaches.

When you encounter a concept in a core note, cross-reference AIPrimer.md to see where it lives in the real system:

- **Embeddings + chunking** → how the menu, recipe, and allergen files are indexed
- **ReAct loop** → the fully annotated order-placement trace (Thought/Action/Observation × 6 steps)
- **Vector index** → the FAISS store behind the `retrieve_menu_info` retrieval tool
- **Tool schemas** → `find_nearest_location`, `check_item_availability`, `calculate_order_total`

### Concept-Level Examples in Individual Notes

The core notes also contain smaller, self-contained examples designed to isolate a single mechanic without domain noise:

- **CoTReasoning.md** and **ReActAndSemanticKernel.md** use a *train travel speed problem* to show CoT decomposition and the ReAct trace in isolation
- **RAGAndEmbeddings.md** and **VectorDBs.md** ground their examples in generic document retrieval patterns

These examples are intentionally narrow. AIPrimer.md is how you see all the layers connecting at once.

---

## Concept Dependency Graph

Some concepts in later documents depend on concepts from earlier ones:

```
CoTReasoning.md
  ├── "reasoning tokens" ────────────► needed to understand SK's hidden plan steps
  ├── "action language" ──────────────► core of ReAct in ReActAndSemanticKernel.md
  ├── "context window as scratchpad" ─► needed to understand RAG context injection
  └── "CoT failure modes" ────────────► directly informs agent failure modes in SK_Supplement

RAGAndEmbeddings.md
  ├── "embedding vectors" ──────────── prerequisite for VectorDBs.md (what is being indexed)
  ├── "cosine similarity" ────────────► used throughout VectorDBs.md for distance metrics
  ├── "chunking" ──────────────────────► defines the data structures that go into vector DBs
  └── "same model constraint" ─────────► a VectorDBs operational pitfall (Pitfall #2)

VectorDBs.md
  ├── "HNSW, IVF, DiskANN" ───────────► referenced in RAGAndEmbeddings.md §5.4 (indexing step)
  ├── "hybrid BM25 + vector" ─────────► referenced in RAGAndEmbeddings_Supplement.md §3
  └── "ANN recall" ────────────────────► RAGAS's context recall metric in RAGAndEmbeddings_Supplement

ReActAndSemanticKernel.md
  ├── "tool schemas" ─────────────────► the action language described in CoTReasoning.md §3.1
  ├── "context grows monotonically" ──► CoTReasoning.md §5 (context management)
  └── "memory via vector DB" ─────────► connects back to VectorDBs.md (RAG-based memory)
```

---

## Reading Paths by Goal

### "I have an interview at an AI company next week"
1. [AI_Interview_Primer.md](./AI_Interview_Primer/AI_Interview_Primer.md) — full pass, take notes on gaps  
2. [CoTReasoning.md](./CoTReasoning/CoTReasoning.md) §1–3 — fill the CoT gap  
3. [ReActAndSemanticKernel.md](./ReActAndSemanticKernel/ReActAndSemanticKernel.md) §1–5 + §12 — ReAct loop + comparison table  
4. [RAGAndEmbeddings.md](./RAGAndEmbeddings/RAGAndEmbeddings.md) §1 + §4–7 — embeddings + RAG pipeline  
5. [VectorDBs.md](./VectorDBs/VectorDBs.md) §2–4 — distance metrics + HNSW + IVF + comparison table  
6. Return to [AI_Interview_Primer.md](./AI_Interview_Primer/AI_Interview_Primer.md) — second pass, verify you can answer every question cold

### "I'm building a RAG-based agent from scratch"
1. [RAGAndEmbeddings.md](./RAGAndEmbeddings/RAGAndEmbeddings.md) — complete (ingestion + query pipeline)  
2. [VectorDBs.md](./VectorDBs/VectorDBs.md) — complete (choose and tune your index)  
3. [VectorDBs_Supplement.md](./VectorDBs/VectorDBs_Supplement.md) — tuning recipes + pitfalls  
4. [RAGAndEmbeddings_Supplement.md](./RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) — HyDE, RAGAS, failure modes  
5. [ReActAndSemanticKernel.md](./ReActAndSemanticKernel/ReActAndSemanticKernel.md) §7–8 — LangChain or SK for the agent wrapper  
6. [CoTReasoning.md](./CoTReasoning/CoTReasoning.md) §5 — context management for long agent runs

### "I want to understand how agents 'think' at a deep level"
1. [CoTReasoning.md](./CoTReasoning/CoTReasoning.md) — complete  
2. [CoTReasoning_Supplement.md](./CoTReasoning/CoTReasoning_Supplement.md) — complete  
3. [ReActAndSemanticKernel.md](./ReActAndSemanticKernel/ReActAndSemanticKernel.md) §1–6 — how thinking becomes acting  
4. [ReActAndSemanticKernel_Supplement.md](./ReActAndSemanticKernel/ReActAndSemanticKernel_Supplement.md) §1–2 — multi-agent thinking

### "I need to choose between LangChain and Semantic Kernel for a production project"
1. [ReActAndSemanticKernel.md](./ReActAndSemanticKernel/ReActAndSemanticKernel.md) §7–11 — complete comparison  
2. [ReActAndSemanticKernel_Supplement.md](./ReActAndSemanticKernel/ReActAndSemanticKernel_Supplement.md) §2–5 — LangGraph, HITL, tool design, failure modes  
3. [AI_Interview_Primer.md](./AI_Interview_Primer/AI_Interview_Primer.md) §3 — crisp summary of tradeoffs

### "I'm optimizing a slow or low-accuracy RAG system"
1. [RAGAndEmbeddings_Supplement.md](./RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) §1 — identify which failure mode you have  
2. [RAGAndEmbeddings.md](./RAGAndEmbeddings/RAGAndEmbeddings.md) §8–12 — chunking strategies and advanced techniques  
3. [RAGAndEmbeddings_Supplement.md](./RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) §2 — HyDE, FLARE, query decomposition  
4. [VectorDBs_Supplement.md](./VectorDBs/VectorDBs_Supplement.md) §4 — benchmarking protocol  
5. [VectorDBs.md](./VectorDBs/VectorDBs.md) §5 — hybrid retrieval  
6. [RAGAndEmbeddings_Supplement.md](./RAGAndEmbeddings/RAGAndEmbeddings_Supplement.md) §4 — RAGAS evaluation

---

## The Single Most Important Insight from Each Document

| Document | The One Insight to Internalize |
|----------|-------------------------------|
| **AIPrimer** | Every concept in the AI track has a concrete home in the PizzaBot system — read this first so every subsequent document feels like filling in a blueprint, not memorising abstractions. |
| **LLMFundamentals** | An LLM is a probability distribution over tokens, shaped first by next-token prediction on internet text, then steered by RLHF. Every capability and every failure mode flows from this. |
| **PromptEngineering** | The system prompt is a contract: it sets persona, output format, and safety rails all at once. If you make it ambiguous, the model will resolve the ambiguity for you — usually not how you intended. |
| **CoTReasoning** | An agent is an LLM whose next-token prediction is constrained to output a structured action, and whose environment executes that action and feeds the result back as tokens. Planning is emergent from this loop — not hard-coded. |
| **ReActAndSemanticKernel** | The difference between a chatbot and an agent is the loop: reason → act → observe → reason again. The LLM never executes anything. The host program does. |
| **RAGAndEmbeddings** | You cannot use two different embedding models in the same pipeline. The query and document embeddings must live in the exact same learned vector space or similarity scores are meaningless. |
| **VectorDBs** | HNSW's `efSearch` and IVF's `nprobe` are runtime dials — you control the recall/latency tradeoff at query time without rebuilding the index. Know these parameters cold. |
| **CoTReasoning_Supplement** | Process Reward Models reward each reasoning step, not just the final answer — this is why o1-class models have more reliable chains, not just better final answers. |
| **RAGAndEmbeddings_Supplement** | HyDE reverses the asymmetry problem: embed a hypothetical answer (same linguistic register as documents) instead of the raw question. |
| **ReActAndSemanticKernel_Supplement** | Multi-agent architecture solves context window saturation: each specialist agent maintains a short, focused context so attention stays sharp throughout long workflows. |
| **VectorDBs_Supplement** | Always load all data before creating the index, not the other way around. The index is optimized around the data distribution at build time. |
| **EvaluatingAISystems** | Faithfulness and answer relevance measure different failure modes — a faithful answer can still be irrelevant, and a relevant answer can still hallucinate. You need both metrics. |
| **FineTuning** | Fine-tuning encodes style, format, and tone into weights. RAG encodes facts. Use fine-tuning when prompting can't reliably produce the output shape you need; use RAG when the data changes. |
| **SafetyAndHallucination** | Indirect prompt injection — where a retrieved document contains adversarial instructions — is the highest-severity RAG risk in production. Design your system prompt to be injection-resistant from day one. |
| **CostAndLatency** | Prefix caching eliminates redundant processing of your system prompt on every call. For agents with long, stable system prompts this is the single highest-ROI latency and cost optimization available. |

---

<!-- AGENT-TODO: start
When all todos below are implemented, delete this entire block — from the opening comment tag to the closing comment tag, inclusive.

- [ ] Add Step 0 reading path entries to every goal-based reading path in the "Reading Paths by Goal" section (each path currently starts at step 1; prepend LLMFundamentals + PromptEngineering for any reader marked as "new to LLMs")
- [ ] Add a new reading path: "I'm completely new to LLMs" — route: LLMFundamentals → PromptEngineering → AIPrimer → Story Arc Step 1 onward
- [ ] Add Interview Checklist tables (Must Know / Likely Asked / Trap to Avoid) to each core AI note: CoTReasoning.md, RAGAndEmbeddings.md, VectorDBs.md, ReActAndSemanticKernel.md (fixes.md §2b)
- [ ] Add estimated reading/study times to each step in the Story Arc and to each path in "Reading Paths by Goal" (fixes.md §3a)
- [ ] Verify the Concept Dependency Graph entries still match the actual section content in each note; add dependency arrows for CostAndLatency → CoTReasoning ("thinking budget tokens") and SafetyAndHallucination → PromptEngineering ("injection-resistant system prompts")
AGENT-TODO: end -->
