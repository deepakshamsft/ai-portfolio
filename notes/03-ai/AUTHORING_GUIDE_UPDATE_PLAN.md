# AI Track Authoring Guide Update — Implementation Plan

**Target:** `notes/03-ai/authoring-guide.md`  
**Effort:** 6-8 hours (expanded from 2-3 hours due to comprehensive pattern adoption)  
**Status:** Draft v2.0 — aligned with ML track workflow patterns

---

## Executive Summary

This plan adapts the ML track's authoring patterns to AI track content. Key additions:

1. **Workflow-Based Chapter Pattern** for procedural AI chapters (prompt engineering, RAG pipeline, ReAct workflows, evaluation loops)
2. **Code Snippet Guidelines** (4 rules) adapted to API calls, prompt templates, and agent decision logic
3. **Industry Tools Integration** pattern showing manual → production (LangChain/LlamaIndex/RAGAS)
4. **Notebook Exercise Pattern** with decision checkpoints for iterative AI workflows
5. **AI-Specific Decision Checkpoint Templates** for prompt quality, retrieval relevance, agent tool selection

**Grand Challenge Preserved:** AI track's unique mission remains intact (e.g., chatbot accuracy, retrieval precision, prompt optimization, agent reliability).

---

## Part 1: Workflow-Based AI Chapters (NEW SECTION)

> **Adapted from:** ML track authoring-guide.md §"Workflow-Based Chapter Pattern"

### Identifying Procedural AI Chapters

An AI chapter is workflow-based if:
- ✅ It teaches a **sequence of decisions** more than a single concept
- ✅ Practitioner asks "what should I try next?" after each section
- ✅ Multiple techniques/prompts are chosen based on output quality
- ✅ The chapter reads like an iterative refinement guide, not a concept introduction

**Examples in AI Track:**
- **Workflow-based:** 
  - Ch02 Prompt Engineering (inspect output → diagnose failure mode → refine prompt → re-evaluate)
  - Ch06 ReAct (define tools → chain reasoning → execute actions → validate)
  - Ch07 RAG Pipeline Setup (chunk documents → embed → retrieve → rank → generate)
  - Ch08/Ch12 Evaluation (define metrics → run tests → analyze failures → iterate)
  
- **Concept-based:** 
  - Ch01 Embeddings (concept → math → similarity → visualization)
  - Ch03 Vector Databases (concept → indexing → search)
  - Ch04 Chain of Thought (concept → reasoning patterns → examples)

### AI Workflow Chapter Template

```markdown
# Ch.N — [Topic Name]

[Same header: story, curriculum context, notation]

---

## 0 · The Challenge — Where We Are
[Same as concept-based template]

## 1 · Core Idea
[Brief overview of the workflow purpose]

## 1.5 · The Practitioner Workflow — Your N-Phase Process

**Before diving into theory, understand the workflow you'll follow with every AI task:**

> 🤖 **What you'll build by the end:** [Description of final deliverable/agent/pipeline]

```
Phase 1: [ACTION]           Phase 2: [ACTION]           Phase 3: [ACTION]
──────────────────────────────────────────────────────────────────────────
[What you do]               [What you do]               [What you do]

→ DECISION:                 → DECISION:                 → DECISION:
  [Quality criteria]          [Strategy selection]        [Iteration path]
```

**The workflow maps to this chapter:**
- **Phase 1 ([ACTION])** → §X Section Name
- **Phase 2 ([ACTION])** → §Y Section Name  
- **Phase 3 ([ACTION])** → §Z Section Name

> 💡 **Usage note:** [Brief note on phase dependencies and when to iterate]

---

## 2 · Running Example
[Same as concept-based template — use domain-specific AI task]

## 3 · Implementation

### 3.X · [Phase Name] **[Phase N: ACTION]**
[Section content with phase marker in header]

[Code snippet showing phase implementation]

```python
# Phase N: [Brief description]
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7
)

output = response.choices[0].message.content

# DECISION LOGIC (inline annotation)
if quality_metric(output) < threshold:
    strategy = "❌ REFINE - Add examples to prompt"
elif quality_metric(output) < ideal_threshold:
    strategy = "⚠️ ITERATE - Adjust temperature"
else:
    strategy = "✅ DEPLOY"

print(f"Quality: {quality_metric(output):.2f} → {strategy}")
```

> 💡 **Industry Standard:** `langchain.prompts.ChatPromptTemplate`
> ```python
> from langchain.prompts import ChatPromptTemplate
> from langchain_openai import ChatOpenAI
> 
> template = ChatPromptTemplate.from_messages([
>     ("system", "{system_prompt}"),
>     ("user", "{query}")
> ])
> chain = template | ChatOpenAI(model="gpt-4", temperature=0.7)
> output = chain.invoke({"system_prompt": prompt, "query": question})
> ```
> **When to use:** Production pipelines requiring prompt versioning and chain composition.
> **Common alternatives:** `guidance` (constrained generation), `Anthropic SDK`, `LlamaIndex PromptTemplate`
> **See also:** [LangChain docs](https://python.langchain.com/docs/modules/model_io/prompts/)

### 3.X.1 DECISION CHECKPOINT — Phase N Complete

**What you just observed:**
- [Specific output quality metric with number from code execution]
- [Pattern in failures: hallucinations, off-topic, format issues]
- [Performance characteristic: latency, token usage]

**What it means:**
- [Interpretation: why this failure mode occurred]
- [Impact: effect on downstream pipeline or user experience]
- [Root cause: prompt ambiguity, insufficient context, model limitation]

**What to do next:**
→ **Option 1 (Refine Prompt):** Add few-shot examples showing desired output format
→ **Option 2 (Change Strategy):** Switch from zero-shot to chain-of-thought reasoning
→ **Option 3 (Model Selection):** Use higher-capability model if quality threshold not met
→ **For our scenario:** Choose [option] because [specific reasoning based on observed metrics]

---

[Repeat pattern for all phases]

## N-1 · Putting It Together — The Complete Workflow

[Mermaid flowchart showing all phases integrated with decision branches specific to AI task]

## N · Progress Check — What We Can Solve Now
[Same as concept-based template]

## N+1 · Bridge to the Next Chapter
[Same as concept-based template]
```

### AI-Specific Decision Checkpoint Templates

#### Template 1: Prompt Engineering Iteration

```markdown
### X.Y DECISION CHECKPOINT — Prompt Refinement Cycle N Complete

**What you just observed:**
- Prompt version N produced accuracy of [X%] on validation set
- Common failure mode: [hallucination / off-topic / format violation]
- Token usage: [N tokens] (within/exceeding budget)

**What it means:**
- The model lacks [context/examples/constraints] to handle [edge case]
- Current prompt structure assumes [incorrect assumption about model behavior]
- [Specific error pattern] indicates [root cause diagnosis]

**What to do next:**
→ **Add Few-Shot Examples:** Include 2-3 examples showing [desired behavior]
→ **Strengthen Constraints:** Add explicit instruction "[specific constraint]"
→ **Decompose Task:** Break into subtasks if accuracy < [threshold]%
→ **For our case:** Add examples because [specific reasoning]
```

#### Template 2: RAG Pipeline Optimization

```markdown
### X.Y DECISION CHECKPOINT — Retrieval Quality Assessment

**What you just observed:**
- Top-K retrieval precision: [X%] (documents relevant to query)
- Average semantic similarity score: [0.XX]
- Context window utilization: [X%] of available tokens

**What it means:**
- Chunking strategy captures [well/poorly] semantic boundaries
- Embedding model [distinguishes/confuses] domain-specific terminology
- Reranking [improves/degrades] initial retrieval by [X%]

**What to do next:**
→ **Adjust Chunk Size:** Reduce to [N tokens] if precision < [threshold]
→ **Add Metadata Filters:** Pre-filter by [attribute] before semantic search
→ **Hybrid Search:** Combine with BM25 if semantic alone insufficient
→ **Rerank:** Apply cross-encoder if top-10 contains irrelevant docs
→ **For our scenario:** Use hybrid search because [reasoning]
```

#### Template 3: Agent Tool Selection

```markdown
### X.Y DECISION CHECKPOINT — ReAct Decision Path Validation

**What you just observed:**
- Agent selected [Tool X] for [N%] of queries in category [Y]
- Tool execution success rate: [X%]
- Average reasoning steps before action: [N]

**What it means:**
- Agent correctly identifies [tool use case] but struggles with [edge case]
- Tool descriptions [are/aren't] sufficiently distinct
- Reasoning traces show [overfitting to examples / correct generalization]

**What to do next:**
→ **Refine Tool Descriptions:** Clarify when to use [Tool X] vs [Tool Y]
→ **Add Decision Examples:** Show [edge case] → [correct tool] mapping
→ **Adjust Temperature:** Lower to [0.X] if selection too random
→ **Validate Tools First:** Add pre-execution validation if errors > [X%]
→ **For our scenario:** Refine descriptions because [specific confusion observed]
```

#### Template 4: Evaluation Loop

```markdown
### X.Y DECISION CHECKPOINT — Test Suite Results

**What you just observed:**
- [N/M] test cases passed ([X%] success rate)
- Failure modes: [hallucination: Y%, refusal: Z%, format: W%]
- Average evaluation time: [X seconds] per case

**What it means:**
- Current approach handles [common cases] but fails on [edge cases]
- [Specific metric] below production threshold of [X]
- Failure distribution suggests [systemic issue vs random errors]

**What to do next:**
→ **Augment Test Cases:** Add [N] cases covering [failure mode]
→ **Iterate on System Prompt:** Address [specific weakness]
→ **Add Guardrails:** Implement [validation logic] for [failure type]
→ **Escalate to Human:** Route [edge case type] to human review
→ **For our scenario:** Add guardrails because [reasoning based on failure distribution]
```

---

## Part 2: Code Snippet Guidelines for AI Chapters

> **Adapted from:** ML track authoring-guide.md §"Code Snippet Guidelines for Workflow Chapters"

**Rule 1: Each phase ends with executable code showing that phase's API workflow**

```python
# ✅ Good: Phase 1 code snippet (prompt iteration loop)
prompts = {
    "v1": "Summarize this: {text}",
    "v2": "Provide a concise 2-sentence summary of: {text}",
    "v3": "Summarize in exactly 2 sentences. Format: [Sentence 1]. [Sentence 2].\n\nText: {text}"
}

for version, template in prompts.items():
    prompt = template.format(text=document)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content
    
    # DECISION LOGIC
    sentences = output.split('.')
    if len(sentences) == 2:
        verdict = "✅ FORMAT CORRECT"
    else:
        verdict = f"❌ WRONG FORMAT ({len(sentences)} sentences)"
    
    print(f"{version}: {verdict}")
```

**Rule 2: Decision logic appears in code comments or inline conditionals, not just prose**

```python
# ✅ Good: Inline decision annotation for agent tool selection
def select_tool(query: str, tools: list[Tool]) -> Tool:
    """Agent reasoning for tool selection."""
    
    # DECISION LOGIC: Route based on query intent
    if "weather" in query.lower():
        return tools["weather_api"]  # ✅ Factual lookup
    elif "calculate" in query.lower() or any(op in query for op in ['+', '-', '*', '/']):
        return tools["calculator"]   # ✅ Computation
    elif "search" in query.lower():
        return tools["web_search"]   # ✅ Information retrieval
    else:
        return tools["llm_direct"]   # ⚠️ Fallback to direct LLM
```

**Rule 3: Code should be copy-paste executable with real API endpoints**

Include:
- All necessary imports (`openai`, `langchain`, `anthropic`, etc.)
- API key setup (with placeholder or env var pattern)
- Real or realistic dataset/query examples
- Expected output format in comments
- Error handling for API failures

```python
# ✅ Good: Complete executable example
import os
from openai import OpenAI

# Setup (user should set OPENAI_API_KEY environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Real query from running example
query = "What are the key differences between transformers and RNNs?"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": query}],
    temperature=0.7
)

print(response.choices[0].message.content)
# Expected output: [2-3 paragraph explanation covering attention, parallelization, context]
```

❌ **Avoid:**
```python
# Bad: Placeholder variables without context
response = llm(your_prompt_here)  # ❌ Not executable
```

**Rule 4: Show progressive building, not isolated snippets**

```python
# ✅ Good: References earlier setup
# Using the client and query from Phase 1 above...
# Now adding system prompt for structured output

messages = [
    {"role": "system", "content": "You are a technical educator. Answer in exactly 3 bullet points."},
    {"role": "user", "content": query}  # Reuses query from earlier
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7
)

# DECISION: Check if format constraint was followed
bullets = [line for line in response.choices[0].message.content.split('\n') if line.strip().startswith('-')]
if len(bullets) == 3:
    print("✅ Format correct")
else:
    print(f"⚠️ Got {len(bullets)} bullets instead of 3 → Refine system prompt")
```

---

## Part 3: Industry Tools Integration (AI Track)

> **Adapted from:** ML track authoring-guide.md §"Industry Standard Tools Integration"

**Core principle:** Show manual API calls first (build intuition), then show framework abstraction.

### Required Callout Box Pattern for AI Chapters

```markdown
> 💡 **Industry Standard:** `langchain.chains.RetrievalQA`
> 
> ```python
> from langchain.chains import RetrievalQA
> from langchain_openai import ChatOpenAI
> from langchain_community.vectorstores import Chroma
> 
> # Manual approach shown above for learning
> # Production: Use chain abstraction
> qa_chain = RetrievalQA.from_chain_type(
>     llm=ChatOpenAI(model="gpt-4"),
>     retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
>     chain_type="stuff",  # or "map_reduce", "refine"
>     return_source_documents=True
> )
> result = qa_chain.invoke({"query": question})
> ```
> 
> **When to use:** Production RAG pipelines requiring versioning, logging, and chain composition.
> **Common alternatives:** `LlamaIndex QueryEngine`, `Haystack Pipeline`, `Semantic Kernel`
> **See also:** [LangChain RAG tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
```

### AI-Specific Industry Tools by Chapter

| Chapter | Manual Approach | Industry Framework | When to Use Framework |
|---------|----------------|-------------------|----------------------|
| **Ch02 Prompt Engineering** | String formatting + API call | `LangChain PromptTemplate`, `guidance` | Prompt versioning, A/B testing, production deployment |
| **Ch03 Vector Databases** | Direct `numpy` similarity | `Pinecone`, `Weaviate`, `Qdrant`, `Chroma` | >1M vectors, production scale, multi-user |
| **Ch06 ReAct / Function Calling** | Manual tool routing logic | `LangChain Agent`, `Semantic Kernel Planner` | Multi-tool orchestration, dynamic tool selection |
| **Ch07 RAG Pipeline** | Manual chunking + embed + retrieve | `LangChain RetrievalQA`, `LlamaIndex QueryEngine` | Production RAG, multi-stage retrieval, hybrid search |
| **Ch08 Evaluation** | Manual test harness | `RAGAS`, `TruLens`, `DeepEval`, `PromptFoo` | Systematic evaluation, LLM-as-judge, regression testing |
| **Ch09 Fine-tuning** | Manual training loop | `transformers.Trainer`, `PEFT`, `Axolotl` | Production fine-tuning, LoRA, quantization |
| **Ch10 Guardrails** | Manual validation logic | `NeMo Guardrails`, `Guardrails AI` | Complex policy enforcement, multi-modal validation |

### Callout Frequency Guidelines

**Add 1 industry callout per:**
- Major architectural pattern (RAG pipeline, agent loop, evaluation harness)
- Production-scale concern (vector database, batch inference, caching)
- Framework-enabled capability (chain composition, prompt versioning, auto-logging)

**Typical per-chapter count:** 3-5 callouts

**Placement:**
- After demonstrating manual implementation (show principle first)
- Before "Putting It Together" section (bridge to production)
- In notebook exercises (manual in markdown, framework in solution)

---

## Part 4: Notebook Exercise Pattern (AI Track)

> **Adapted from:** ML track authoring-guide.md §"Notebook Exercise Pattern"

### Exercise Notebook Enhancement Pattern

**Structure:**
- **Solution notebook:** Fully implemented code with API outputs
- **Exercise notebook:** Markdown prompts + placeholder code cells (`# TODO: Implement...`)

**Required enhancements for workflow-based AI chapters:**

#### 1. Industry Standard Callout Boxes (AI Adapted)

Add to markdown cells after each major concept:

```markdown
> 💡 **Industry Standard Pattern:** After implementing manually, use:
> ```python
> from langchain.prompts import ChatPromptTemplate
> from langchain_openai import ChatOpenAI
> 
> template = ChatPromptTemplate.from_messages([
>     ("system", "You are a helpful assistant."),
>     ("user", "{query}")
> ])
> chain = template | ChatOpenAI(model="gpt-4")
> response = chain.invoke({"query": "Your question here"})
> ```
> **When to use:** Production deployments requiring prompt versioning and observability.
> **Common alternatives:** `guidance` (constrained generation), `Anthropic SDK` (Claude-specific), `LlamaIndex` (RAG focus)
```

**Pattern frequency:** 3-5 callouts per notebook

**AI-specific placement:**
- After prompt engineering iterations (manual string format → `PromptTemplate`)
- After retrieval implementation (manual similarity → `VectorStore.similarity_search()`)
- After agent tool selection (manual routing → `LangChain Agent`)
- After evaluation loops (manual scoring → `RAGAS.evaluate()`)

#### 2. Decision Logic Templates (AI Workflows)

Add before code cells requiring iterative refinement:

```markdown
**Decision Logic Template for Prompt Quality:**

When implementing prompt evaluation, include quality-based branching:

\```python
def evaluate_prompt(prompt: str, test_cases: list) -> dict:
    """Evaluate prompt quality across test cases."""
    results = {"passed": 0, "failed": 0, "failure_modes": []}
    
    for case in test_cases:
        output = llm(prompt.format(**case["input"]))
        
        # DECISION LOGIC (add this pattern)
        if output_matches_format(output, case["expected_format"]):
            if semantic_similarity(output, case["expected_content"]) > 0.85:
                results["passed"] += 1
                verdict = "✅ PASS"
            else:
                results["failed"] += 1
                results["failure_modes"].append("content_mismatch")
                verdict = "⚠️ FORMAT OK, CONTENT WEAK"
        else:
            results["failed"] += 1
            results["failure_modes"].append("format_violation")
            verdict = "❌ FORMAT VIOLATION"
        
        print(f"Case {case['id']:2d}: {verdict}")
    
    return results
\```

**Quality Thresholds:**
- Format match rate > 95% → Deploy
- Format match rate 80-95% → Refine constraints
- Format match rate < 80% → Redesign prompt structure
```

**Pattern frequency:** 2-4 per notebook

**AI-specific placement:**
- Prompt iteration loops (quality assessment → refinement decision)
- Retrieval ranking (relevance score → reranking decision)
- Agent tool selection (confidence threshold → fallback decision)
- Evaluation scoring (metric threshold → pass/fail/iterate)

#### 3. Visual Indicators (AI Context)

Use consistent emoji/symbols for AI task outcomes:

| Indicator | Meaning | AI Use Case |
|-----------|---------|-------------|
| ✅ | Correct/Pass/Deploy | Output meets quality threshold |
| ⚠️ | Acceptable/Monitor | Output functional but suboptimal |
| ⚡ | Iterate/Refine | Output requires prompt adjustment |
| ❌ | Fail/Reject/Escalate | Output violates constraints or factually wrong |
| 💡 | Industry standard | Production-ready framework pattern |
| 🔍 | Debug/Investigate | Unexpected behavior requiring analysis |
| 🤖 | Agent decision | Autonomous tool selection or routing |

### AI-Specific Implementation Checklist

When creating/updating exercise notebooks for AI workflow chapters:

- [ ] **Industry callouts added** (3-5 locations: manual API → framework)
- [ ] **Decision logic templates added** (2-4 locations: quality thresholds for iteration)
- [ ] **Visual indicators consistent** (✅ ❌ ⚠️ ⚡ 💡 🤖 used appropriately)
- [ ] **Quality thresholds documented** (e.g., BLEU > 0.7, semantic sim > 0.85)
- [ ] **API key handling shown** (`os.getenv("OPENAI_API_KEY")` pattern)
- [ ] **Error handling included** (rate limits, timeouts, invalid responses)
- [ ] **Cost awareness** (token usage tracking, model selection guidance)
- [ ] **Code cells remain placeholder** (`# TODO: Implement...` preserved in exercise)
- [ ] **Markdown cells expanded** (guidance added, not replaced)

### Anti-Patterns to Avoid (AI Context)

❌ **Don't:**
- Hardcode API keys in notebooks (`openai.api_key = "sk-..."` → security risk)
- Use production endpoints in exercises without rate limiting
- Show only framework code without manual implementation (defeats learning goal)
- Include outdated API syntax (e.g., `openai.Completion` → use `openai.chat.completions`)
- Mix multiple decision templates in one markdown cell
- Use vague quality criteria ("good enough" → define metric threshold)

✅ **Do:**
- Use environment variables for API keys with clear setup instructions
- Show token usage and cost estimation in examples
- Demonstrate manual approach first, then framework equivalent
- Use current API patterns (`openai>=1.0.0` client syntax)
- Keep each template focused on one decision type
- Define specific metrics (BLEU, ROUGE, semantic similarity, format compliance)

---

## Part 5: AI Track Procedural Chapter Audit

### Current Chapter Assessment

| Chapter | Current Structure | Should Use Workflow? | Priority | Rationale |
|---------|------------------|---------------------|----------|-----------|
| **Ch01 Embeddings** | Concept-based | ❌ NO | - | Single concept (vector representations) |
| **Ch02 Prompt Engineering** | Concept-based → **Workflow** | ✅ YES | **HIGH** | Iterative refinement loop (draft → evaluate → refine → deploy) |
| **Ch03 Vector Databases** | Concept-based | ❌ NO | - | Infrastructure component, not workflow |
| **Ch04 Chain of Thought** | Concept-based | ⚠️ CONSIDER | LOW | Could add "when to use CoT vs direct" decision logic |
| **Ch05 Fine-tuning** | Concept-based | ⚠️ CONSIDER | MEDIUM | Has workflow (prepare data → train → evaluate → iterate) but heavy on concept |
| **Ch06 ReAct / Function Calling** | Concept-based → **Workflow** | ✅ YES | **HIGH** | Multi-phase: define tools → compose → execute → validate → refine |
| **Ch07 RAG Pipeline** | Concept-based → **Workflow** | ✅ YES | **CRITICAL** | Clear 5-phase workflow: chunk → embed → index → retrieve → generate |
| **Ch08 Evaluation** | Concept-based → **Workflow** | ✅ YES | **HIGH** | Testing loop: define metrics → run tests → analyze failures → iterate |
| **Ch09 Multimodal** | Concept-based | ❌ NO | - | Concept introduction (vision, audio, cross-modal) |
| **Ch10 Guardrails** | Concept-based | ⚠️ CONSIDER | LOW | Could frame as validation workflow |
| **Ch11 Inference Optimization** | Concept-based | ⚠️ CONSIDER | LOW | Diagnostic workflow (profile → optimize → validate) |
| **Ch12 Production Testing** | Concept-based → **Workflow** | ✅ YES | **HIGH** | Clear testing workflow (unit → integration → E2E → monitoring) |

### Workflow Adoption Priorities

**CRITICAL (Immediate Action):**
- **Ch07 RAG Pipeline:** Natural 5-phase structure (chunking → embedding → indexing → retrieval → generation), clear decision points at each phase

**HIGH (Next Sprint):**
- **Ch02 Prompt Engineering:** Iterative refinement is core concept, workflow structure reinforces learning
- **Ch06 ReAct:** Tool orchestration naturally workflow-based (define → chain → execute)
- **Ch08/Ch12 Evaluation:** Testing inherently procedural (setup → run → analyze → iterate)

**MEDIUM (Consider for Future):**
- **Ch05 Fine-tuning:** Has workflow elements but heavy theoretical content may conflict
- **Ch11 Inference Optimization:** Performance tuning follows diagnostic workflow

**LOW (Optional Enhancement):**
- **Ch04 Chain of Thought:** Could add decision logic for "when to use CoT"
- **Ch10 Guardrails:** Validation could be framed as workflow

---

## Part 6: AI-Specific Workflow Templates

### Template: Prompt Engineering Chapter (Ch02)

**5-Phase Workflow:**

```
Phase 1: Baseline         Phase 2: Diagnose        Phase 3: Refine          Phase 4: Validate         Phase 5: Deploy
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Zero-shot prompt          Identify failure modes   Add examples/constraints  Test on holdout set      Monitor production

→ DECISION:               → DECISION:              → DECISION:               → DECISION:              → DECISION:
  Quality < 70%?            Hallucination?           Format violations?        Accuracy > 90%?          Drift detected?
  → Phase 2                 → Add grounding          → Strengthen rules        → Deploy                 → Retrigger Phase 2
  Quality ≥ 70%?            Off-topic?               Content quality low?      Accuracy 80-90%?         Performance OK?
  → Phase 4 (skip refine)   → Clarify scope          → Add few-shot examples   → Phase 3 (iterate)      → Monitor
```

**Decision Checkpoint Example (after Phase 2):**

```markdown
### 3.2.1 DECISION CHECKPOINT — Failure Mode Diagnosis Complete

**What you just observed:**
- Baseline prompt accuracy: 65% (13/20 test cases passed)
- Failure modes: 4 hallucinations, 2 off-topic, 1 format violation
- Average response length: 127 tokens (target: 50-100)

**What it means:**
- Model lacks grounding context (hallucinations on factual questions)
- Task scope ambiguous (off-topic responses indicate unclear boundaries)
- Output format loosely specified (one case returned paragraph instead of list)

**What to do next:**
→ **Add Grounding Context:** Include "Only use information from: [source]" constraint
→ **Clarify Scope:** Add "Do not answer questions about [off-topic categories]" rule
→ **Specify Format:** Change to "Answer in exactly 3 bullet points. Use '-' prefix."
→ **For our scenario:** Apply all three because failure modes are independent (not competing)
```

### Template: RAG Pipeline Chapter (Ch07)

**5-Phase Workflow:**

```
Phase 1: Chunk           Phase 2: Embed           Phase 3: Index           Phase 4: Retrieve         Phase 5: Generate
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Split documents          Generate vectors         Build search index       Query + rank results     LLM with context

→ DECISION:              → DECISION:              → DECISION:              → DECISION:              → DECISION:
  Chunk size?              Embedding model?         Index type?              Top-K value?             Prompt strategy?
  - Semantic boundary      - Domain match           - Flat (< 1M vectors)    - Precision vs recall    - Stuff (< 4k tokens)
  - Fixed tokens           - Cost vs quality        - HNSW (> 1M vectors)    - Rerank if needed       - Map-reduce (> 4k)
  - Overlap 10-15%         - OpenAI vs open         - Hybrid (BM25 + sem)    - Filter by metadata     - Refine (iterative)
```

**Decision Checkpoint Example (after Phase 4):**

```markdown
### 3.4.1 DECISION CHECKPOINT — Retrieval Quality Assessment

**What you just observed:**
- Top-5 retrieval precision: 60% (3/5 chunks relevant per query)
- Average cosine similarity: 0.72 (threshold: 0.75 for "relevant")
- BM25 hybrid retrieval improved precision to 80%

**What it means:**
- Pure semantic search misses keyword-specific queries (e.g., product codes, technical IDs)
- Chunk boundaries split some concepts across multiple chunks (context fragmentation)
- Reranking with cross-encoder could boost precision further (80% → ~90%)

**What to do next:**
→ **Enable Hybrid Search:** Combine semantic (0.7 weight) + BM25 (0.3 weight)
→ **Adjust Chunk Size:** Reduce from 512 → 384 tokens to respect semantic boundaries
→ **Add Reranker:** Use `cross-encoder/ms-marco-MiniLM-L-12-v2` on top-10 results
→ **For our scenario:** Start with hybrid search (quick win), then add reranker if precision still < 85%
```

### Template: ReAct / Function Calling (Ch06)

**4-Phase Workflow:**

```
Phase 1: Define Tools    Phase 2: Compose Chain   Phase 3: Execute         Phase 4: Validate
───────────────────────────────────────────────────────────────────────────────────────────
Specify tool signatures  Agent selects tools      Run tool calls           Check results

→ DECISION:              → DECISION:              → DECISION:              → DECISION:
  Tool granularity?        Tool selection correct?  Execution successful?    Output quality OK?
  - Atomic (1 action)      - Reasoning valid?       - Handle errors?         - Answer correct?
  - Composite (multi)      - Confidence score?      - Retry logic?           - Sources cited?
```

**Decision Checkpoint Example (after Phase 2):**

```markdown
### 3.2.1 DECISION CHECKPOINT — Agent Tool Selection Analysis

**What you just observed:**
- Agent selected correct tool for 17/20 queries (85% accuracy)
- 2 errors: confused `calculator` and `web_search` for currency conversion
- 1 error: selected `llm_direct` instead of `weather_api` for weather query
- Average reasoning steps before action: 2.3

**What it means:**
- Tool descriptions insufficiently distinct for edge cases (currency = calculation vs factual lookup)
- Weather query failed because description said "current weather" but query asked for "forecast"
- Reasoning length appropriate (2-3 steps is optimal for this task complexity)

**What to do next:**
→ **Refine Tool Descriptions:** 
   - `calculator`: "For math operations on numbers you already know"
   - `web_search`: "For looking up current facts, rates, or data you don't have"
→ **Update Weather Tool:** Change description to "current weather OR forecast"
→ **Add Few-Shot Examples:** Include currency conversion → `web_search` example
→ **For our scenario:** Refine descriptions (10 min fix) before adding examples (30 min)
```

---

## Part 7: Implementation Checklist

### Phase 1: Authoring Guide Updates (Priority: CRITICAL)

**Target file:** `notes/03-ai/authoring-guide.md`

**Additions (insert after current "Chapter Template" section):**

1. **§ Workflow-Based Chapter Pattern** (~800 words)
   - When to use (AI chapter decision criteria)
   - Modified template with 5-phase examples
   - Decision checkpoint format (3-part structure)
   - When NOT to use workflow structure

2. **§ Code Snippet Guidelines for AI Chapters** (~400 words)
   - Rule 1: Each phase ends with executable API code
   - Rule 2: Decision logic in inline comments
   - Rule 3: Copy-paste executable with real endpoints
   - Rule 4: Progressive building, not isolated snippets

3. **§ Industry Tools Integration** (~600 words)
   - Core principle (manual → framework)
   - Callout box pattern
   - AI-specific tools by chapter (table)
   - Placement guidelines (3-5 per chapter)

4. **§ Notebook Exercise Pattern** (~500 words)
   - Industry callout boxes (LangChain/LlamaIndex/RAGAS)
   - Decision logic templates (AI workflows)
   - Visual indicators (AI context)
   - Implementation checklist
   - Anti-patterns to avoid

5. **§ AI Track Procedural Chapter Audit** (~300 words)
   - Table of all chapters with workflow assessment
   - Priority rankings (CRITICAL/HIGH/MEDIUM/LOW)
   - Rationale for each classification

6. **§ AI-Specific Workflow Templates** (~1200 words)
   - Ch02 Prompt Engineering template (5-phase)
   - Ch07 RAG Pipeline template (5-phase)
   - Ch06 ReAct template (4-phase)
   - Ch08/Ch12 Evaluation template (4-phase)

**Estimated word count:** ~3800 words (8-10 pages)  
**Estimated effort:** 6-8 hours

### Phase 2: Chapter Content Updates (Priority: HIGH)

**Order of execution:**

1. **Ch07 RAG Pipeline** (CRITICAL — most naturally workflow-based)
   - Add §1.5 workflow overview (5 phases)
   - Add decision checkpoints after each phase
   - Add industry callouts (LangChain, LlamaIndex, Chroma/Pinecone)
   - Update notebook exercise with decision templates

2. **Ch02 Prompt Engineering** (HIGH — foundational skill)
   - Add §1.5 iterative refinement workflow
   - Add decision checkpoints for failure mode diagnosis
   - Add industry callouts (LangChain PromptTemplate, guidance)
   - Update notebook with quality threshold templates

3. **Ch06 ReAct** (HIGH — complex orchestration)
   - Add §1.5 tool orchestration workflow
   - Add decision checkpoints for tool selection validation
   - Add industry callouts (LangChain Agent, Semantic Kernel)
   - Update notebook with agent decision logic templates

4. **Ch08/Ch12 Evaluation** (HIGH — quality assurance)
   - Add §1.5 testing workflow
   - Add decision checkpoints for metric threshold evaluation
   - Add industry callouts (RAGAS, TruLens, DeepEval)
   - Update notebook with test harness templates

### Phase 3: Notebook Audits (Priority: MEDIUM)

For each workflow chapter, audit exercise notebooks:

- [ ] Manual implementation shown first (no framework dependencies)
- [ ] Industry callout added after manual implementation
- [ ] Decision logic template added before iteration cells
- [ ] Visual indicators consistent (✅ ❌ ⚠️ ⚡ 💡 🤖)
- [ ] Quality thresholds documented with specific numbers
- [ ] API key handling secure (`os.getenv()` pattern)
- [ ] Token usage and cost estimation included
- [ ] Error handling for API failures

**Notebooks to audit:**
1. `ch02_prompt_engineering/notebook_exercise.ipynb`
2. `ch06_react/notebook_exercise.ipynb`
3. `ch07_rag/notebook_exercise.ipynb`
4. `ch08_evaluation/notebook_exercise.ipynb`
5. `ch12_testing/notebook_exercise.ipynb`

---

## Part 8: Success Criteria

### Authoring Guide Completeness

- [x] Workflow pattern section added with AI-specific examples
- [x] Code snippet guidelines adapted to API calls and prompt templates
- [x] Industry tools documented (LangChain, LlamaIndex, RAGAS, etc.)
- [x] Decision checkpoint templates for all AI workflow types
- [x] Clear guidance on when to use workflow vs concept structure
- [x] Chapter-by-chapter audit table with priorities

### Content Quality Standards

- [x] AI track grand challenge preserved (chatbot accuracy, retrieval precision, agent reliability)
- [x] Patterns adapted, not blindly copied from ML track
- [x] Decision checkpoints use AI-specific metrics (BLEU, semantic similarity, format compliance)
- [x] Industry tools relevant to AI practitioners (not ML-specific sklearn tools)
- [x] Code examples use current API syntax (OpenAI SDK v1.0+, LangChain v0.1+)

### Implementation Readiness

- [x] Ch07 RAG identified as CRITICAL priority (natural 5-phase workflow)
- [x] Ch02/Ch06/Ch08/Ch12 identified as HIGH priority
- [x] Specific workflow templates provided for each priority chapter
- [x] Decision checkpoint examples demonstrate 3-part structure
- [x] Notebook audit checklist provided with AI-specific criteria

---

## Appendix: Industry Tools Reference (AI Track)

### Core Frameworks

| Framework | Use Case | Chapters | Key Features |
|-----------|----------|----------|--------------|
| **LangChain** | General-purpose LLM orchestration | Ch02, Ch06, Ch07, Ch08 | Chains, agents, prompt templates, callbacks |
| **LlamaIndex** | RAG-focused framework | Ch07 | Query engines, index structures, retrievers |
| **Haystack** | Production NLP pipelines | Ch07, Ch12 | Document stores, pipelines, REST API |
| **Semantic Kernel** | Microsoft's LLM orchestration | Ch06 | Planners, plugins, memory |

### Specialized Tools

| Tool | Category | Use Case | Chapters |
|------|----------|----------|----------|
| **RAGAS** | Evaluation | RAG pipeline metrics (faithfulness, relevance) | Ch08, Ch12 |
| **TruLens** | Observability | LLM app monitoring and evaluation | Ch08, Ch12 |
| **DeepEval** | Testing | Unit tests for LLM outputs | Ch08, Ch12 |
| **PromptFoo** | Prompt testing | Systematic prompt evaluation | Ch02, Ch08 |
| **guidance** | Constrained generation | Structured output enforcement | Ch02, Ch10 |
| **Guardrails AI** | Validation | Output validation and correction | Ch10 |
| **NeMo Guardrails** | Policy enforcement | Multi-modal safety rails | Ch10 |

### Vector Databases

| Database | Scale | Use Case | Chapters |
|----------|-------|----------|----------|
| **Pinecone** | Cloud, serverless | Production RAG (managed) | Ch03, Ch07 |
| **Weaviate** | Self-hosted/cloud | Hybrid search, multi-modal | Ch03, Ch07 |
| **Qdrant** | Self-hosted/cloud | High-performance search | Ch03, Ch07 |
| **Chroma** | Local, embedded | Development, small scale | Ch03, Ch07 |
| **Milvus** | Self-hosted | Large-scale distributed | Ch03, Ch07 |

### Model Providers

| Provider | Models | API Style | Chapters |
|----------|--------|-----------|----------|
| **OpenAI** | GPT-4, GPT-3.5, embeddings | REST + Python SDK | All |
| **Anthropic** | Claude 3.x | REST + Python SDK | Ch02, Ch09 |
| **Cohere** | Command, embed | REST + Python SDK | Ch01, Ch07 |
| **Hugging Face** | Open models | `transformers` library | Ch05, Ch09, Ch11 |
| **Together AI** | Open models (hosted) | OpenAI-compatible API | Ch11 |

---

## Timeline & Effort Estimate

| Phase | Tasks | Estimated Effort | Priority |
|-------|-------|-----------------|----------|
| **Phase 1: Authoring Guide** | Add 6 sections (~3800 words) | 6-8 hours | CRITICAL |
| **Phase 2: Ch07 RAG** | Add workflow structure + notebook | 4-6 hours | CRITICAL |
| **Phase 3: Ch02 Prompts** | Add workflow structure + notebook | 3-4 hours | HIGH |
| **Phase 4: Ch06 ReAct** | Add workflow structure + notebook | 3-4 hours | HIGH |
| **Phase 5: Ch08/Ch12 Eval** | Add workflow structure + notebook | 4-5 hours | HIGH |
| **Phase 6: Notebook Audits** | Audit 5 exercise notebooks | 2-3 hours | MEDIUM |

**Total estimated effort:** 22-30 hours  
**Recommended execution:** 2-3 sprints over 2 weeks

---

## Files Modified (Projected)

1. ✅ **`notes/03-ai/authoring-guide.md`** (THIS PLAN - foundational guidance)
2. ⏳ `notes/03-ai/07_rag/README.md` (workflow structure, decision checkpoints)
3. ⏳ `notes/03-ai/07_rag/notebook_exercise.ipynb` (decision templates, industry callouts)
4. ⏳ `notes/03-ai/02_prompt_engineering/README.md` (iterative refinement workflow)
5. ⏳ `notes/03-ai/02_prompt_engineering/notebook_exercise.ipynb` (quality thresholds)
6. ⏳ `notes/03-ai/06_react/README.md` (tool orchestration workflow)
7. ⏳ `notes/03-ai/06_react/notebook_exercise.ipynb` (agent decision logic)
8. ⏳ `notes/03-ai/08_evaluation/README.md` (testing workflow)
9. ⏳ `notes/03-ai/08_evaluation/notebook_exercise.ipynb` (metric thresholds)
10. ⏳ `notes/03-ai/12_testing/README.md` (production testing workflow)
11. ⏳ `notes/03-ai/12_testing/notebook_exercise.ipynb` (test harness patterns)

**Status legend:**
- ✅ Complete
- ⏳ Pending
- 🔄 In progress
