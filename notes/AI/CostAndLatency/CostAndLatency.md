# Cost & Latency — Running AI Systems in Production

> **The gap between a demo and a production system is almost entirely cost and latency.** A prototype that calls GPT-4 with a 50k-token context, runs self-consistency sampling 5 times, and uses an LLM judge to evaluate every response works beautifully at zero users. At 10,000 users, it costs thousands of dollars per day and responds in 30 seconds. This document maps the levers that control cost and latency in real deployments.

---

## 1 · Core Idea

Every LLM API call costs money and time. Both are functions of one thing: **the number of tokens processed**.

```
Cost    = (input_tokens  × $/1M input)  + (output_tokens × $/1M output)
Latency = time_to_first_token (TTFT)   + tokens_generated × ms/token
```

Every architectural decision — which model, how much context, how many calls, whether to stream — maps directly to these two formulas.

---

## 2 · Where Tokens Come From

In a RAG + agent pipeline, the token budget breaks down roughly as:

| Component | Typical token count | Notes |
|---|---|---|
| System prompt | 200–800 | Often fixed per deployment |
| Few-shot examples | 300–1500 | Per-call if not cached |
| Retrieved chunks (RAG) | 1k–8k | Scales with k and chunk size |
| Conversation history | 0–50k | Grows unboundedly in chat — biggest leak |
| User message | 10–500 | Relatively small |
| **Total input** | **~2k–60k** | Context window determines the ceiling |
| Model output | 100–2000 | Determined by task; agents generate more |

**The biggest cost leak in production:** conversation history. A chat app that passes the full conversation history to every API call has linearly growing costs per session. Solutions: summarise older turns, truncate aggressively, or store history in a vector DB and retrieve only the relevant parts.

---

## 3 · Model Cost Tiers (2025–2026 ballpark)

| Model tier | Cost range | When to use |
|---|---|---|
| Frontier (GPT-4o, Claude 3.5 Sonnet) | $2–15 / 1M tokens | Complex reasoning, high-stakes outputs, judge models |
| Mid-tier (GPT-4o-mini, Claude Haiku) | $0.15–1 / 1M tokens | Most retrieval and summarisation tasks |
| Open-source (Llama 3, Mistral 7B) | $0–0.3 / 1M tokens (self-hosted) | High-volume, latency-sensitive, private data |
| Embedding models | $0.01–0.13 / 1M tokens | Cheap; embed everything at ingestion, not query time if possible |

**Practical rule:** use the cheapest model that passes your evaluation threshold. Run `EvaluatingAISystems.md` metrics on both models. The quality difference between mid-tier and frontier is often smaller than expected for structured tasks with good prompts.

---

## 4 · Latency Components

```
Total latency = network RTT
              + time_to_first_token (TTFT)
              + (tokens_to_generate × ms_per_token)
              + (optional: NLI verification, self-consistency, judge call)

TTFT scales with:    input_token_count  (larger context → longer prefill → longer TTFT)
ms_per_token scales with:  model size    (larger model → lower tokens/sec)
```

### Streaming

Streaming returns tokens as they are generated rather than waiting for the full response. Time-to-first-token remains the same; perceived latency drops dramatically because the user sees text appearing.

**Use streaming whenever the output is text shown to a user.** Never stream when the application needs to parse the full response before acting (e.g., JSON tool calls in an agent loop).

### The KV Cache

Transformers cache the key-value matrices of previously computed tokens (the KV cache). For a repeated prefix (system prompt + few-shot examples), the provider can reuse the cached computation rather than recomputing it on every call.

```
Without prefix caching:   every call pays input_tokens × full prefill cost
With prefix caching:      every call pays only new_tokens × prefill cost
                          (cached prefix is free or deeply discounted)
```

**Implication:** put your system prompt and few-shot examples at the beginning of the context. Keep them identical across calls. Most major providers (OpenAI, Anthropic) offer prefix caching automatically or explicitly. Savings of 50–80% on input token costs for chat applications are typical.

---

## 5 · The Accuracy vs. Cost Tradeoff

Every accuracy-improving technique adds cost. Knowing the magnitude helps you decide what to afford.

| Technique | Accuracy gain | Cost multiplier | Latency multiplier |
|---|---|---|---|
| Switching to frontier model | +5–15% on complex tasks | 10–50× | 2–3× |
| Adding RAG (3 chunks) | +15–30% on factual tasks | 1.5–3× | 1.5–2× |
| Chain-of-thought | +10–25% on reasoning tasks | 2–4× (more output tokens) | 2–4× |
| Self-consistency (N=5) | +3–8% | 5× | 5× |
| LLM judge for every response | Evaluation quality | 1.5–3× | 1.5–3× |
| NLI claim verification | Hallucination detection | 1.1–1.3× (small model) | 1.1× |

**The hierarchy:** reach for cheaper techniques first. Self-consistency at 5× cost rarely beats a better prompt at 1× cost. An NLI model for hallucination detection costs 0.1× what an LLM judge costs.

---

## 6 · Cost Optimisation Patterns

### Prompt caching

Keep system prompts and few-shot examples structurally identical across calls. Any content the provider can cache doesn't get billed again.

### Tiered model routing

Route requests to the cheapest model that can handle them; escalate to a stronger model only on failure or low-confidence signal.

```python
def route_query(query: str) -> str:
    # Try the cheap model first
    result = cheap_model.generate(query)
    if low_confidence(result) or contains_complex_reasoning(query):
        result = frontier_model.generate(query)
    return result
```

### Batch processing

For non-interactive workloads (nightly summaries, document ingestion), use batch APIs (50% cost reduction on most providers). Latency is irrelevant; throughput is not.

### Context window discipline

Aggressively trim the context before each call:
1. Summarise conversation history older than N turns
2. Limit retrieved chunks to the minimum k that meets recall thresholds
3. Remove few-shot examples once the model has demonstrated the pattern (session-level cache)

### Request deduplication and caching

Cache model responses for identical (prompt, parameters) pairs. Semantic caching (cache responses for semantically similar queries) can achieve 20–40% cache hit rates on FAQ-style applications.

```python
import hashlib
_cache = {}

def cached_llm_call(prompt: str, model: str, temperature: float) -> str:
    key = hashlib.sha256(f"{prompt}|{model}|{temperature}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    result = llm_api.generate(prompt, model=model, temperature=temperature)
    _cache[key] = result
    return result
```

---

## 7 · Real Numbers — Cost Estimation

**Example: RAG question-answering at 10k queries/day**

```
Per query:
  Input tokens:  system_prompt(500) + chunks(2000) + query(100) = 2600 tokens
  Output tokens: answer ≈ 250 tokens

Daily cost (mid-tier model at $1/1M input, $3/1M output):
  Input:  10,000 × 2,600 × ($1/1,000,000)  = $26/day
  Output: 10,000 × 250  × ($3/1,000,000)   = $7.50/day
  Total:  ~$33/day  ≈  $1,000/month

With prefix caching on the 500-token system prompt:
  Cached 500 tokens free → saves $5/day → $150/month
  Total after caching:  ~$875/month
```

**Same workload on a self-hosted Llama 3 8B:**
```
Cloud GPU (A100 40GB): ~$1.5/hour → ~$1,080/month at 100% utilisation
Break-even vs. API: ~same, but with full data privacy and no per-token billing
```

---

## 8 · PizzaBot Connection

> See [AIPrimer.md](../AIPrimer.md) for the full system definition.

**Token budget per order request** (approximate, mid-tier model):

| Component | Tokens | Notes |
|---|---|---|
| System prompt | ~300 | Fixed; prefix-cached after first request |
| Conversation history | ~400 | ~4 prior turns summarised |
| Retrieved RAG chunks (k=3) | ~450 | Menu item + allergen entry + FAQ section |
| User message | ~40 | "Large Margherita + 2 garlic breads to 42 Maple St" |
| Tool call outputs (3 calls) | ~200 | Location + availability ×2 + order total |
| **Total input** | **~1,390** | Well within 8k context; cheap per call |
| Model output | ~150 | Order confirmation JSON + natural language summary |

**Cost at scale** (GPT-4o-mini at $0.15/1M input, $0.60/1M output):
```
Per order: (1390 × 0.00000015) + (150 × 0.00000060) ≈ $0.00021 + $0.00009 = $0.00030
10,000 orders/day: ~$3.00/day  (~$90/month)
```

**Optimisations applied:**
- **Prefix caching** on the system prompt — 300 tokens cached after the first call per session (saves ~$0.00004/call)
- **Semantic caching** on frequently repeated queries ("Is the Margherita gluten-free?") — cache hit rate ~30% for a menu of 20 items
- **Mid-tier model** selected over frontier after eval showed quality parity on structured order confirmation

---

## 9 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The cost formula: (input × $/1M) + (output × $/1M) | Where does conversation history cause cost to blow up, and how do you fix it? | Saying streaming reduces latency — it reduces *perceived* latency; TTFT is unchanged |
| What the KV cache is and why keeping system prompts identical matters | When would you use a self-hosted model vs. an API? | Ignoring the cost of output tokens — for long-form generation they dominate |
| The accuracy vs. cost tradeoff table (which techniques are cheap, which are expensive) | How would you estimate monthly API cost for a given application? | Saying self-consistency is always worth it — 5× cost for +5% accuracy is rarely a good trade |
| Streaming: when to use it and when not to | What is semantic caching and what cache hit rate is realistic? | Treating all models as equivalent — model tier selection is the single biggest cost lever |

---

## 9 · Bridge

Cost & Latency completed the operational layer. You are now equipped with the full AI engineering stack: model fundamentals → prompting → reasoning → retrieval → vector storage → agent orchestration → evaluation → safety → operational costs. The `AI_Interview_Primer.md` is your synthesis resource — it covers all of these in the rapid-fire format interviewers use.

> *The best model for the job is the cheapest one that passes your eval threshold. Measure first. Spend last.*

---

## Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| How LLM API costs are structured: (input tokens × price/1M) + (output tokens × price/1M), with output typically 3–5× more expensive | A RAG system costs $3,000/month. Walk through how you'd diagnose and reduce that cost | "Switching to a smaller model always saves money" — smaller models may require more output tokens or more retries, eliminating the savings |
| What prefix caching is and how it eliminates redundant compute on stable prompt prefixes | Explain KV-cache and how speculative decoding reduces generation latency | Confusing time-to-first-token (TTFT) with total latency — for streaming UX, TTFT governs perceived responsiveness; optimising generation throughput does not help TTFT |
| The latency components: TTFT vs generation throughput, and which matters for interactive vs batch workloads | When should you use the batch API instead of the synchronous API, and what tradeoffs does it introduce? | Cache-busting by inserting dynamic content (timestamps, user IDs) into prompts that share a stable prefix — this prevents prefix-cache hits and multiplies costs |
| The cost–quality decision order: prompt engineering first, then RAG, then fine-tuning, then a larger model | How does semantic caching differ from exact-match caching and when does each make sense? | Measuring cost only at the generator API — embedding calls, re-ranker calls, and judge-model calls can collectively exceed the generator cost in a production RAG pipeline |

## Illustrations

![Cost and latency — token-cost stack, latency components, cost-vs-accuracy tiers, optimisation patterns](img/Cost%20and%20Latency.png)

## Illustrations

![Cost and latency — token-cost stack, latency components, cost-vs-accuracy tiers, optimisation patterns](img/Cost%20and%20Latency.png)
