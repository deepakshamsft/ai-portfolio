# Cost & Latency — Running AI Systems in Production

> **The story.** When the OpenAI API launched in June **2020**, GPT-3 cost $0.06 per 1 K tokens and almost nobody worried about it because almost nobody had production traffic. By **2023** the picture had inverted: every serious AI deployment was a finance problem. The optimisation stack was built fast. **KV-cache reuse** — keeping the attention key/value tensors hot across decoding steps — became standard once **Flash Attention** (Tri Dao, **2022**) made long contexts feasible. **Continuous batching** from the **Orca** paper (Yu et al., OSDI **2022**) replaced static batching and lifted serving throughput ~10×. **Speculative decoding** (Leviathan et al., Google, **2022**) slashed latency by drafting tokens with a small model and verifying with a large one. **Prompt caching** appeared in the **Anthropic** API in August 2024 and **OpenAI** in October 2024, cutting input-token cost ~90% for repeated prefixes. Open-source pricing collapsed in parallel — a 2026 query against a hosted Llama-class or DeepSeek model costs a small fraction of a 2023 GPT-4 call — and the cost-and-latency budget is what decides whether your system ships or stalls.
>
> **Where you are in the curriculum.** The gap between a demo and a production system is almost entirely cost and latency. A prototype that calls GPT-4 with a 50 K-token context, runs self-consistency 5×, and uses an LLM judge to evaluate every response works beautifully at zero users. At 10 000 users it costs thousands of dollars per day and responds in 30 seconds. This document maps the levers — model tier, context length, caching, streaming, batching, quantisation, speculative decoding — that turn that demo into something you can run on a budget. It is the closing chapter of the AI track and the bridge to the [AIInfrastructure](../../ai_infrastructure) track where these levers become hardware decisions.
>
> **Notation.** $c = \frac{n_\text{in} \cdot p_\text{in} + n_\text{out} \cdot p_\text{out}}{10^6}$ — cost per request in USD; $n_\text{in}, n_\text{out}$ — input and output token counts; $p_\text{in}, p_\text{out}$ — price per million tokens; $\lambda$ — request arrival rate (req/s); $\bar{t}$ — mean service time per request.

***

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — a production AI ordering system satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.1-9: All targets hit! 30% conversion ✅, $40.60 AOV ✅, 2.2s latency ✅, $0.010/conv ✅, safety validated ✅
- ✅ **Security audit**: Passed adversarial testing, approved for public launch
- 🎉 **Ready to ship**: Production-ready system with all constraints satisfied

**What's blocking us:**

🚨 **Cost per conversation needs optimization to hit 10.6 month ROI target**

**Current economics (pre-optimization):**
```
Cost breakdown per conversation:
- Fine-tuned Llama-3-8B inference (self-hosted): $0.004/conv
- RAG retrieval (embedding + vector search): $0.002/conv
- Safety validation (Azure Content Safety): $0.002/conv
- Guardrails overhead: $0.001/conv
- Monitoring/logging: $0.001/conv

Total: $0.010/conv

Monthly cost at 50 visitors/day:
- 50 visitors × 28% conversion = 14 orders/day = 420 orders/month
- 420 conv/month × $0.010 = $4.20/month infrastructure cost ✅

Current ROI:
- Revenue: 30% × $40.60 × 50 = $609/day = $18,270/month
- Labor savings: $11,064/month
- Total benefit: $18,270 - $12,705 + $11,064 = $16,629/month
- Payback: $300,000 / $16,629 = **18 months**

Target ROI: 10.6 months (need $28,302/month benefit)
```

**The problem:**
- Current payback: **18 months** (above 10.6 month target)
- To hit 10.6 months: need $28,302/month total benefit (currently $16,629/month)
- **Gap: $11,673/month**

**Two paths to close the gap:**
1. **Scale traffic**: 50 → 88 daily visitors (+76%) → hits 10.6 month target
2. **Optimize operations**: Reduce latency → better UX → higher conversion

**Why latency optimization matters:**

Current state: 2.2s p95 latency
```
User experience research:
- <1s response: "Instant" (phone-like experience)
- 1-2s response: "Fast" (acceptable)
- 2-3s response: "Noticeable delay" (current state)
- >3s response: "Slow" (cart abandonment increases)

Conversion correlation:
- <1.5s latency: 32% conversion (phone parity!)
- 2.0s latency: 30% conversion (current)
- 2.5s latency: 28% conversion
- 3.0s latency: 25% conversion (target threshold)

Opportunity: 2.2s → 1.5s latency = 30% → 32% conversion = +$2,435/month revenue
```

**What this chapter unlocks:**

🚀 **Cost & latency optimization stack:**
1. **Prompt caching**: Cache 50-token system prompt across requests (90% token reuse)
2. **Streaming responses**: Return first tokens immediately while generating rest (perceived latency <1s)
3. **KV-cache reuse**: Reuse attention tensors for repeated context prefixes
4. **Speculative decoding**: Draft with Llama-3-1B, verify with Llama-3-8B (30% faster)
5. **Batched inference**: Process multiple requests together (2× throughput)
6. **INT8 quantization**: Reduce model size from 16GB → 8GB (faster memory access)

⚡ **Expected improvements:**
- **Latency**: 2.2s → **1.5s p95** (32% reduction) → "fast" user experience
- **Conversion**: 30% → **32%** (latency-driven UX improvement)
- **Cost**: $0.010 → **$0.005/conv** (prompt caching + INT8 quantization)
- **Throughput**: 10 conv/sec → **20 conv/sec** (batched inference)
- **Infrastructure cost**: $4.20/month → **$2.10/month** (50% reduction)

**Final ROI calculation (Ch.10 complete):**
```
Revenue: 32% × $40.60 × 50 = $649.60/day = $19,488/month
Baseline revenue: 22% × $38.50 × 50 = $12,705/month
Revenue lift: $19,488 - $12,705 = $6,783/month

Labor savings: 70% reduction = $11,064/month

Total monthly benefit: $6,783 + $11,064 = $17,847/month
Payback period: $300,000 / $17,847 = **16.8 months**

Still above 10.6 month target, but:
- At 65 daily visitors: $300,000 / $23,291 = **12.9 months**
- At 88 daily visitors: $300,000 / $31,488 = **9.5 months** ✅ (beats 10.6 target!)
```

**Constraint status after Ch.10 (FINAL):**

| Constraint | Status | Final State |
|------------|--------|-------------|
| #1 BUSINESS VALUE | ✅ **TARGET MOSTLY HIT** | 32% conv (>25% ✅), +$2.10 AOV (target +$2.50, 84% achieved), 70% labor savings (✅) |
| #2 ACCURACY | ✅ **TARGET HIT** | ~5% error rate (target <5% ✅) |
| #3 LATENCY | ✅ **TARGET EXCEEDED** | **1.5s p95** (target <3s ✅, beats target by 50%!) |
| #4 COST | ✅ **TARGET EXCEEDED** | **$0.005/conv** (target <$0.08 ✅, 94% under budget!) |
| #5 SAFETY | ✅ **TARGET HIT** | <2% jailbreak vulnerability, 100% allergen validation ✅ |
| #6 RELIABILITY | ✅ **TARGET HIT** | >99% uptime, graceful degradation ✅ |

**Final verdict: READY FOR PRODUCTION LAUNCH** ✅

All 6 constraints satisfied. ROI achievable at 88 daily visitors (realistic with basic marketing).

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

> See [AIPrimer.md](../ai-primer.md) for the full system definition.

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

## 10 · Bridge

Cost & Latency completed the operational layer. You are now equipped with the full AI engineering stack: model fundamentals → prompting → reasoning → retrieval → vector storage → agent orchestration → evaluation → safety → operational costs. The consolidated [`InterviewGuides/`](../../interview_guides) is your synthesis resource — it covers all of these in the rapid-fire format interviewers use.

> *The best model for the job is the cheapest one that passes your eval threshold. Measure first. Spend last.*

---

## 11 · Progress Check — What We Can Solve Now

🎉 **FINAL MILESTONE**: All 6 constraints exceeded! Ready for production!

**Unlocked capabilities:**
- ✅ **Prompt caching**: 90% token reuse for system prompts
- ✅ **Streaming responses**: First tokens in <500ms (perceived instant)
- ✅ **KV-cache reuse**: Attention tensors cached across requests
- ✅ **Speculative decoding**: Llama-3-1B drafts, Llama-3-8B verifies (30% faster)
- ✅ **Batched inference**: 2× throughput for concurrent requests
- ✅ **INT8 quantization**: Model size 16GB → 8GB (faster memory access)

**Final progress toward constraints:**

| Constraint | Status | Final State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ✅ **TARGET MOSTLY HIT** | **32% conversion** (>25% ✅, beats phone 22%!), **+$2.10 AOV** (target +$2.50, 84% achieved), 70% labor savings (✅) |
| #2 ACCURACY | ✅ **TARGET HIT!** | ~5% error rate (target <5% ✅) — maintained through optimizations |
| #3 LATENCY | ✅ **TARGET CRUSHED!** | **1.5s p95** (target <3s ✅) — **beats target by 50%!** |
| #4 COST | ✅ **TARGET CRUSHED!** | **$0.005/conv** (target <$0.08 ✅) — **94% under budget!** |
| #5 SAFETY | ✅ **TARGET HIT!** | <2% jailbreak vulnerability, 100% allergen validation (✅) |
| #6 RELIABILITY | ✅ **TARGET HIT!** | >99% uptime, graceful degradation (✅) |

**What we can solve:**

✅ **Sub-2-second latency (1.5s p95) → phone-like UX**:
```
Latency breakdown before Ch.10 (2.2s p95):
- LLM inference: 1.5s
- RAG retrieval: 0.3s
- Safety validation: 0.2s
- Network overhead: 0.2s
Total: 2.2s

Latency breakdown after Ch.10 (1.5s p95):
- LLM inference (INT8 quant): 0.8s (-700ms, 47% faster)
- RAG retrieval (cached embeddings): 0.2s (-100ms)
- Safety validation (parallel): 0.2s (unchanged)
- Network overhead (streaming): 0.1s (-100ms)
- KV-cache reuse: -0.2s (attention cached)
Total: 1.5s

Improvement: 2.2s → 1.5s = -700ms (32% reduction)

User experience:
- 2.2s: "Noticeable delay" (30% conversion)
- 1.5s: "Fast" (32% conversion) ← **phone-like responsiveness!**
```

✅ **50% cost reduction ($0.010 → $0.005/conv)**:
```
Cost breakdown before Ch.10 ($0.010/conv):
- Fine-tuned Llama-3-8B inference: $0.004
- RAG retrieval (embedding + vector search): $0.002
- Safety validation: $0.002
- Guardrails + monitoring: $0.002
Total: $0.010/conv

Cost breakdown after Ch.10 ($0.005/conv):
- Llama-3-8B INT8 (50% faster inference): $0.002 (-$0.002)
- RAG retrieval (prompt caching): $0.0002 (-$0.0018, 90% cache hit)
- Safety validation (batched): $0.001 (-$0.001)
- Guardrails + monitoring: $0.0018 (-$0.0002)
Total: $0.005/conv

Savings: $0.010 → $0.005 = -$0.005/conv (50% reduction)

Monthly savings at 50 visitors/day:
- 420 orders/month × $0.005 savings = $2.10/month infrastructure savings
- (Minimal absolute savings, but massive % improvement and headroom for scale)
```

✅ **Streaming responses (perceived <1s latency)**:
```
Before streaming (2.2s total latency):
User: "What's your most popular pizza?"
[2.2s wait with loading spinner]
Bot: "Our Pepperoni pizza is the most popular choice..."

After streaming (1.5s total, <500ms to first token):
User: "What's your most popular pizza?"
[<500ms wait]
Bot: "Our Pepperoni pizza" [visible immediately]
     "is the most popular choice. It features..." [streams in real-time]

Perceived latency: <1s (user sees progress immediately)
Actual latency: 1.5s (unchanged)
UX impact: "Instant" feeling → conversion 30% → 32% (+2 points)
```

✅ **2× throughput with batched inference**:
```
Before batching (sequential processing):
- Request 1: 1.5s latency
- Request 2: waits for Request 1, then 1.5s
- Request 3: waits for Request 1+2, then 1.5s
Throughput: 1 request / 1.5s = 0.67 requests/sec

After batching (parallel processing):
- Requests 1, 2, 3 batched together
- All processed in single forward pass
- Each still completes in ~1.5s (shared overhead)
Throughput: 3 requests / 1.5s = 2 requests/sec

Result: ✅ 2× throughput! Can handle peak traffic (holiday ordering rush)
```

**Final business metrics:**
- **Order conversion**: **32%** (target >25% ✅, beats phone 22% ✅)
- **Average order value**: **$40.60** (+$2.10 vs. baseline $38.50 ✅)
- **Cost per conversation**: **$0.005** (target <$0.08 ✅, 94% under budget!)
- **Error rate**: **~5%** (target <5% ✅)
- **Latency**: **1.5s p95** (target <3s ✅, beats by 50%!)
- **Throughput**: **20 orders/sec** (vs. 10 before, 2× improvement)
- **Security**: <2% jailbreak vulnerability, 100% allergen validation ✅

**Final ROI calculation:**

**At 50 daily visitors (current):**
```
Revenue: 32% × $40.60 × 50 = $649.60/day = $19,488/month
Baseline: 22% × $38.50 × 50 = $423.50/day = $12,705/month
Revenue lift: $19,488 - $12,705 = $6,783/month

Labor savings: 70% reduction = $11,064/month

Total monthly benefit: $6,783 + $11,064 = $17,847/month
Payback period: $300,000 / $17,847 = **16.8 months**
```

**Scale scenario: 88 daily visitors (realistic with marketing):**
```
Revenue: 32% × $40.60 × 88 = $1,141.70/day = $34,251/month
Baseline: 22% × $38.50 × 88 = $743.46/day = $22,304/month
Revenue lift: $34,251 - $22,304 = $11,947/month

Labor savings: $11,064/month

Total monthly benefit: $11,947 + $11,064 = $23,011/month
Payback period: $300,000 / $23,011 = **13 months**
```

**Scale scenario: 120 daily visitors (aggressive marketing):**
```
Revenue: 32% × $40.60 × 120 = $1,559.04/day = $46,771/month
Baseline: 22% × $38.50 × 120 = $1,014/day = $30,420/month
Revenue lift: $46,771 - $30,420 = $16,351/month

Labor savings: $11,064/month

Total monthly benefit: $16,351 + $11,064 = $27,415/month
Payback period: $300,000 / $27,415 = **10.9 months ✅ (beats 10.6 month target!)**
```

**VERDICT: READY FOR PRODUCTION LAUNCH** ✅

All 6 constraints satisfied:
1. ✅ **BUSINESS VALUE**: 32% conversion (>25%), +$2.10 AOV (target +$2.50, 84% achieved), 70% labor savings
2. ✅ **ACCURACY**: ~5% error rate (<5%)
3. ✅ **LATENCY**: 1.5s p95 (<3s, beats by 50%!)
4. ✅ **COST**: $0.005/conv (<$0.08, 94% under budget!)
5. ✅ **SAFETY**: <2% jailbreak, 100% allergen validation
6. ✅ **RELIABILITY**: >99% uptime, graceful degradation

ROI achievable:
- **Conservative** (88 visitors/day): 13 months
- **Target** (120 visitors/day): 10.9 months ✅
- **Aggressive** (150+ visitors/day): <9 months

**System comparison: Final vs. Baseline**

| Metric | Phone Baseline | PizzaBot (Final) | Improvement |
|--------|----------------|------------------|-------------|
| Conversion | 22% | **32%** | +45% |
| AOV | $38.50 | **$40.60** | +5.5% |
| Error rate | ~3% (human) | **~5%** | Comparable |
| Latency | Instant (phone) | **1.5s** | Near-instant |
| Labor cost | $157,680/year | **$47,304/year** | -70% |
| Hours/day | 24/7 (3 shifts) | **24/7 (1 staff)** | -67% headcount |
| Cost/conv | ~$7.50 (labor) | **$0.005** | -99.9% |

**Why the CEO should greenlight launch:**

1. **All targets exceeded**: Not just hit, but crushed on latency (-50%) and cost (-94%)
2. **Beats phone baseline**: 32% vs. 22% conversion, $40.60 vs. $38.50 AOV
3. **Production-ready**: Security audit passed, automated testing, monitoring, graceful degradation
4. **Clear ROI path**: 10.9 months at 120 daily visitors (achievable with basic marketing)
5. **Future-proof**: 2× throughput headroom, 94% cost budget remaining for growth
6. **Competitive moat**: Brand voice fine-tuning creates differentiation impossible for competitors to copy

**Post-launch optimization opportunities:**
- Multi-location franchise expansion: HNSW indexing scales to 100+ locations
- Seasonal menu updates: Daily menu changes without downtime
- A/B testing: Safe experimentation with automated regression detection
- Advanced upselling: Personalized recommendations based on order history
- Voice interface: <1.5s latency enables phone integration

**This is the end of the AI track.** Ch.1-10 took PizzaBot from 8% conversion (failing prototype) to 32% conversion (production-ready system beating human baseline). Every chapter solved a specific technical challenge while maintaining the business value story.

**Key interview concepts from this chapter:**

## Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| How LLM API costs are structured: (input tokens × price/1M) + (output tokens × price/1M), with output typically 3–5× more expensive | A RAG system costs $3,000/month. Walk through how you'd diagnose and reduce that cost | "Switching to a smaller model always saves money" — smaller models may require more output tokens or more retries, eliminating the savings |
| What prefix caching is and how it eliminates redundant compute on stable prompt prefixes | Explain KV-cache and how speculative decoding reduces generation latency | Confusing time-to-first-token (TTFT) with total latency — for streaming UX, TTFT governs perceived responsiveness; optimising generation throughput does not help TTFT |
| The latency components: TTFT vs generation throughput, and which matters for interactive vs batch workloads | When should you use the batch API instead of the synchronous API, and what tradeoffs does it introduce? | Cache-busting by inserting dynamic content (timestamps, user IDs) into prompts that share a stable prefix — this prevents prefix-cache hits and multiplies costs |
| The cost–quality decision order: prompt engineering first, then RAG, then fine-tuning, then a larger model | How does semantic caching differ from exact-match caching and when does each make sense? | Measuring cost only at the generator API — embedding calls, re-ranker calls, and judge-model calls can collectively exceed the generator cost in a production RAG pipeline |

## Illustrations

![Cost and latency — token-cost stack, latency components, cost-vs-accuracy tiers, optimisation patterns](img/Cost%20and%20Latency.png)
