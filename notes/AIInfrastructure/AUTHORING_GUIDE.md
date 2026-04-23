# AI Infrastructure Track — Authoring Guide

> **Purpose**: This guide defines the unified Grand Challenge framework that threads through all 10 AI Infrastructure chapters, ensuring every concept lands in a concrete, business-driven context.

---

## The Running Example — InferenceBase Startup

Every chapter uses **one consistent system**: **InferenceBase** — a seed-stage AI startup building a document intelligence API.

**The scenario**: *You're the founding Platform Engineer at InferenceBase. The product takes enterprise PDFs, runs them through an LLM, and returns structured JSON. The CEO just forwarded the latest AWS bill — $80,000/month in OpenAI API charges — and asked you to evaluate whether self-hosting Llama-3-8B makes economic sense. You have a $15,000/month cloud compute budget and two weeks to deliver a recommendation.*

**Current state**:
- Product: Document intelligence API (extract structured data from PDFs using LLM)
- Traffic: ~10,000 requests/day
- Current cost: $80,000/month in OpenAI API calls
- Target: Replace with self-hosted Llama-3-8B on $15,000/month budget
- Constraint: 2-week evaluation window before next board meeting

**Business requirements (what success looks like)**:
- **Cost**: <$15,000/month (vs. $80k OpenAI baseline)
- **Latency**: ≤2s p95 (match current OpenAI latency)
- **Throughput**: ≥10,000 requests/day (current traffic)
- **Quality**: ≥95% answer accuracy (Llama-3-8B vs GPT-3.5-turbo baseline)
- **Reliability**: >99% uptime
- **Timeline**: Production-ready in 8 weeks (2 weeks evaluation + 6 weeks implementation)

---

## The Grand Challenge — Cost-Effective Self-Hosting

The overarching question: **"Can we self-host Llama-3-8B for <$15k/month and match OpenAI's performance?"**

### The 6 Technical Constraints

| # | Constraint | Target | Why it matters |
|---|------------|--------|----------------|
| **#1** | **COST** | <$15,000/month compute | CEO's hard budget limit - 81% cost reduction from $80k OpenAI baseline |
| **#2** | **LATENCY** | ≤2s p95 | Current OpenAI SLA - users expect instant responses |
| **#3** | **THROUGHPUT** | ≥10,000 req/day | Current traffic - need to handle existing load |
| **#4** | **MEMORY** | Fit Llama-3-8B in available GPU VRAM | Model must load without OOM errors |
| **#5** | **QUALITY** | ≥95% accuracy | Llama-3-8B must match GPT-3.5-turbo quality on document extraction |
| **#6** | **RELIABILITY** | >99% uptime | Production SLA - cannot lose requests during deployment |

### Constraint Progression Across Chapters

Each chapter solves a specific bottleneck on the path to production:

| Ch | Chapter | Constraint Focus | Progress |
|----|---------|------------------|----------|
| **1** | GPU Architecture | #4 (Memory) | Understand hardware specs → identify RTX 4090 as candidate ($1.50/hr) |
| **2** | Memory Budgets | #4 (Memory) | Calculate exact VRAM: 16GB params + 4GB KV cache = 20GB → fits in 24GB |
| **3** | Quantization | #1 (Cost), #4 (Memory) | INT4 quantization → 8GB params, enables multi-batch → cuts cost 60% |
| **4** | Distributed Training | (Training focus) | Data parallelism for fine-tuning → not blocking inference launch |
| **5** | Inference Optimization | #2 (Latency), #3 (Throughput) | PagedAttention + batching → 2x throughput, 1.2s p95 latency ✅ |
| **6** | Serving Frameworks | #2 (Latency), #3 (Throughput) | vLLM benchmark: 12k req/day on 1x RTX 4090 → need 1 GPU ✅ |
| **7** | Networking | #3 (Throughput), #6 (Reliability) | NVLink multi-GPU → 40k req/day capacity for growth |
| **8** | Cloud Infrastructure | #1 (Cost) | RunPod RTX 4090: $1.50/hr × 730 hr = $1,095/month ✅ (vs $15k budget) |
| **9** | MLOps | #6 (Reliability) | Checkpointing + monitoring → 99.5% uptime ✅ |
| **10** | Production Platform | All constraints | Full stack: LB → vLLM → GPU → monitoring → ALL TARGETS MET ✅ |

### Final System Status

**Ch.10 delivers**:
- ✅ **Cost**: $1,095/month (93% under budget, 98.6% savings vs $80k baseline)
- ✅ **Latency**: 1.2s p95 (40% better than 2s target)
- ✅ **Throughput**: 12,000 req/day (120% of target, with headroom for growth)
- ✅ **Memory**: 8GB INT4 model + 4GB KV cache = 12GB used (50% of 24GB VRAM)
- ✅ **Quality**: 96.2% accuracy (vs 95% target, 1.2% below GPT-3.5-turbo)
- ✅ **Reliability**: 99.5% uptime (above 99% target)

**ROI**: $948,540/year savings ($80k → $1.1k/month), 2-week implementation → immediate payback

---

## Chapter Structure — Standard Template

Every chapter follows this structure to maintain consistency:

### § 0 · The Challenge — Where We Are (NEW!)

```markdown
## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ [List progress from previous chapters]
- ⚡ **Current state**: [Key metrics]

**What's blocking us**:

🚨 **[Specific technical problem this chapter solves]**

**Current situation**: [Concrete failure scenario]

**Problems**:
1. ❌ [Problem 1 with impact]
2. ❌ [Problem 2 with impact]
...

**Business impact**:
- [How this blocker affects the $15k budget constraint]
- [How this affects latency/throughput targets]
- [CEO/board pressure point]

**What this chapter unlocks**:

🚀 **[Core capability]**:
1. [Specific technique/solution 1]
2. [Specific technique/solution 2]
...

⚡ **Expected improvements**:
- **[Metric 1]**: X → Y (improvement %)
- **[Metric 2]**: X → Y (improvement %)
...

**Constraint status after this chapter**:
- #1 (Cost): [Status - met/on track/blocked]
- #2 (Latency): [Status]
... [all 6 constraints]
```

### § N · Progress Check — What We Can Solve Now (NEW!)

```markdown
## N · Progress Check — What We've Accomplished

🎉 **[Major milestone achieved]**

**Unlocked capabilities**:
- ✅ [Capability 1 with concrete metric]
- ✅ [Capability 2 with concrete metric]
...

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ⚡/✅/❌ | [Specific number vs target] |
| #2 LATENCY | ⚡/✅/❌ | [Specific number vs target] |
| #3 THROUGHPUT | ⚡/✅/❌ | [Specific number vs target] |
| #4 MEMORY | ⚡/✅/❌ | [Specific number vs target] |
| #5 QUALITY | ⚡/✅/❌ | [Specific number vs target] |
| #6 RELIABILITY | ⚡/✅/❌ | [Specific number vs target] |

**What we can solve now**:

✅ **[Specific problem]**:
```
[Concrete example showing before/after with real numbers]
```

**What's still blocking**:
- ❌ [Remaining issue 1] → needs [next chapter]
- ❌ [Remaining issue 2] → needs [chapter X]

**Next chapter**: [ChapterName] unlocks [capability] → [specific metric improvement]

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| [Concept 1] | [Question 1] | [Common mistake 1] |
| [Concept 2] | [Question 2] | [Common mistake 2] |
...
```

---

## Writing Guidelines

### 1. InferenceBase-First Framing

Every concept must land in InferenceBase context:

❌ **Generic**: "Tensor Cores accelerate matrix multiplication"
✅ **InferenceBase**: "Llama-3-8B's attention layers require 16 TFLOP/inference. Tensor Cores deliver 165 TFLOP/s, so compute isn't the bottleneck — memory bandwidth is"

### 2. Numbers Are Non-Negotiable

Every claim needs a number tied to InferenceBase:

❌ **Vague**: "Quantization reduces memory"
✅ **Specific**: "INT4 quantization: 16GB FP16 → 8GB INT4 = 50% VRAM reduction → enables batch size 4 → 4× throughput"

### 3. Show the Constraint Trade-offs

Make tensions explicit:

```markdown
**Constraint conflict**:
- Batching increases throughput (✅ #3) but increases latency (❌ #2)
- Solution: PagedAttention allows batch=4 without latency spike (1.2s p95 maintained)
```

### 4. Progressive Reveal

Each chapter builds on previous unlocks:

- Ch.1: "RTX 4090 has 24GB VRAM" → **But will the model fit?**
- Ch.2: "16GB params + 4GB KV = 20GB → yes!" → **But can we go smaller?**
- Ch.3: "INT4 → 8GB params" → **But how do we serve efficiently?**
- Ch.5: "PagedAttention + batching" → **But which framework?**
- Ch.6: "vLLM wins benchmark" → **Now launch!**

### 5. Diagram Requirements

Every chapter must include:

1. **Architecture diagram**: Show how component fits in full stack
2. **Before/After comparison**: Visualize the improvement (e.g., memory breakdown FP16 vs INT4)
3. **Bottleneck diagram**: Show where constraint was blocking (e.g., Roofline plot)
4. **Solution diagram**: Show how technique solves bottleneck (e.g., batching timeline)

Reference diagrams in text:
```markdown
![Memory breakdown — FP16 vs INT4 quantization showing 50% VRAM reduction](img/memory-breakdown.png)
```

---

## Constraint Evidence Standards

When claiming a constraint is met, provide:

### #1 Cost (<$15k/month)

```
GPU cost: RunPod RTX 4090 @ $1.50/hr × 730 hr/mo = $1,095/mo ✅
Inference cost: $1,095 / 12,000 req/day / 30 days = $0.003/req
Savings: $80,000 - $1,095 = $78,905/mo (98.6% reduction) ✅
```

### #2 Latency (≤2s p95)

```
Measured p95 latency: 1.2s (vLLM benchmark on RTX 4090, batch=4)
Target: ≤2s ✅ (40% headroom)
Breakdown: 200ms prompt processing + 1,000ms generation (50 tokens @ 50 tok/s)
```

### #3 Throughput (≥10k req/day)

```
Measured: 12,000 req/day on 1× RTX 4090 (vLLM continuous batching)
Target: ≥10,000 req/day ✅ (120% of target, 20% growth headroom)
Bottleneck: Memory bandwidth (989 GB/s effective vs 1,008 GB/s peak)
```

### #4 Memory (Fit in VRAM)

```
Model: 8GB (INT4 quantized Llama-3-8B)
KV cache: 4GB (batch=4, seq_len=2048)
Activations: 2GB (forward pass)
Total: 14GB used / 24GB available ✅ (58% utilization, room for batch growth)
```

### #5 Quality (≥95% accuracy)

```
Llama-3-8B INT4: 96.2% extraction accuracy on InferenceBase eval set
GPT-3.5-turbo baseline: 97.4%
Delta: -1.2 percentage points (acceptable for 98.6% cost savings)
Target: ≥95% ✅
```

### #6 Reliability (>99% uptime)

```
Measured uptime: 99.5% over 30-day test period
Downtime: 3.6 hours (checkpoint restore after preemption)
Target: >99% ✅
Mitigation: Checkpoint every 10 min → <10 min recovery time
```

---

## Example: § 0 Challenge for Ch.3 (Quantization)

```markdown
## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1: Identified RTX 4090 as target GPU (24GB VRAM, $1.50/hr)
- ✅ Ch.2: Calculated exact memory: 16GB params + 4GB KV cache = 20GB → fits!
- ⚡ **Current metrics**: 20GB VRAM used (83% of 24GB), batch size = 1, throughput = 3,000 req/day

**What's blocking us**:

🚨 **VRAM headroom exhausted — cannot increase batch size for throughput**

**Current situation**: Running Llama-3-8B FP16 on RTX 4090

```
VRAM breakdown (24GB total):
- Model parameters (FP16): 16GB
- KV cache (batch=1, seq=2048): 4GB
- Activations (forward pass): 2GB
- Available headroom: 2GB

Current performance:
- Batch size: 1 (cannot increase — would OOM)
- Throughput: 3,000 req/day (30% of 10k target) ❌
- Latency: 2.8s p95 (above 2s target) ❌
- Cost: $1,095/month (within budget) ✅
```

**Problems**:
1. ❌ **Cannot batch requests**: 20GB model leaves only 4GB for KV cache (batch=1 max)
2. ❌ **Low throughput**: Single-request processing → 3,000 req/day (need 10,000)
3. ❌ **Latency target missed**: Sequential processing → 2.8s p95 (need ≤2s)
4. ❌ **No growth headroom**: 83% VRAM utilization → cannot scale traffic

**Business impact**:
- Throughput shortfall: Can only handle 30% of current traffic → **cannot launch**
- Missing latency target: 2.8s > 2s SLA → users will complain
- Zero scaling capacity: Traffic spike → immediate OOM crash
- CEO: "If we can't handle current load, self-hosting is a non-starter. What's the fix?"

**What this chapter unlocks**:

🚀 **Quantization techniques to shrink model footprint**:
1. **INT8 quantization**: 16GB → 8GB params (50% reduction)
2. **INT4 quantization**: 16GB → 4GB params (75% reduction) 
3. **GPTQ/AWQ**: Post-training quantization (no retraining needed)
4. **Perplexity benchmarking**: Validate quality doesn't collapse

⚡ **Expected improvements**:
- **VRAM**: 20GB → 12GB total (16GB params → 8GB INT4, KV cache unchanged)
- **Batch size**: 1 → 4 (12GB freed headroom enables 4× KV cache)
- **Throughput**: 3,000 → 12,000 req/day (4× from batching) ✅ Exceeds 10k target!
- **Latency**: 2.8s → 1.2s p95 (batching amortizes overhead) ✅
- **Quality**: 97.4% → 96.2% accuracy (1.2 point drop, but still >95% target) ✅

**Constraint status after Ch.3**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month, well under $15k)
- #2 (Latency): ✅ **TARGET HIT!** (2.8s → 1.2s p95, beats 2s target by 40%)
- #3 (Throughput): ✅ **TARGET HIT!** (3,000 → 12,000 req/day, 120% of 10k target)
- #4 (Memory): ✅ **OPTIMIZED** (20GB → 12GB, 50% VRAM utilization, room to grow)
- #5 (Quality): ✅ **TARGET HIT!** (96.2% accuracy, above 95% threshold)
- #6 (Reliability): ⚡ **ON TRACK** (pending Ch.9 MLOps)

Quantization unlocks batching → meets 3 of 6 core constraints in one chapter!
```

---

## Validation Checklist

Before marking a chapter complete, verify:

- [ ] § 0 Challenge section present with InferenceBase failure scenario
- [ ] § N Progress Check section with constraint status table
- [ ] All 6 constraints referenced with specific metrics
- [ ] Business impact clearly stated (cost/latency/throughput)
- [ ] Before/after numbers shown for all improvements
- [ ] At least 3 diagrams referenced (architecture, bottleneck, solution)
- [ ] "Next chapter" bridge explains what's still blocking
- [ ] Interview checklist updated with chapter-specific concepts
- [ ] No generic examples — everything ties to InferenceBase
- [ ] Constraint progression consistent with track arc

---

## Summary — The Through-Line

**The story in one sentence**: InferenceBase starts with an $80k/month OpenAI bill and 2 weeks to decide if self-hosting is viable — by Ch.10, they've launched a production system on $1,095/month that beats OpenAI's latency and saves $948k/year.

**The arc**:
1. Ch.1-2: "Can we even run this model?" → Yes, RTX 4090 works
2. Ch.3: "Can we make it efficient?" → INT4 quantization unlocks batching
3. Ch.4: (Training) → "Can we fine-tune?" → Yes, but not blocking launch
4. Ch.5-6: "Can we serve at scale?" → vLLM + PagedAttention → 12k req/day
5. Ch.7: "Can we grow?" → Multi-GPU → 40k req/day capacity
6. Ch.8: "What's the real cost?" → $1,095/month RunPod → 98.6% savings
7. Ch.9-10: "Is it production-ready?" → Monitoring + reliability → **Ship it!**

Every chapter solves one specific bottleneck on the path from "$80k problem" to "$1k solution."
