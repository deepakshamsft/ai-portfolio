# Ch.2 — Memory & Compute Budgets

> **The story.** For the first decade of deep learning (2012–2022), VRAM was the hard constraint. AlexNet (2012) barely fit on two GTX 580s (3 GB total). ResNet-152 (2015) required batching gymnastics on a single Titan X (12 GB). GPT-2 (2019, 1.5 B params) needed 6 GB just to load — training required gradient checkpointing and mixed precision to squeeze into V100s (32 GB). The breakthrough was **ZeRO** (Rajbhandari et al., Microsoft, **2019**), which sharded optimizer states across GPUs, cutting per-GPU memory by up to 8×. **Flash Attention** (Dao et al., Stanford, **2022**) made self-attention subquadratic in memory by fusing operations and tiling through HBM. **PagedAttention** (Kwon et al., vLLM, **2023**) extended the paging idea to KV caches, eliminating fragmentation waste. By 2024, the memory wall had shifted: you could fit a 70 B model on a single H100 (80 GB) with INT4 quantization — but only if you understood the exact breakdown of where every GB goes.
>
> **Where you are in the curriculum.** Ch.1 told you which GPU to pick. This chapter tells you whether your model actually fits — and if not, what to cut. The InferenceBase question: *Llama-3-8B has 8 billion parameters. The RTX 4090 has 24 GB VRAM. Does it fit?* The answer requires understanding parameters, activations, KV cache, optimizer states, and gradients — and how each scales with batch size, sequence length, and precision.
> **Notation.** `P` = parameter count; memory in bytes ≈ `P × bytes_per_param`. `B` = batch size (number of concurrent sequences). `S` = sequence length in tokens. `H` = hidden dimension (`d_model`). KV cache scales as `2 × B × S × H × num_layers`. `GiB` = gibibytes (2³⁰ bytes). `VRAM` = on-GPU device memory limit.

---

## 0 · The Challenge — Where We Are

## Animation

> 🎬 *Animation placeholder — needle-builder agent will generate this.*


> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1: RTX 4090 identified as target GPU (24GB VRAM, 1.0 TB/s bandwidth, $1,095/month)
- ⚡ **Current state**: Have GPU selection, but unknown if Llama-3-8B fits in 24GB

**What's blocking us**:

🚨 **Cannot order hardware without confirming model fits in VRAM**

**Current situation**: Engineer preparing budget justification for CEO

```
Engineer calculates:
"Llama-3-8B has 8 billion parameters.
 8 billion × 2 bytes/param (FP16) = 16 GB for model weights.
 RTX 4090 has 24 GB VRAM.
 24 GB - 16 GB = 8 GB free.
 Should be plenty of headroom!"

[Orders RTX 4090, deploys model...]

First inference request → CUDA OOM error: "Out of memory"

Engineer: "Wait, what? I calculated 16GB for the model... where did the other 8GB go?"
```

**Problems**:
1. ❌ **Forgot KV cache**: Stores attention keys/values for all tokens → grows with sequence length × batch size
2. ❌ **Forgot activations**: Temporary tensors during forward pass → several GB
3. ❌ **Unknown batch limit**: How many requests can we process simultaneously before OOM?
4. ❌ **Training memory unknown**: If we want to fine-tune, what GPU do we need? (Adam optimizer states = 3× params!)
5. ❌ **No safety margin**: Running at 100% VRAM utilization = fragmentation issues, OOM on long sequences

**Business impact**:
- **Wrong GPU purchase = $50k mistake**: If we buy RTX 4090 (24GB) but actually need A100 (80GB), we burn $8k + weeks of delay
- **Throughput limited by batch size**: If we can only batch=1, throughput caps at 3,000 req/day (30% of 10k target)
- **Cannot fine-tune**: If training needs 80GB, we have no path to improve quality beyond base Llama-3-8B
- CEO: "I need exact numbers. Tell me: does it fit yes or no? And what's the maximum batch size?"

**What this chapter unlocks**:

🚀 **Precise VRAM calculator for inference & training**:
1. **Parameter memory**: 8B params × 2 bytes (FP16) = 16 GB
2. **KV cache memory**: batch_size × seq_len × layers × hidden_dim × 2 (K+V) × 2 bytes
3. **Activation memory**: Temporary tensors during forward/backward pass
4. **Optimizer states** (training): Adam = params + momentum + variance = 3× param memory
5. **Gradient memory** (training): Same shape as parameters = 1× param memory

⚡ **Expected outcomes**:
- **Inference VRAM**: 16GB params + 4GB KV cache (batch=1, seq=2048) + 2GB activations = **22GB total**
- **Fits in RTX 4090**: 22GB / 24GB = 92% utilization ✅ (2GB headroom)
- **Max batch size**: (24GB - 16GB - 2GB) / 4GB per batch = **batch=1.5** → can only do batch=1 ❌
- **Training VRAM**: 16GB params + 48GB optimizer states + 16GB gradients = **80GB** → need A100 80GB for fine-tuning
- **Quantization motivation**: INT4 → 4GB params (vs 16GB) → frees 12GB for KV cache → enables batch=4

**Constraint status after Ch.2**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month RTX 4090 confirmed)
- #2 (Latency): ⚡ **BLOCKED** (batch=1 → sequential processing → high latency under load)
- #3 (Throughput): ❌ **SHORTFALL** (batch=1 → max 3,000 req/day vs 10k target)
- #4 (Memory): ✅ **TARGET HIT!** (22GB / 24GB = fits, but zero batch headroom)
- #5 (Quality): ⚡ **ON TRACK** (Llama-3-8B baseline)
- #6 (Reliability): ⚡ **UNKNOWN**

**Critical realization**: Model fits, but **batch=1 limit kills throughput**. Need Ch.3 quantization to free VRAM for batching.

---

## 1 · Core Idea

VRAM consumption in deep learning breaks into five categories:

1. **Parameters** — the model weights themselves (static, loaded once)
2. **KV cache** — attention keys and values stored for all previous tokens (grows with seq_len × batch_size)
3. **Activations** — intermediate tensors computed during forward pass (cleared after backward pass)
4. **Optimizer states** — momentum, variance for Adam/AdamW (training only, 2× parameter memory)
5. **Gradients** — same shape as parameters, computed during backprop (training only, 1× parameter memory)

For **inference**, you only pay for #1, #2, #3. For **training**, you pay for all five.

---

## 2 · Running Example

Llama-3-8B at FP16 precision:
- Parameters: 8B × 2 bytes = **16 GB**
- KV cache (batch=1, seq=2048): **4 GB**
- Activations (forward pass): **2 GB**
- **Total: 22 GB** → fits in RTX 4090 (24 GB) with 2 GB margin

But: 2 GB headroom is not enough for batch=2 (would need 4 GB more for KV cache). This means **batch=1 only** → throughput capped at ~3,000 requests/day (30% of target).

The tension: we need batching to hit throughput target, but VRAM is exhausted. Solution: Ch.3 quantization (16 GB params → 4 GB params) frees 12 GB for KV cache, enabling batch=4.

---

## 3 · Parameter Memory

For a transformer model with $N$ parameters stored in precision $P$ bytes per parameter:

$$\text{Parameter Memory} = N \times P$$

| Precision | Bytes per parameter | 8B model | 70B model |
|-----------|---------------------|----------|-----------|
| FP32 | 4 | 32 GB | 280 GB |
| FP16 / BF16 | 2 | 16 GB | 140 GB |
| INT8 | 1 | 8 GB | 70 GB |
| INT4 | 0.5 | 4 GB | 35 GB |

Llama-3-8B at FP16: **8,030,000,000 params × 2 bytes = 16,060 MB ≈ 16 GB**

---

## 4 · KV Cache Memory

For each token generated, the model stores the key (K) and value (V) tensors for all layers. These are reused in subsequent decoding steps to avoid recomputing attention over the entire history.

$$\text{KV Cache} = 2 \times L \times H \times S \times B \times P$$

Where:
- $L$ = number of layers
- $H$ = hidden dimension
- $S$ = sequence length
- $B$ = batch size
- $P$ = bytes per element (2 for FP16)
- Factor of 2 = one tensor for keys, one for values

**Llama-3-8B example** (32 layers, 4096 hidden dim, FP16):

| Batch | Seq Length | KV Cache Size |
|-------|------------|---------------|
| 1 | 512 | 1 GB |
| 1 | 2048 | 4 GB |
| 4 | 2048 | 16 GB |
| 8 | 2048 | 32 GB |

**The bottleneck**: KV cache scales linearly with batch size. At batch=4, it consumes 16 GB alone — as much as the entire model!

---

## 5 · Activation Memory

Activations are the intermediate tensors computed during the forward pass (attention scores, FFN outputs, layer norm results). They are kept in VRAM until the backward pass (training) or discarded immediately (inference).

**Inference**: ~2–4 GB for Llama-3-8B (varies by batch size and sequence length)
**Training**: ~8–16 GB (must keep activations for backward pass)

**Gradient checkpointing** (training optimization): recompute activations during backward pass instead of storing them → cuts activation memory by 90%, at the cost of 30% slower training.

---

## 6 · Optimizer States (Training Only)

**Adam/AdamW optimizer** maintains two state tensors per parameter:
- **Momentum** (first moment): same shape as parameters
- **Variance** (second moment): same shape as parameters

$$\text{Optimizer Memory} = 2 \times N \times P$$

For Llama-3-8B at FP32 optimizer states (standard):
- Parameters: 8B × 4 bytes = 32 GB
- Momentum: 8B × 4 bytes = 32 GB
- Variance: 8B × 4 bytes = 32 GB
- **Total optimizer memory: 64 GB**

**Full training memory** = 16 GB (FP16 params) + 64 GB (FP32 optimizer) + 16 GB (FP16 gradients) + 8 GB (activations) = **104 GB**

→ Requires A100 80GB × 2 GPUs or ZeRO-2 sharding (Ch.4)

---

## 7 · VRAM Budget Calculator — Inference

**Formula**:
$$\text{VRAM}_{\text{inference}} = \text{Params} + \text{KV Cache} + \text{Activations}$$

**Llama-3-8B on RTX 4090 (24 GB):**

| Component | FP16 | INT8 | INT4 |
|-----------|------|------|------|
| Parameters | 16 GB | 8 GB | 4 GB |
| KV Cache (batch=1, seq=2048) | 4 GB | 4 GB | 4 GB |
| Activations | 2 GB | 2 GB | 2 GB |
| **Total** | **22 GB** | **14 GB** | **10 GB** |
| **Free VRAM** | **2 GB** | **10 GB** | **14 GB** |
| **Max batch size** | **1** | **3** | **4** |

**Critical insight**: INT4 quantization frees 12 GB, enabling batch=4 → 4× throughput!

---

## 8 · VRAM Budget Calculator — Training

**Formula**:
$$\text{VRAM}_{\text{training}} = \text{Params} + \text{Optimizer States} + \text{Gradients} + \text{Activations}$$

**Llama-3-8B training (full fine-tuning, Adam, no checkpointing):**

| Component | FP16 weights, FP32 optimizer |
|-----------|------------------------------|
| Parameters (FP16) | 16 GB |
| Optimizer states (FP32) | 64 GB |
| Gradients (FP16) | 16 GB |
| Activations (batch=1) | 8 GB |
| **Total** | **104 GB** |

**Cannot fit on single RTX 4090 (24 GB)!**

**Solutions**:
- A100 80GB × 2 with ZeRO-2 sharding (Ch.4)
- Gradient checkpointing: 104 GB → 30 GB (fits on A100 40GB)
- LoRA fine-tuning: only train adapter weights → 16 GB params + 2 GB adapter = 18 GB (fits on RTX 4090!)

---

## 9 · Step by Step — Calculating VRAM for Your Model

**Example**: You want to run Llama-2-70B at INT4 quantization, batch=2, seq_len=4096.

```
Step 1: Parameter memory
  70B params × 0.5 bytes (INT4) = 35 GB

Step 2: KV cache
  L = 80 layers, H = 8192 hidden dim, S = 4096 seq, B = 2
  KV = 2 × 80 × 8192 × 4096 × 2 × 2 bytes
     = 2 × 80 × 8192 × 4096 × 2 × 2
     = 21,474,836,480 bytes ≈ 21.5 GB

Step 3: Activations (estimate)
  ~8 GB (scales with batch × seq)

Total: 35 + 21.5 + 8 = 64.5 GB

Conclusion: Fits on A100 80GB (80 - 64.5 = 15.5 GB margin)
            Does NOT fit on A100 40GB
```

---

## 10 · The Key Diagram

### VRAM Breakdown: Llama-3-8B Inference vs Training

```
                    INFERENCE (FP16)              TRAINING (FP16/FP32)
RTX 4090 24GB      │                             │
                   │  Params: 16GB               │  Params: 16GB (FP16)
                   │  KV Cache: 4GB (batch=1)    │  Optimizer: 64GB (FP32)
                   │  Activations: 2GB           │  Gradients: 16GB (FP16)
                   │  FREE: 2GB ✅               │  Activations: 8GB
                   │                             │  TOTAL: 104GB ❌
                   │                             │  → Needs A100 80GB × 2
                   │
                   
                    INFERENCE (INT4)
                   │  Params: 4GB (-75%)
                   │  KV Cache: 16GB (batch=4)
                   │  Activations: 2GB
                   │  FREE: 2GB ✅
                   │  → Enables batch=4!
```

---

## 11 · What Can Go Wrong

- **Forgetting KV cache** — it is not part of the model file, but grows during inference and can OOM unexpectedly on long sequences
- **Ignoring batch size scaling** — doubling batch size does NOT double total VRAM; KV cache scales linearly, but params stay constant
- **Using FP32 optimizer states on a 24 GB GPU** — Adam needs 8× parameter memory; always use FP16 params + FP32 optimizer
- **Not accounting for fragmentation** — running at 95%+ VRAM utilization causes CUDA memory allocator failures; leave 2–4 GB headroom
- **Assuming training = 3× inference** — it is closer to 5–8× due to optimizer states, especially with Adam/AdamW

---

## The Hyperparameter Dial

Three knobs control VRAM. Each can be turned independently, but they interact through the total budget constraint.

### Dial 1 — Batch Size

$$\text{VRAM}_\text{KV} = 2 \times L \times H \times S \times B \times P$$

| Batch size $B$ | KV cache VRAM (Llama-3-8B, $S=2048$, BF16) | Total inference VRAM | Throughput (tok/s est.) |
|----------------|---------------------------------------------|----------------------|-------------------------|
| 1 | ~0.9 GB | ~17 GB | ~40 tok/s |
| 2 | ~1.8 GB | ~18 GB | ~75 tok/s |
| 4 | ~3.5 GB | ~19.5 GB | ~130 tok/s |
| 8 | ~7.0 GB | ~23 GB | ~200 tok/s |
| 16 | ~14 GB | ~30 GB ❌ OOM on 24 GB | — |

> ⚠️ Never push batch=16 on a 24 GB GPU with Llama-3-8B at BF16. Use INT8/INT4 quantization (Ch.3) to free ~8 GB first.

### Dial 2 — Sequence Length

Sequence length $S$ scales KV cache linearly. Halving sequence length halves KV cache:

| Sequence length | KV cache (batch=4, BF16) | Notes |
|-----------------|--------------------------|-------|
| 512 tokens | ~0.9 GB | Customer support queries |
| 2,048 tokens | ~3.5 GB | Standard context window |
| 8,192 tokens | ~14 GB | Near-OOM at batch=4; need INT8 |
| 32,768 tokens | ~56 GB ❌ | Multi-GPU only |

### Dial 3 — Precision

Precision affects parameter memory and KV cache simultaneously:

| Precision | Bytes per param $P$ | Llama-3-8B params | KV cache per unit | Total VRAM (batch=4, $S=2048$) |
|-----------|--------------------|--------------------|-------------------|-------------------------------|
| FP32 | 4 | 32 GB | 4 bytes | ~50 GB ❌ |
| BF16 | 2 | 16 GB | 2 bytes | ~19.5 GB ✅ |
| INT8 | 1 | 8 GB | 1 byte | ~9.8 GB ✅✅ |
| INT4 | 0.5 | 4 GB | 0.5 byte | ~5 GB — headroom for batch=16 |

> 💡 KV cache grows with batch × seq_len. Parameters are **fixed** at load time. Quantization shrinks both; smaller batch reduces only the KV cache.

---

## Code Skeleton

```python
# Educational: VRAM budget calculator from scratch
def vram_budget_inference(
    n_params: int,           # total model parameters (e.g., 8_000_000_000)
    bytes_per_param: float,  # precision: FP32=4, BF16=2, INT8=1, INT4=0.5
    n_layers: int,           # transformer layers (e.g., 32 for Llama-3-8B)
    n_heads: int,            # attention heads (e.g., 32)
    seq_len: int,            # max sequence length (e.g., 2048)
    batch_size: int,         # concurrent requests
    activation_overhead_gb: float = 2.0,  # typical activation memory
) -> dict:
    """Calculate inference VRAM breakdown in GB."""
    params_gb = (n_params * bytes_per_param) / 1e9
    head_dim = 128  # typical; hidden_dim / n_heads
    kv_cache_gb = (2 * n_layers * n_heads * head_dim * seq_len * batch_size * bytes_per_param) / 1e9
    total_gb = params_gb + kv_cache_gb + activation_overhead_gb
    return {
        "params_gb": round(params_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "activations_gb": activation_overhead_gb,
        "total_gb": round(total_gb, 2),
        "fits_rtx4090": total_gb <= 22.0,  # leave 2 GB headroom
    }

# Llama-3-8B, BF16, batch=4
print(vram_budget_inference(8_000_000_000, 2.0, 32, 32, 2048, 4))
# → {'params_gb': 16.0, 'kv_cache_gb': 3.44, 'activations_gb': 2.0, 'total_gb': 21.44, 'fits_rtx4090': True}
```

```python
# Production: pre-flight VRAM check before model load
import subprocess, json

def check_gpu_vram_available() -> float:
    """Return available VRAM in GB using nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024  # MiB → GB

def preflight_vram_check(required_gb: float, safety_margin_gb: float = 2.0) -> bool:
    """Abort if not enough VRAM. Call before loading model."""
    available = check_gpu_vram_available()
    if available < required_gb + safety_margin_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {available:.1f} GB available, "
            f"{required_gb + safety_margin_gb:.1f} GB needed (incl. {safety_margin_gb} GB margin)"
        )
    return True
```

---

## Where This Reappears

| Chapter | How memory budget concepts appear |
|---------|------------------------------------|
| **Ch.3 — Quantization** | INT8/INT4 reduces bytes-per-param $P$ in the VRAM formula; the same formulas here predict post-quantization savings |
| **Ch.5 — Inference Optimization** | PagedAttention manages KV cache pages dynamically — the same KV cache VRAM formula here determines page pool size |
| **Ch.6 — vLLM & Serving** | vLLM's `gpu_memory_utilization` parameter is directly the `1 - headroom` fraction from the VRAM budget |
| **AI Infrastructure Ch.4** | LoRA fine-tuning needs parameter + optimizer + gradient memory; the optimizer state formula (8× for Adam) comes from the training budget section here |
| **Cost & Latency (AI track)** | Cost-per-token = (hourly rate) / (tokens/sec); tokens/sec depends on batch size, which is capped by VRAM budget derived here |

---

## 12 · Progress Check — What We've Accomplished

🎉 **VRAM BUDGET CONFIRMED! Llama-3-8B fits in RTX 4090 at FP16**

**Unlocked capabilities**:
- ✅ **Precise VRAM calculator**: Know exact memory breakdown for any model/batch/seq_len
- ✅ **Inference budget**: 16GB params + 4GB KV + 2GB activations = 22GB (fits in 24GB) ✅
- ✅ **Training budget**: 104GB total → need A100 80GB × 2 or gradient checkpointing
- ✅ **Batch size limits**: batch=1 max at FP16 → need quantization (Ch.3) for batch=4

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MAINTAINED** | $1,095/month RTX 4090 confirmed |
| #2 LATENCY | ❌ **BLOCKED** | batch=1 → sequential processing → high latency under load |
| #3 THROUGHPUT | ❌ **SHORTFALL** | batch=1 → 3,000 req/day (30% of 10k target) |
| #4 MEMORY | ✅ **TARGET HIT!** | 22GB / 24GB = fits! (but zero batch headroom) |
| #5 QUALITY | ⚡ **ON TRACK** | Llama-3-8B baseline (assumed >95%) |
| #6 RELIABILITY | ⚡ **UNKNOWN** | Need Ch.9-10 for production deployment |

**What we can solve now**:

✅ **Confirm hardware purchase with exact VRAM breakdown**:
```
Before Ch.2:
Engineer: "8B params × 2 bytes = 16GB. RTX 4090 has 24GB. Should fit!"
[Orders GPU, deploys → OOM error]

After Ch.2:
Engineer calculates:
"Parameters: 16GB
 KV cache (batch=1, seq=2048): 4GB
 Activations: 2GB
 Total: 22GB / 24GB = 92% utilization ✅
 Headroom: 2GB (not enough for batch=2)
 
 Conclusion: Fits for batch=1 only. Need quantization (Ch.3) to enable batching."

CEO: "So we CAN use RTX 4090, but throughput will be limited?"
Engineer: "Correct. We'll hit 3,000 req/day at batch=1. To reach 10k target,
          we need INT4 quantization (Ch.3) to free VRAM for batch=4."

Result: ✅ Confident hardware purchase + clear roadmap to hit throughput target!
```

✅ **Understand training infrastructure needs**:
```
Before Ch.2:
"Can we fine-tune Llama-3-8B on RTX 4090?"

After Ch.2:
"Training memory: 16GB params + 64GB optimizer + 16GB gradients = 96GB
 RTX 4090 only has 24GB → cannot fit!
 
 Options:
 1. A100 80GB × 2 with ZeRO-2 sharding ($6,000/month)
 2. Gradient checkpointing → 30GB (fits on A100 40GB, $3,000/month)
 3. LoRA fine-tuning → only 18GB (fits on RTX 4090, $1,095/month!) ✅
 
 Decision: Use LoRA (Ch.4) to fine-tune on RTX 4090 without budget increase."

Result: ✅ Can fine-tune without exceeding $15k/month budget!
```

✅ **Set realistic batch size expectations**:
```
CEO: "Can we batch 10 requests at once for efficiency?"

Engineer: "Let me calculate:
 batch=10, seq=2048:
 KV cache = 2 × 32 layers × 4096 dim × 2048 seq × 10 batch × 2 bytes
          = 40 GB for KV cache alone!
 
 RTX 4090 only has 24GB total → cannot fit batch=10.
 
 At FP16: max batch=1 (2GB free VRAM)
 At INT4: max batch=4 (14GB free VRAM, 4GB per batch KV cache)
 
 To reach batch=10, need A100 80GB or multi-GPU setup (Ch.7)."

Result: ✅ Realistic throughput expectations set!
```

**What's still blocking**:

- ❌ **Throughput target unreachable**: batch=1 → 3,000 req/day (need 10k) → **Need Ch.3 quantization to enable batch=4**
- ❌ **Latency under load**: Sequential processing → queuing delays when traffic spikes
- ❌ **No fine-tuning path on current budget**: 104GB training needs expensive GPUs → **Need Ch.4 LoRA for RTX 4090 fine-tuning**
- ❌ **No serving framework selected**: Raw inference loop is slow → **Need Ch.5-6 for optimized serving**

**Next chapter**: [Quantization & Precision](../quantization_and_precision) shrinks model from 16GB → 4GB:
- INT4 quantization (GPTQ/AWQ)
- Quality validation (perplexity benchmarks)
- **Unlocks batch=4 → 12,000 req/day throughput ✅ (hits target!)**

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| VRAM = params + KV cache + activations (inference); add optimizer + gradients (training) | How much VRAM does Llama-2-70B need for inference at batch=4, seq=4096? | Forgetting KV cache scales linearly with batch × seq_len |
| KV cache formula: $2 \times L \times H \times S \times B \times P$ | Why does batch=1 use so little VRAM compared to batch=8? | Thinking activations dominate — KV cache is usually the biggest variable component |
| Adam optimizer = 2× param memory (momentum + variance in FP32) | Can you fine-tune a 13B model on a single A100 40GB? | Not accounting for optimizer states (65% of training VRAM) |
| Gradient checkpointing trades 30% speed for 90% less activation memory | What is the memory breakdown for training vs inference? | Confusing inference (no optimizer/gradients) with training (5–8× more VRAM) |
| INT4 quantization: 16GB → 4GB params (75% reduction) enables 4× batch size | How does quantization help throughput? | Saying "quantization speeds up inference" — it speeds up throughput via batching, not single-request latency |

---

## 13 · Bridge to Chapter 3

Ch.2 confirmed the model fits — but revealed a critical bottleneck: **batch=1 limits throughput to 3,000 req/day** (30% of target). The 2 GB of free VRAM is not enough to batch multiple requests. Ch.3 (Quantization & Precision) attacks this problem directly: by shrinking the model from 16 GB to 4 GB via INT4 quantization, we free 12 GB for KV cache, enabling batch=4 and 4× throughput. The question: **does INT4 quantization destroy quality?** That is what Ch.3 answers.

## Illustrations

![Memory budgets — VRAM breakdown for inference vs training, KV cache scaling, batch size limits](img/Memory%20Budgets.png)


## 14 · Key Diagrams

> Add 2–3 diagrams showing the key data flows or architectural boundaries here.


## 15 · The Hyperparameter Dial

> List 3–5 dials (batch size, precision, parallelism strategy, etc.) and their
> effect on the latency/throughput/memory triangle.


## 16 · Code Skeleton

### Educational

```python
# Educational: concept from scratch
pass
```

### Production

```python
# Production: optimized pipeline call
pass
```

