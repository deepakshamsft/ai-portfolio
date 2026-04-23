# Ch.1 — GPU Architecture Fundamentals

> **The story.** GPUs were invented to draw triangles. NVIDIA's **GeForce 256** (October **1999**) coined the term *GPU* for a fixed-function 3-D graphics pipeline aimed at *Quake* players, not scientists. The pivot to general-purpose compute began when **Ian Buck** (then a Stanford PhD student, later NVIDIA's VP of Accelerated Computing) shipped **CUDA** in **2006**, exposing the GPU's parallel execution model as a C-like programming environment. For six years CUDA was a niche HPC tool. Then in **September 2012** **Alex Krizhevsky** trained **AlexNet** on two consumer **GTX 580** cards in his bedroom and won ImageNet by 10 percentage points — the moment that made GPUs synonymous with deep learning. **cuDNN** (2014) gave Caffe and TensorFlow a fast convolution library. **Pascal P100** (2016) introduced HBM2; **Volta V100** (2017) introduced **Tensor Cores** for mixed-precision matmul; **A100** (2020) added bf16 and Multi-Instance GPU; **H100** (2022) added FP8 and the Transformer Engine; **B100/B200 Blackwell** (2024) crossed into native FP4 and 192 GB HBM3e. Every layer of the modern AI stack \u2014 PyTorch, vLLM, TensorRT-LLM, Triton, FlashAttention \u2014 is built on this 25-year hardware lineage and the CUDA programming model that made it accessible.\n>\n> **Where you are in the curriculum.** This is the foundation chapter for the AI Infrastructure track. **Running scenario:** InferenceBase needs to self-host Llama-3-8B to cut an $80k/month OpenAI bill. Before ordering a single machine, the Platform Engineer has to answer: *which GPU, and why?* That question is impossible to answer correctly without understanding what a GPU actually does \u2014 and why its specs translate to AI workloads the way they do.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ❌ **Starting from zero** — no infrastructure, no hardware selected
- 📊 **Current state**: Spending $80,000/month on OpenAI API (GPT-3.5-turbo)
- 🎯 **CEO's question**: "Can we cut this bill by 80% with self-hosting?"

**What's blocking us**:

🚨 **Zero visibility into GPU requirements — cannot order hardware without understanding specs**

**Current situation**: Platform Engineer's whiteboard session with CEO

```
CEO: "I just got the OpenAI bill. $80,000 for one month. That's $960k/year. 
     We're burning investor cash on API calls. Can we self-host Llama-3-8B
     and cut this to $15k/month?"

Engineer: "Maybe. Llama-3-8B is 8 billion parameters, needs BF16 precision..."

CEO: "What does that mean in English?"

Engineer: "I... need to calculate the GPU requirements first."

CEO: "Fine. You have two weeks to tell me:
     1. Which GPU do we need?
     2. How much will it cost per month?
     3. Will it match OpenAI's speed?
     If the answer is yes, we build it. If no, we keep paying OpenAI."
```

**Problems**:
1. ❌ **No GPU expertise**: Team has only used OpenAI API — zero experience with GPU hardware
2. ❌ **Spec sheet confusion**: A100 says "312 TFLOP/s", RTX 4090 says "165 TFLOP/s", but what matters for LLMs?
3. ❌ **Unknown VRAM needs**: 8B parameters... is that 8GB? 16GB? 32GB? No one knows.
4. ❌ **Bandwidth vs compute**: Specs quote TFLOP/s, but is that the bottleneck for inference?
5. ❌ **Cost modeling impossible**: Can't estimate cloud GPU costs without knowing which GPU to use
6. ❌ **Wrong purchase = $50k mistake**: Ordering the wrong hardware means burning weeks + budget

**Business impact**:
- **$80k/month burn rate**: Every month of delay = $80,000 wasted on OpenAI
- **Zero ROI visibility**: Cannot calculate payback period without hardware costs
- **Board meeting in 2 weeks**: Need to present "build vs buy" recommendation with numbers
- **Investor pressure**: Seed round running low — need to show path to profitability
- CEO: "If we can't figure out the GPU question in 2 weeks, we're stuck with OpenAI forever. Make it happen."

**What this chapter unlocks**:

🚀 **GPU hardware fundamentals to make informed decisions**:
1. **Understand GPU architecture**: CUDA cores vs Tensor Cores → what accelerates LLMs
2. **Read GPU datasheets correctly**: VRAM, bandwidth, TFLOP/s → which specs matter
3. **Roofline analysis**: Memory-bound vs compute-bound → why LLM inference is bandwidth-limited
4. **Compare GPU options**: A100 vs H100 vs RTX 4090 → cost/performance for Llama-3-8B
5. **Identify target GPU**: RTX 4090 (24GB, 1.0 TB/s, $1.50/hr) → best price/performance for 8B model

⚡ **Expected outcomes**:
- **GPU selection**: RTX 4090 identified as target (24GB VRAM, $1.50/hr cloud rental)
- **Cost estimate**: $1.50/hr × 730 hr/mo = **$1,095/month** (vs $15k budget — excellent!) ✅
- **Bottleneck identified**: LLM inference is **memory-bound**, not compute-bound (1-5 FLOP/byte, ridge=164)
- **Next question unlocked**: "Will 24GB VRAM fit Llama-3-8B?" → Need Ch.2 memory calculator

**Constraint status after Ch.1**:
- #1 (Cost): ⚡ **ON TRACK** ($1,095/month estimated, well under $15k budget)
- #2 (Latency): ⚡ **UNKNOWN** (need Ch.5-6 for latency benchmarks)
- #3 (Throughput): ⚡ **UNKNOWN** (need Ch.5-6 for throughput tests)
- #4 (Memory): ⚡ **IDENTIFIED** (24GB VRAM target, but need Ch.2 to confirm model fits)
- #5 (Quality): ⚡ **ASSUMED** (Llama-3-8B benchmarks suggest >95%, need validation)
- #6 (Reliability): ⚡ **UNKNOWN** (need Ch.9-10 for production readiness)

**Foundation established**: Understand GPU hardware specs → can now calculate exact VRAM needs (Ch.2)

---

## 1 · Core Idea

A GPU is not a fast CPU. It is a **massively parallel matrix-multiply machine** designed for one job: apply the same arithmetic operation to thousands of numbers simultaneously. A modern high-end CPU has 16–64 cores that each run different code at high clock speeds. An H100 has **16,896 CUDA cores** and **528 Tensor Cores** all running the same operation in lock-step on enormous blocks of data. That distinction — parallel uniformity over sequential flexibility — is the reason GPUs transformed AI.

Every layer of every neural network reduces to matrix multiplications. The transformer attention mechanism is matrix multiplications. Convolutions are matrix multiplications in disguise. If you start there, the rest of GPU architecture follows naturally.

---

## 2 · The InferenceBase Angle

InferenceBase's Llama-3-8B model has 8 billion parameters. Running one inference forward pass requires roughly **16 TFLOP** of computation (in FP16). At 50 tokens per second — the minimum useful generation speed — the GPU must sustain approximately **800 GFLOP/s** of actual throughput. A consumer RTX 4090 delivers 165 TFLOP/s (BF16 tensor). That sounds like plenty. But in practice, inference is almost never compute-bound: the GPU spends most of its time *waiting for data to arrive from memory*, not doing arithmetic. Understanding why requires understanding the GPU memory hierarchy and the concept of **arithmetic intensity**.

---

## 3 · The GPU Hardware Stack

### Streaming Multiprocessors (SMs)

The SM is the fundamental building block of a GPU. Each SM contains:

```
One Streaming Multiprocessor (SM)
├── CUDA Cores     — scalar floating-point arithmetic units (FP32, FP64)
├── Tensor Cores   — 4×4 matrix multiply-accumulate units (FP16, BF16, INT8, FP8)
├── L1 Cache       — ~128 KB, private to the SM
├── Shared Memory  — programmer-controlled scratchpad, part of the L1 budget
├── Registers      — 64K 32-bit registers; private to each thread
└── Warp Schedulers — schedule 32-thread warps onto execution units

A "warp" = 32 threads that execute the same instruction in lock-step (SIMT).
A "thread block" = up to 1,024 threads; scheduled onto one SM.
A "grid" = many thread blocks; distributed across all SMs on the GPU.
```

The key consequence: **all 32 threads in a warp must execute the same instruction**. A divergent `if/else` in a warp causes the losing branch to idle — this is called warp divergence, and it kills utilisation. Neural network code almost never branches, which is part of why it maps so naturally to this model.

### CUDA Cores vs Tensor Cores

| Unit | What it does | Speed |
|------|-------------|-------|
| **CUDA Core** | One FP32 multiply-add per clock per core. Flexible — handles any scalar arithmetic. | 1 op / clock |
| **Tensor Core** | One 4×4 × 4×4 matrix multiply-accumulate per clock. Fixed operation, no flexibility. | 256 FP16 ops / clock |

A Tensor Core is ~64× more efficient than a CUDA core for matrix multiplies — but it can only do matrix multiplies. This is a deliberate architectural trade: sacrifice generality for throughput on the one operation that dominates AI workloads.

**Tensor Core generation table:**

| GPU | Architecture | Tensor Core precision | Peak TFLOP/s (BF16) |
|-----|-------------|----------------------|---------------------|
| V100 | Volta | FP16 | 125 |
| A100 | Ampere | BF16, INT8, TF32 | 312 |
| H100 | Hopper | FP8, BF16, INT8 | 989 (with sparsity: 1,979) |
| RTX 4090 | Ada Lovelace | BF16, INT8 | 165 |
| RTX 3090 | Ampere | BF16, INT8 | 71 |

> **BF16 is the number to compare** for LLM inference. FP32 TFLOP/s is marketed prominently but rarely used for inference.

---

## 4 · The Memory Hierarchy

This is the most important section in this chapter. Almost all inference performance problems are memory problems.

```
Hierarchy (fastest → slowest, smallest → largest)

┌────────────────────────────────────────────────────────────────────────┐
│  Registers     │  ~256 KB per SM  │  ~TB/s   │  Compiler-managed       │
├────────────────────────────────────────────────────────────────────────┤
│  L1 / Shared   │  ~128 KB per SM  │  ~30 TB/s │  Programmer-managed    │
├────────────────────────────────────────────────────────────────────────┤
│  L2 Cache      │  40–50 MB        │  ~12 TB/s │  Hardware-managed      │
├────────────────────────────────────────────────────────────────────────┤
│  HBM (VRAM)    │  24–80 GB        │  1–4 TB/s │  Device memory (slow!) │
├────────────────────────────────────────────────────────────────────────┤
│  System RAM    │  64–2,048 GB     │  ~50 GB/s │  PCIe transfer (slow!) │
├────────────────────────────────────────────────────────────────────────┤
│  NVMe SSD      │  TBs             │  ~7 GB/s  │  Disk offloading        │
└────────────────────────────────────────────────────────────────────────┘
```

**The critical number:** HBM bandwidth. An H100 has **3.35 TB/s** of HBM3 bandwidth. An A100 has **2 TB/s**. A consumer RTX 4090 has **1.008 TB/s**. This is the bottleneck for LLM inference. The compute units are almost always faster than the memory system can feed them.

### HBM (High Bandwidth Memory)

HBM is a 3D-stacked DRAM technology mounted directly adjacent to the GPU die, connected via a wide (4096-bit) bus. Standard GDDR6X (used in consumer GPUs) connects over a narrower bus and has lower bandwidth.

| GPU | VRAM | VRAM type | Bandwidth |
|-----|------|-----------|-----------|
| H100 SXM | 80 GB | HBM3 | 3.35 TB/s |
| A100 80GB | 80 GB | HBM2e | 2.0 TB/s |
| A100 40GB | 40 GB | HBM2e | 1.6 TB/s |
| A10G | 24 GB | GDDR6 | 0.6 TB/s |
| RTX 4090 | 24 GB | GDDR6X | 1.008 TB/s |
| RTX 3090 | 24 GB | GDDR6X | 0.936 TB/s |
| RTX 3080 Ti | 12 GB | GDDR6X | 0.912 TB/s |

The RTX 4090 has surprisingly competitive bandwidth for its price, which is why it is the preferred consumer GPU for LLM inference.

---

## 5 · Arithmetic Intensity and the Roofline Model

### Arithmetic Intensity

**Arithmetic intensity** ($I$) is the ratio of floating-point operations performed to bytes of memory moved:

$$I = \frac{\text{FLOPs}}{\text{Bytes accessed from HBM}}$$

| $I$ | Interpretation |
|-----|---------------|
| Low | Memory-bound: the compute units are idle while waiting for data |
| High | Compute-bound: the memory system is idle while compute runs |

The crossover point — the **ridge point** — is where memory bandwidth and compute throughput are exactly balanced:

$$I_{\text{ridge}} = \frac{\text{Peak TFLOP/s}}{\text{Peak Bandwidth (TB/s)}}$$

For an A100:

$$I_{\text{ridge}} = \frac{312 \text{ TFLOP/s}}{2.0 \text{ TB/s}} = 156 \text{ FLOP/byte}$$

Any operation with $I < 156$ FLOP/byte is **memory-bound** on an A100. Any operation with $I > 156$ is **compute-bound**.

### The Roofline Model

The Roofline Model is a visual tool that maps every operation onto a 2D space of arithmetic intensity vs. achievable FLOP/s:

```
Achievable
TFLOP/s
  │
  │                  ┌──────────────────── Peak Compute
  │                 /  (compute-bound)
  │                /
  │               /  ← slope = bandwidth
  │              /
  │             /
  │____________/ ← ridge point
  │(memory-bound)
  └───────────────────────── Arithmetic Intensity (FLOP/byte)
```

**Where common AI operations land:**

| Operation | Typical arithmetic intensity | Bottleneck |
|-----------|------------------------------|-----------|
| Large matrix multiply (large batch) | 100–10,000 FLOP/byte | Compute-bound |
| LLM prefill (full prompt, large batch) | ~100–500 FLOP/byte | Compute-bound (batch≥32) |
| LLM decode (single token generation) | ~1–5 FLOP/byte | **Memory-bound** |
| Embedding lookup | ~0.1 FLOP/byte | **Severe memory-bound** |
| Layer norm | ~2 FLOP/byte | **Memory-bound** |
| Attention (short seq.) | ~20 FLOP/byte | Memory-bound |
| Attention (long seq., FlashAttn) | ~50–200 FLOP/byte | Depends on seq len |

**The key insight for LLM inference:** during the decode phase, you generate one token at a time, so the batch size is 1. A matrix multiply with batch=1 loads the full weight matrix from HBM (GB of data) to perform just 2 FLOPs per weight element — giving an arithmetic intensity of roughly **1–2 FLOP/byte**. This is far below the A100's ridge point of 156 FLOP/byte. **The GPU compute units are idle more than 99% of the time during token-by-token LLM generation.** This is why batching is so critical — and why PagedAttention (Ch.5) exists.

---

## 6 · How a Matrix Multiply Maps to a GPU

A matrix multiply $C = A \times B$ where $A$ is $(M \times K)$ and $B$ is $(K \times N)$ requires $2MKN$ FLOPs (M×K×N multiply-adds, each counted as 2 operations).

```
GPU execution of GEMM (General Matrix Multiply):

1. Decompose C into tiles (e.g., 128×128 output tiles)
2. Each tile is assigned to one thread block → one SM
3. Within an SM:
   a. Load a tile of A and a tile of B into shared memory
   b. Tensor Cores compute the tile product (4×4 × 4×4 accumulations)
   c. Repeat until the full K dimension is consumed
   d. Write the output tile back to HBM
4. All SMs execute in parallel — the full C matrix is computed simultaneously

The key: shared memory acts as a fast staging area that hides HBM latency.
Data is loaded from HBM once per tile, reused many times in shared memory.
```

This is why **larger matrices run more efficiently than smaller ones**: larger tiles mean more reuse of data that was loaded from HBM, which increases arithmetic intensity and moves the operation toward the compute-bound regime.

---

## 7 · Key Specs to Read on a GPU Datasheet

When comparing GPUs for an AI workload, five numbers matter:

| Spec | What it tells you | Where to look |
|------|------------------|---------------|
| **BF16 TFLOP/s** | Peak compute for mixed-precision training/inference | "Tensor Performance" table |
| **Memory bandwidth (TB/s)** | How fast data moves from VRAM to compute — the real bottleneck for inference | "Memory" table |
| **VRAM capacity (GB)** | Maximum model + KV cache + activations you can fit | "GPU Memory" |
| **NVLink bandwidth (GB/s)** | How fast this GPU can share tensors with an adjacent GPU | "NVLink" section |
| **TDP (Watts)** | Power draw — determines cooling, rack density, and electricity bill | "Thermal Design Power" |

**Specs that are frequently misleading:**

| Spec | Why it misleads |
|------|----------------|
| FP32 TFLOP/s | LLMs almost never use FP32 for inference |
| CUDA core count | Higher count ≠ better performance; bandwidth is usually the constraint |
| Raw TFLOP/s peak | Peak assumes sustained tensor core utilisation, which requires very large, dense matrices — rarely achieved in practice |

---

## 8 · GPU Generations You Will Encounter

```
Generation timeline (datacenter focus):

2017  V100 (Volta)
      ├─ First with Tensor Cores (FP16 only)
      ├─ HBM2: 900 GB/s (32 GB) / 900 GB/s (16 GB)
      └─ Still common in older cloud instances (p3 on AWS)

2020  A100 (Ampere)
      ├─ BF16 Tensor Cores — the sweet spot for LLM training
      ├─ HBM2e: 2.0 TB/s (80 GB), 1.6 TB/s (40 GB)
      ├─ MIG (Multi-Instance GPU): partition one A100 into 7 isolated instances
      └─ The workhorse of LLM training and inference (2021–2024)

2022  H100 (Hopper)
      ├─ FP8 Tensor Cores — first hardware support for FP8 inference
      ├─ HBM3: 3.35 TB/s (SXM flavor) — 1.7× over A100
      ├─ NVLink 4.0: 900 GB/s bidirectional per GPU
      └─ The current standard for large-scale training

2024  B200 (Blackwell)
      ├─ FP4 support, 20 PFLOP/s peak (FP4 sparsity)
      ├─ HBM3e: 8.0 TB/s
      └─ Targeted at trillion-parameter training

Consumer (relevant for self-hosting):
      RTX 4090  — 24 GB GDDR6X, 1.0 TB/s, 165 TFLOP/s (BF16), ~$1,600
      RTX 3090  — 24 GB GDDR6X, 0.9 TB/s,  71 TFLOP/s (BF16), ~$700 used
```

---

## 9 · Step by Step — How a Token Gets Generated

Tracing one decode step of an LLM inference forward pass at the hardware level:

```
Step 1: LOAD input embedding
        GPU fetches the embedding vector for the last generated token
        from HBM. ~4 KB for a 4096-dim vector in BF16.
        → Memory-bound, nothing to compute yet.

Step 2: ATTENTION — Q/K/V projections
        Three matrix multiplies: [1 × 4096] × [4096 × 4096]
        FLOPs = 2 × 1 × 4096 × 4096 ≈ 33M FLOPs
        Bytes loaded = 3 × 4096 × 4096 × 2 bytes ≈ 96 MB
        Arithmetic intensity ≈ 0.34 FLOP/byte → deep memory-bound

Step 3: ATTENTION — score computation
        Q × Kᵀ for each head: shape [1 × 128] × [128 × seq_len]
        At seq_len = 512: 33M FLOPs, 0.5 MB bytes read
        Intensity ≈ 66 FLOP/byte → still memory-bound (ridge=156)

Step 4: FFN — two linear layers
        [1 × 4096] × [4096 × 16384] then [1 × 16384] × [16384 × 4096]
        ≈ 268M FLOPs, 512 MB loaded
        Intensity ≈ 0.5 FLOP/byte → memory-bound

Step 5: OUTPUT head — project to vocab
        [1 × 4096] × [4096 × 32000]
        FLOPs ≈ 262M, Bytes ≈ 256 MB
        Intensity ≈ 1.0 FLOP/byte → memory-bound

TOTAL per decode step (Llama-3-8B, seq_len=512):
  ≈ 16 GFLOP, ≈ 1.2 GB from HBM
  On A100 (2 TB/s): 0.6 ms minimum (bandwidth-limited)
  Actual: ~1–2 ms with overhead → ~500–1000 tokens/sec maximum
  With batch size 1 (single user): GPU is utilised <0.1% of peak compute
```

This walkthrough explains why **batching transforms LLM inference economics** more than any other single technique.

---

## 10 · The Key Diagram

### Roofline: A100 with Common LLM Operations

```
Achievable
TFLOP/s
 312 │                           ┌─────────────────── Peak Compute (312 TFLOP/s)
     │           Compute-bound  /
     │                         /
  50 │                Large ★ /
     │              matmul    /  Prefill (large batch) ★
     │                       /
  10 │                      /        Attention (long) ★
     │                     /
   1 │      Memory-bound  /  Prefill (small batch) ★
     │                   /
 0.1 │  Decode ★        /  Embedding ★
     │   (batch=1)     /
     └─────────────────┬────────────────────────────── Arithmetic Intensity
                       │                               (FLOP/byte)
                    156 ← ridge point
      0.5    5     50  156       500       5000
```

---

## 11 · What Can Go Wrong

- **Buying for TFLOP/s when you need VRAM** — a GPU with 2× the compute but half the VRAM won't run your model at all; always size for VRAM first, then check bandwidth.
- **Ignoring memory bandwidth for inference** — LLM decode is bandwidth-limited; the A10G (24 GB, 0.6 TB/s) is significantly slower than the RTX 4090 (24 GB, 1.0 TB/s) for LLM inference despite similar VRAM.
- **Comparing FP32 TFLOP/s across vendors** — AMD, Intel, and NVIDIA quote different precisions prominently; always normalise to BF16 tensor performance.
- **Underestimating PCIe bottleneck for multi-GPU** — using multiple consumer GPUs connected only via PCIe (not NVLink) means all-reduce gradient sync is 10–20× slower than expected; tensor parallelism barely scales.
- **Conflating training and inference hardware needs** — training wants maximum compute (BF16 TFLOP/s, HBM3 bandwidth, NVLink); inference for batch=1 wants maximum bandwidth per dollar, often making used A100s or RTX 4090s better value than new H100s.

---

## 11.5 · Progress Check — What We've Accomplished

🎉 **GPU HARDWARE SELECTION COMPLETE! Target identified: RTX 4090**

**Unlocked capabilities**:
- ✅ **Read GPU datasheets**: Understand VRAM, bandwidth, TFLOP/s → know which specs drive LLM performance
- ✅ **Roofline analysis**: LLM inference is memory-bound (1-5 FLOP/byte vs 164 ridge point) → bandwidth is king
- ✅ **Cost-performance comparison**: RTX 4090 ($1.50/hr) beats A100 ($3/hr) for single-user inference
- ✅ **Hardware recommendation**: RTX 4090 (24GB VRAM, 1.0 TB/s, 165 TFLOP/s BF16) selected

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **EXCELLENT!** | $1,095/month estimated (RTX 4090 @ $1.50/hr × 730 hr/mo) — **93% under budget!** |
| #2 LATENCY | ⚡ **IDENTIFIED BOTTLENECK** | Memory bandwidth (1.0 TB/s) is bottleneck, not compute (165 TFLOP/s) — need Ch.5-6 for latency tests |
| #3 THROUGHPUT | ⚡ **PENDING** | Batch=1 gives ~500 tokens/sec max (bandwidth-limited) — need batching (Ch.5) to hit 10k req/day |
| #4 MEMORY | ⚡ **CANDIDATE IDENTIFIED** | 24GB VRAM available — but does Llama-3-8B fit? Need Ch.2 memory calculator! |
| #5 QUALITY | ⚡ **ASSUMED** | Llama-3-8B benchmark scores suggest >95% accuracy — need validation |
| #6 RELIABILITY | ⚡ **UNKNOWN** | Need Ch.9-10 for production deployment patterns |

**What we can solve now**:

✅ **Read a GPU datasheet and identify the right specs**:
```
Before Ch.1:
Engineer looks at A100 datasheet: "312 TFLOP/s sounds fast! But wait,
  RTX 4090 is only 165 TFLOP/s... does that mean A100 is 2× better?"

After Ch.1:
Engineer: "LLM decode lands at 1-5 FLOP/byte (memory-bound region).
  A100: 2.0 TB/s bandwidth → ridge point = 156 FLOP/byte
  RTX 4090: 1.0 TB/s bandwidth → ridge point = 164 FLOP/byte
  
  For memory-bound workloads, performance ∝ bandwidth, not TFLOP/s.
  A100 has 2× bandwidth but costs 10× more ($80k vs $8k retail).
  For single-user inference (batch=1), RTX 4090 delivers 50% of A100 
  throughput at 10% of the cost. Winner: RTX 4090!"

Result: ✅ Can make informed hardware decisions!
```

✅ **Understand why LLM inference is slow (and what to optimize)**:
```
Before Ch.1:
"Llama-3-8B has 8 billion parameters. An H100 has 1,979 TFLOP/s.
 Why is inference still slow?"

After Ch.1:
"Decode generates one token at a time (batch=1).
 Each token requires loading 16GB of weights from HBM.
 Arithmetic intensity = 16 GFLOP / 16 GB = 1 FLOP/byte.
 Ridge point for H100 = 989 TFLOP/s / 3.35 TB/s = 295 FLOP/byte.
 
 1 FLOP/byte << 295 ridge → deep in memory-bound region.
 H100's 1,979 TFLOP/s is irrelevant — memory bandwidth limits throughput.
 
 To improve: increase arithmetic intensity by batching multiple requests
 (load weights once, process N tokens) → Ch.5 PagedAttention"

Result: ✅ Know the bottleneck before building!
```

✅ **Estimate monthly GPU costs**:
```
RTX 4090 cloud rental pricing:
- RunPod on-demand: $1.50/hr
- Lambda Labs: $1.10/hr (but less availability)
- Vast.ai spot: $0.80/hr (unreliable, can be preempted)

Conservative estimate (RunPod on-demand):
$1.50/hr × 730 hr/month = $1,095/month

vs OpenAI baseline: $80,000/month
Savings: $78,905/month (98.6% reduction!)

vs Budget constraint: $15,000/month
Headroom: $13,905/month (can scale to 13× more GPUs if needed)

Result: ✅ Business case is solid!
```

✅ **Present to CEO with confidence**:
```
CEO: "So, which GPU do we need?"

Engineer: "RTX 4090. Here's why:
1. Llama-3-8B inference is memory-bound, not compute-bound.
2. RTX 4090 has 24GB VRAM and 1.0 TB/s bandwidth at $1.50/hr.
3. A100 has 2× bandwidth but 10× the cost — overkill for our use case.
4. Monthly cost: $1,095 vs $80,000 OpenAI → 98.6% savings.
5. Budget headroom: $13,905 → room to scale to 13 GPUs if traffic grows.

Next steps: Ch.2 calculates exact VRAM (does 24GB fit Llama-3-8B?),
            then Ch.5-6 benchmark latency/throughput to confirm <2s SLA."

CEO: "Perfect. Green light to continue. What's the timeline?"

Engineer: "2 more weeks to validate memory fits + benchmark performance.
           If tests pass, we deploy in Week 3."

Result: ✅ CEO has confidence in the plan!
```

**What's still blocking**:

- ❌ **Memory fit unknown**: 8B parameters = 16GB FP16... but what about KV cache, activations, optimizer states? → **Need Ch.2 memory calculator**
- ❌ **Latency/throughput untested**: Roofline predicts memory-bound, but actual p95 latency? → **Need Ch.5-6 benchmarks**
- ❌ **Quantization unexplored**: Can INT4 quantization cut VRAM 4× (16GB → 4GB) without killing quality? → **Need Ch.3**
- ❌ **Batching strategy missing**: How to batch 10k req/day without blowing latency SLA? → **Need Ch.5 PagedAttention**
- ❌ **Serving framework unknown**: vLLM? TensorRT-LLM? TGI? → **Need Ch.6 benchmarks**
- ❌ **Production readiness unclear**: Monitoring, checkpointing, fault tolerance? → **Need Ch.9-10**

**Next chapter**: [Memory & Compute Budgets](../MemoryAndComputeBudgets/) calculates exact VRAM breakdown:
- Parameters: 8B × 2 bytes/param (FP16) = ?
- KV cache: batch_size × seq_len × layers × hidden_dim = ?
- Activations: forward pass temp tensors = ?
- **Total: ? GB** → does it fit in 24GB RTX 4090? ✅ or ❌?

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Tensor Cores accelerate matrix multiply (256 FP16 ops/clock vs 1 FP32 op/clock CUDA core) | What is arithmetic intensity and why does it matter for LLMs? | "More CUDA cores = faster" — bandwidth is almost always the real constraint for inference |
| LLM decode is memory-bound (1-5 FLOP/byte) vs A100 ridge point (156 FLOP/byte) | Explain the Roofline Model and where LLM inference sits on it | Confusing peak TFLOP/s (marketing) with achievable TFLOP/s (actual throughput after memory stalls) |
| HBM bandwidth (TB/s) is the key inference spec; VRAM capacity (GB) is the key sizing spec | Why is a matrix multiply with batch=1 slow despite large TFLOP/s numbers? | Forgetting the ridge point — a GPU with 3× the compute but the same bandwidth is no faster for memory-bound workloads |
| A100 ridge point ≈ 156 FLOP/byte; decode ops land at ~1–5 FLOP/byte | How does batching improve GPU utilisation for LLM inference? | Quoting FP32 TFLOP/s when comparing cards — always compare BF16 tensor throughput |
| RTX 4090 has 24GB VRAM + 1.0 TB/s bandwidth → best price/performance for 8B models | What is the difference between V100, A100, and H100 for LLM workloads? | Assuming NVLink is the default — consumer GPUs only have PCIe |

---

## 12 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|-----------|-------------|---------------|
| Tensor Cores do matrix multiply; CUDA cores do scalar arithmetic | What is arithmetic intensity and why does it matter for LLMs? | "More CUDA cores = faster" — bandwidth is almost always the real constraint for inference |
| LLM decode is memory-bound, not compute-bound, at batch size 1 | Explain the Roofline Model and where LLM inference sits on it | Confusing peak TFLOP/s (marketing) with achievable TFLOP/s (actual throughput after memory stalls) |
| HBM bandwidth (TB/s) is the key inference spec; VRAM capacity (GB) is the key sizing spec | Why is a matrix multiply with batch=1 slow despite large TFLOP/s numbers? | Forgetting the ridge point — a GPU with 3× the compute but the same bandwidth is no faster for memory-bound workloads |
| A100 ridge point ≈ 156 FLOP/byte; decode ops land at ~1–5 FLOP/byte | How does batching improve GPU utilisation for LLM inference? | Quoting FP32 TFLOP/s when comparing cards — always compare BF16 tensor throughput |
| Warp = 32 threads executing the same instruction; warp divergence kills utilisation | What is the difference between V100, A100, and H100 for LLM workloads? | Assuming NVLink is the default — consumer GPUs only have PCIe |

---

## Bridge to Chapter 2

Ch.1 established the hardware: what a GPU is, where its bottlenecks lie, and how AI operations map to its compute and memory systems. Ch.2 (Memory & Compute Budgets) takes that foundation and asks the next question: *given a specific model architecture, exactly how much VRAM does it need* — at inference, at training, and as the sequence length and batch size grow? The roofline model told you whether you are memory-bound. Ch.2's VRAM calculator tells you whether the model fits at all.

## Illustrations

![GPU architecture — SM hardware stack, memory hierarchy, roofline model, tiled matmul](img/GPU%20Architecture.png)
