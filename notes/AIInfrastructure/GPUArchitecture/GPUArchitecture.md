# Ch.1 — GPU Architecture Fundamentals

> **Running scenario:** InferenceBase needs to self-host Llama-3-8B to cut a $80k/month OpenAI bill. Before ordering a single machine, the Platform Engineer has to answer: *which GPU, and why?* That question is impossible to answer correctly without understanding what a GPU actually does — and why its specs translate to AI workloads the way they do.

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
