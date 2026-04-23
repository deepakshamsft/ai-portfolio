# AI Infrastructure · Interview Guide

This guide consolidates interview preparation material from the AI Infrastructure track, covering GPU architecture, memory budgets, quantization, parallelism, inference optimization, and production serving.

---

## GPU Architecture

### Must Know

**Tensor Cores do matrix multiply; CUDA cores do scalar arithmetic**
- Tensor Cores: specialized for INT8/FP16/BF16 matrix operations — what LLMs actually do
- CUDA cores: general-purpose FP32 arithmetic — rarely the bottleneck for inference

**LLM decode is memory-bound, not compute-bound, at batch size 1**
- Each token generation reads all model weights from memory
- GPU spends most time waiting for data, not computing
- This is why HBM bandwidth matters more than TFLOP/s for inference

**HBM bandwidth (TB/s) is the key inference spec; VRAM capacity (GB) is the key sizing spec**
- Bandwidth: how fast you can stream weights to compute units (inference throughput)
- Capacity: can the model fit at all? (7B BF16 = 14GB, won't fit on 8GB card)

**Warp = 32 threads executing the same instruction; warp divergence kills utilisation**
- All threads in a warp execute in lockstep
- If threads take different branches (if/else), they execute serially → wasted cycles

### Likely Asked

**What is arithmetic intensity and why does it matter for LLMs?**
- Arithmetic intensity = FLOPs performed per byte read from memory
- LLM decode at batch=1: ~1–5 FLOP/byte (very memory-bound)
- GPU ridge point (A100): ~156 FLOP/byte (where compute and memory balance)
- Below ridge point: adding more compute doesn't help — bandwidth is the limit

**Explain the Roofline Model and where LLM inference sits on it**
- Roofline plots achievable FLOP/s vs arithmetic intensity
- Left of ridge: memory-bound (bandwidth-limited) — where LLM inference lives
- Right of ridge: compute-bound (TFLOP/s-limited) — where training often lives
- Key insight: a GPU with 3× the compute but same bandwidth is no faster for decode

**Why is a matrix multiply with batch=1 slow despite large TFLOP/s numbers?**
- Batch=1 means the matrix is tall and skinny
- GPU can't keep all compute units fed — many sit idle waiting for memory
- Batching increases arithmetic intensity → moves you right on the roofline

**How does batching improve GPU utilisation for LLM inference?**
- Batch=32 means 32 different prompts processed in parallel
- GPU reads weights once but computes 32 outputs → amortizes memory cost
- Arithmetic intensity goes from ~2 FLOP/byte to ~50+ FLOP/byte
- Can saturate compute units, not just bandwidth

**What is the difference between V100, A100, and H100 for LLM workloads?**
| GPU | HBM bandwidth | BF16 TFLOP/s | VRAM | Best for |
|-----|---------------|--------------|------|----------|
| V100 | 900 GB/s | 125 | 16/32 GB | Legacy training |
| A100 | 2 TB/s | 312 | 40/80 GB | Training + inference |
| H100 | 3.35 TB/s | 989 | 80 GB | Large model training |

**For inference:** bandwidth matters most → H100 is 1.7× faster than A100 for decode

### Trap to Avoid

❌ **"More CUDA cores = faster"** — No, bandwidth is almost always the real constraint for inference

❌ **Confusing peak TFLOP/s (marketing) with achievable TFLOP/s** — actual throughput after memory stalls is often 50–70% of peak

❌ **Forgetting the ridge point** — a GPU with 3× the compute but same bandwidth is no faster for memory-bound workloads

❌ **Quoting FP32 TFLOP/s when comparing cards** — always compare BF16 tensor throughput (what LLMs actually use)

❌ **Assuming NVLink is the default** — consumer GPUs (RTX 4090) only have PCIe; data center cards (A100, H100) have NVLink

---

## Memory & Compute Budgets

### Must Know

**7B model in BF16 = 14 GB weights** (2 bytes per param)
- Add KV cache, activations, optimizer states → can easily 3× that
- This is why a 7B model doesn't fit on a 16GB GPU for training

**KV cache grows with sequence length and batch size**
- Per token generated: KV cache adds 2 × layers × hidden_dim × 2 bytes
- LLaMA-2-7B: 2 × 32 × 4096 × 2 = 512 KB per token
- 2048 token sequence → 1 GB KV cache per request

**Quantization saves VRAM: INT4 ~8× smaller than FP32**
- FP32: 4 bytes per param → 7B = 28 GB
- BF16: 2 bytes per param → 7B = 14 GB
- INT8: 1 byte per param → 7B = 7 GB
- INT4: 0.5 bytes per param → 7B = 3.5 GB

### Likely Asked

**How much VRAM does a 70B model need for inference?**
- BF16 weights: 70B × 2 = 140 GB
- KV cache (2048 ctx, batch=1): ~2 GB
- Activations (negligible for inference): ~1 GB
- **Total: ~143 GB** → needs 2×A100-80GB or 1×H100-80GB + INT8 quantization

**Why does training need more memory than inference?**
- Inference: weights + KV cache + activations
- Training: weights + optimizer states (2× weights for Adam) + gradients (1× weights) + activations (batch-size dependent)
- **Total:** ~4× the inference memory requirement

**What is gradient checkpointing?**
- Trade-off: recompute activations during backward pass instead of storing them
- Saves memory (don't store all activations) at the cost of ~30% slower training
- Essential for fitting large models in limited VRAM

### Trap to Avoid

❌ **Forgetting KV cache growth** with sequence length and batch size — kills production deployments

❌ **"Quantization is free"** — perplexity degrades, especially for reasoning tasks; INT4 can lose 5–10% accuracy

❌ **"More GPUs = linear speedup"** — communication overhead breaks this; 4×A100 ≠ 4× faster

---

## Production AI Infrastructure

### The 10 rules for production AI infrastructure

1. **TTFT (Time to First Token) and P99 latency are the user-visible metrics** — throughput is the cost metric
2. **$/GPU-hour is not the right metric** — $/token at your SLA is
3. **On-demand GPU instances for training = leaving 60–80% of budget on the table** — use spot/reserved
4. **Checkpoint early and often** — fault-tolerant training is not optional on spot instances
5. **KV cache avoids recomputing past tokens** — continuous batching keeps GPU busy
6. **vLLM is the dominant open-source choice** for production GPU serving (not Ollama)
7. **All-reduce is the critical collective for DDP** — NVLink >> PCIe for intra-node
8. **Optimising for throughput at the cost of P99 latency** — users notice, accountants don't
9. **"Just use Ollama in production"** — Ollama is for local dev, not high-concurrency serving
10. **"We'll add checkpointing later"** — the later never comes, you lose the run

### Key concepts

**Prefill vs. Decode bottleneck:**
- Prefill (prompt processing): compute-bound, benefits from batching
- Decode (token generation): memory-bound, limited by bandwidth
- Different optimization strategies for each phase

**Continuous batching:**
- Traditional batching: wait for slowest request in batch to finish
- Continuous batching (vLLM): add new requests as others complete → GPU never idle
- Can achieve 2–3× higher throughput

**PagedAttention (vLLM innovation):**
- KV cache stored in non-contiguous memory pages (like OS virtual memory)
- Eliminates memory fragmentation
- Enables 2× larger batch sizes → 2× throughput

**Multi-node scaling challenges:**
- Intra-node: NVLink (600 GB/s A100, 900 GB/s H100) — fast
- Inter-node: InfiniBand (200 Gb/s = 25 GB/s) — 24–36× slower than NVLink
- Communication overhead becomes dominant beyond 8–16 GPUs

### Trap to Avoid

❌ **Confusing prefill (prompt processing) with decode (generation)** — very different bottlenecks

❌ **"Just use Ollama in production"** — Ollama is for local dev, not high-concurrency serving

❌ **Optimising for throughput at the cost of P99 latency** — users notice latency spikes, accountants don't

❌ **"We'll add checkpointing later"** — you lose the training run when a spot instance dies

❌ **Forgetting inter-node bandwidth is a hard limit** on multi-node scaling (InfiniBand << NVLink)

---

## Quick Reference Table

| Topic | Key Insight | Common Mistake |
|-------|-------------|----------------|
| GPU Architecture | Bandwidth > compute for LLM inference | "More CUDA cores = faster" |
| Memory | 7B BF16 = 14 GB + KV cache + activations | Forgetting KV cache growth |
| Quantization | INT4 ~8× VRAM savings vs FP32 | "Quantization is free" (perplexity loss) |
| Parallelism | Data parallel = gradient sync; Tensor parallel = weight shard | "More GPUs = linear speedup" |
| Inference | KV cache avoids recomputing past tokens | Confusing prefill (compute-bound) with decode (memory-bound) |
| Serving | vLLM for production, Ollama for dev | "Just use Ollama in production" |
| Networking | All-reduce critical for DDP; NVLink >> PCIe | Forgetting inter-node bandwidth limits scaling |
| Cloud | $/token at SLA, not $/GPU-hour | On-demand GPU for training = 60–80% waste |
| MLOps | Checkpoint early and often on spot | "We'll add checkpointing later" |
| Production | TTFT and P99 latency = user metrics; throughput = cost metric | Optimising throughput at cost of P99 |

---

## Related Topics

- [Agentic AI Interview Guide](AgenticAI.md) — Cost & Latency section covers the application-layer view of the same GPU/inference concepts
- [AI / Fine-tuning](../AI/FineTuning/) — QLoRA is quantization + LoRA combined
- [AI / Cost & Latency](../AI/CostAndLatency/) — VRAM side of the same cost model
- [Multimodal AI / Local Diffusion Lab](../MultimodalAI/LocalDiffusionLab/) — same serving patterns apply to diffusion models

---

**End of AI Infrastructure Interview Guide**
