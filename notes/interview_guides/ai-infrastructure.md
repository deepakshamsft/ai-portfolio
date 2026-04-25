# AI Infrastructure · Interview Guide

This guide consolidates interview preparation material from the AI Infrastructure track, covering GPU architecture, memory budgets, quantization, parallelism, inference optimization, and production serving.

---

## 1 · Concept Map — The 10 Questions That Matter

| # | Cluster | What the interviewer is testing |
|---|---------|----------------------------------|
| 1 | **GPU Roofline Model** | Do you know arithmetic intensity? Can you place LLM decode on the roofline? |
| 2 | **KV Cache Sizing** | Can you compute KV cache memory per token/request? Know growth with batch × seq_len? |
| 3 | **Quantization Tradeoffs** | Know the memory savings per dtype? Understand when perplexity loss matters? |
| 4 | **TP vs DP vs ZeRO** | Can you explain tensor parallel vs data parallel and when communication dominates? |
| 5 | **Prefill vs. Decode Bottleneck** | Do you know which phase is compute-bound vs memory-bound and why? |
| 6 | **PagedAttention & Continuous Batching** | Can you explain vLLM's core innovations and the throughput gain they produce? |
| 7 | **Production Serving Stack** | Do you know vLLM vs Ollama vs TGI and when to use each? |
| 8 | **Cloud vs. Self-Host Economics** | Know $/token at SLA vs $/GPU-hour? Spot vs. reserved tradeoffs? |
| 9 | **MLOps & Checkpointing** | Can you explain gradient checkpointing (memory) vs checkpoint saving (fault tolerance)? |
| 10 | **End-to-End TTFT + P99** | Do you distinguish TTFT from throughput? Know which matters for users vs. cost? |

---

## 2 · Section-by-Section Deep Dives

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

- [Agentic AI Interview Guide](agentic-ai.md) — Cost & Latency section covers the application-layer view of the same GPU/inference concepts
- [AI / Fine-tuning](../ai/fine_tuning) — QLoRA is quantization + LoRA combined
- [AI / Cost & Latency](../ai/cost_and_latency) — VRAM side of the same cost model
- [Multimodal AI / Local Diffusion Lab](../multimodal_ai/local_diffusion_lab) — same serving patterns apply to diffusion models

---

## 3 · The Rapid-Fire Round

> 20 Q&A pairs. Each answer: ≤ 3 sentences.

**1. What is arithmetic intensity?**
FLOPs performed per byte read from memory. LLM decode at batch=1 has ~1–5 FLOP/byte — far left of the roofline, memory-bound. Adding compute does not help; bandwidth is the limit.

**2. Where does LLM decode sit on the roofline?**
Left of the ridge point — memory-bound. The GPU spends most time waiting for weights to arrive from HBM, not computing. This is why HBM bandwidth is the key inference spec.

**3. How much VRAM does a 7B BF16 model need?**
~14 GB for weights (2 bytes × 7B params), plus KV cache and activations. Total for serving at moderate sequence length: ~18–20 GB.

**4. How does batching improve GPU utilisation?**
Batch=32 reads weights once and produces 32 outputs — arithmetic intensity rises from ~2 to ~50+ FLOP/byte. This moves the workload toward the ridge point, better utilizing compute units.

**5. What is the KV cache size per token for LLaMA-2-7B?**
2 × 32 layers × 4096 hidden_dim × 2 bytes = 512 KB per token. A 2048-token sequence uses ~1 GB KV cache per request.

**6. What does INT4 quantization buy you?**
~8× VRAM reduction vs FP32 (0.5 bytes vs 4 bytes per param). A 7B model fits in ~3.5 GB. Tradeoff: 5–10% perplexity loss, especially for reasoning tasks.

**7. Prefill vs. decode — what's the bottleneck for each?**
Prefill (processing the prompt) is compute-bound — benefits from large batches. Decode (generating each token) is memory-bound — bottlenecked by HBM bandwidth.

**8. What is PagedAttention?**
KV cache stored in non-contiguous memory pages, like OS virtual memory. Eliminates KV cache fragmentation. Enables 2× larger batch sizes → 2× throughput — the core vLLM innovation.

**9. What is continuous batching?**
Adding new requests to a batch as others complete, rather than waiting for the whole batch. Keeps GPU utilization high. Achieves 2–3× throughput improvement vs static batching.

**10. vLLM vs. Ollama — when to use each?**
vLLM for high-concurrency production serving (continuous batching, PagedAttention, tensor parallelism). Ollama for local development and testing. Ollama has no production concurrency primitives.

**11. What is gradient checkpointing?**
Recomputing activations during the backward pass instead of storing them. Saves ~30–40% of activation memory at the cost of ~30% slower training. Essential for large models in limited VRAM.

**12. Why does training need ~4× more memory than inference?**
Training stores weights + optimizer states (2× weights for Adam) + gradients (1× weights) + activations. Inference stores only weights + KV cache + activations.

**13. TP vs. DP — what is the key difference?**
Tensor Parallelism (TP) shards the model weights across GPUs — each GPU holds part of each layer, and all-reduce happens per layer forward/backward. Data Parallelism (DP) replicates the full model on each GPU, syncing gradients after the backward pass.

**14. Why does communication dominate at large GPU counts in DP?**
All-reduce cost grows with batch size and model size. Beyond 8–16 GPUs, inter-node InfiniBand bandwidth (25 GB/s) vs intra-node NVLink (600+ GB/s) creates a communication wall.

**15. What is ZeRO and which stage should you use?**
ZeRO (Zero Redundancy Optimizer) partitions optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across GPUs. Use Stage 2 for most training; Stage 3 for models that don't fit on a single GPU.

**16. $/GPU-hour vs. $/token — which metric matters?**
$/token at your SLA latency is the right metric. A cheap GPU that misses P99 latency targets has zero value in production. $/GPU-hour is a procurement metric, not a serving metric.

**17. When should you use spot instances?**
For training, where jobs are long and checkpointing is cheap. Never for production inference (users need availability). Spot can save 60–80% vs on-demand.

**18. What is TTFT and why does it matter?**
Time to First Token — latency from request submission to the first generated token. This is the user-perceived "responsiveness" metric. Throughput (tokens/sec overall) is the cost metric; TTFT is the UX metric.

**19. How does speculative decoding work?**
A smaller draft model proposes K tokens cheaply; the large target model verifies them in one forward pass. If all K are accepted, you get K tokens for the cost of 1 large-model step. Works best when the draft model is accurate (same family, quantized version).

**20. What is the difference between BF16 and FP16?**
Both use 16 bits total but different mantissa/exponent splits. BF16 has the same exponent range as FP32 (8 bits) with a shorter mantissa — much more stable during training. FP16 has a smaller exponent range and is more prone to loss spikes (overflow/underflow). Modern GPUs and TPUs prefer BF16 for training.

---

## 4 · Signal Words That Distinguish Answers

**✅ Say this:**
- \"arithmetic intensity\" (not \"FLOP/s\")
- \"memory bandwidth bound\" (for decode)
- \"KV cache pressure\" (memory growth with sequence length)
- \"quantization calibration set\" (INT4 requires calibration data)
- \"PagedAttention\" (not just \"vLLM\")
- \"continuous batching\" (not just \"batching\")
- \"TTFT\" (time to first token, the user-visible metric)
- \"P99 latency\" (not just \"average latency\")
- \"ridge point\" (roofline model crossover)
- \"ZeRO Stage 2 vs 3\" (optimizer state partitioning)

**❌ Don't say this:**
- \"just use a bigger GPU\" (ignores bandwidth vs. compute tradeoff)
- \"it's the same as regular ML\" (LLM inference has unique memory-bound characteristics)
- \"quantization is free\" (perplexity loss is real, especially for reasoning)
- \"more GPUs = linear speedup\" (communication overhead breaks this)
- \"use Ollama in production\" (no concurrency primitives for serving at scale)

