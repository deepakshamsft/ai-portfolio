# AI Infrastructure — How to Read This Collection

> This document is your **entry point and reading map**. It explains the conceptual arc across all chapters, defines the running scenario that threads through every note, shows how each chapter connects to the others, and prescribes reading paths based on your goal.

---

## The Central Story in One Paragraph

Every AI model that looks effortless in a demo is actually running on a complex stack of hardware, software, and operational decisions that determine whether it costs $80,000 a month in API bills or $4,000 a month on dedicated compute. **AI Infrastructure is the discipline of understanding that stack from the silicon up**: how a GPU executes a matrix multiply thousands of times per second, how that operation is kept from being memory-starved, how a single model is split across dozens of machines for training, how an inference server batches thousands of user requests into one GPU kernel, and how you build the operational layer that keeps all of it running reliably at scale. The notes in this collection build that understanding from the ground up — starting with a single GPU, ending with a full production AI platform — and they deliberately connect the hardware primitives to the software decisions engineers actually make.

> **Note on running code:** Unlike the ML or AI tracks, this track has no single canonical dataset to compute against. GPU memory, parallelism strategies, and cluster topology are phenomena you observe through profiling, simulation, and calculation — not through training loops on a fixed dataset. Every notebook in this track is a **calculator, simulator, or model estimator**: you bring your own numbers (model size, request rate, hardware budget) and the notebook tells you what to expect before you spend a dollar on cloud compute.

---

## The Running Scenario — InferenceBase

Every note in this track is anchored to a single growing problem: **InferenceBase**, a seed-stage AI startup building a document intelligence API. The product takes enterprise PDF documents, runs them through a self-hosted LLM, and returns structured JSON. The CEO has just forwarded the latest AWS bill — $80,000 in OpenAI API charges for the month — and asked the founding Platform Engineer (you) to evaluate whether self-hosting Llama-3-8B makes sense.

```
InferenceBase after Ch.1:
  Question: What GPU hardware do we even need for Llama-3-8B?
  Answer:   Understand CUDA cores, tensor cores, VRAM, and bandwidth — pick a card.

InferenceBase after Ch.2:
  Question: Will the model actually fit? What about training updates?
  Answer:   Estimate VRAM precisely — parameters, KV cache, optimizer states.

InferenceBase after Ch.3:
  Question: Can we go smaller without killing quality?
  Answer:   Quantize to INT4, benchmark perplexity, decide if the tradeoff holds.

InferenceBase after Ch.4:
  Question: How do we scale out when one GPU isn't enough?
  Answer:   Data parallelism, tensor parallelism, ZeRO — pick the right strategy.

InferenceBase after Ch.5:
  Question: How do we serve 10,000 requests/day efficiently?
  Answer:   KV cache, continuous batching, PagedAttention — throughput without waste.

InferenceBase after Ch.6:
  Question: Which serving framework — vLLM, TGI, TensorRT-LLM, or llama.cpp?
  Answer:   Benchmark each against the InferenceBase SLA (latency ≤ 2 s, cost ≤ $0.002/req).

InferenceBase after Ch.7:
  Question: We're adding a second node — what breaks at the network layer?
  Answer:   All-reduce, NVLink vs InfiniBand, NCCL — model the bandwidth.

InferenceBase after Ch.8:
  Question: Do we use AWS, Lambda Labs, RunPod, CoreWeave, or on-prem?
  Answer:   Cost-model each option against 30-day request projections.

InferenceBase after Ch.9:
  Question: How do we not lose 12 hours of fine-tuning to a preemption?
  Answer:   Checkpointing, fault tolerance, experiment tracking, SLURM vs k8s.

InferenceBase after Ch.10:
  Question: What does the full production stack look like?
  Answer:   End-to-end architecture: request → load balancer → serving cluster → GPU → response.
```

The key constraint: **InferenceBase has a $15,000/month cloud compute budget to replace $80,000/month in API costs, and the founding Platform Engineer has two weeks**. Every chapter confronts the tradeoffs that constraint forces.

---

## The Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI INFRASTRUCTURE STACK                               │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      OPERATIONS LAYER (Ch.9–10)                         │ │
│  │                                                                          │ │
│  │   Experiment Tracking · Checkpointing · Fault Tolerance                  │ │
│  │   SLURM · Kubernetes + Kubeflow · Full Production Architecture           │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                  │                                            │
│          ┌───────────────────────┴──────────────────────┐                   │
│          │                                               │                   │
│  ┌───────▼───────────────────────┐   ┌──────────────────▼───────────────┐  │
│  │   TRAINING LAYER (Ch.4, 7)    │   │   INFERENCE LAYER (Ch.5, 6)      │  │
│  │                                │   │                                   │  │
│  │  Data / Tensor / Pipeline     │   │  KV Cache · Continuous Batching  │  │
│  │  Parallelism · ZeRO Stages    │   │  PagedAttention · Speculative    │  │
│  │  DDP · FSDP · Megatron-LM    │   │  Decoding · Serving Frameworks   │  │
│  │  All-Reduce · NVLink / IB     │   │  vLLM · TGI · TensorRT-LLM      │  │
│  └───────────────────────────────┘   └──────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    OPTIMIZATION LAYER (Ch.3)                             │ │
│  │                                                                          │ │
│  │         FP16 / BF16 / INT8 / INT4 / FP8 · Loss Scaling                 │ │
│  │         GPTQ · AWQ · GGUF · Knowledge Distillation                      │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                  │                                            │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                    MEMORY LAYER (Ch.2)                                   │ │
│  │                                                                          │ │
│  │   Parameters · Activations · Optimizer States · KV Cache                │ │
│  │   VRAM Budget · Offloading Strategies · Flash Attention                  │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                  │                                            │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                    HARDWARE LAYER (Ch.1, 7, 8)                           │ │
│  │                                                                          │ │
│  │   CUDA Cores · Tensor Cores · HBM · Memory Bandwidth                    │ │
│  │   Roofline Model · NVLink · InfiniBand · Cloud GPU Instances             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Chapter Map

### Foundation

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.1 | [GPUArchitecture/](./GPUArchitecture/) | What does a GPU actually do, and how do its specs translate to AI workloads? |
| Ch.2 | [MemoryAndComputeBudgets/](./MemoryAndComputeBudgets/) | How much VRAM does this model actually need — at inference and at training? |

### Optimization

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.3 | [QuantizationAndPrecision/](./QuantizationAndPrecision/) | How do you shrink a model without destroying its quality? |

### Scale-Out

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.4 | [ParallelismAndDistributedTraining/](./ParallelismAndDistributedTraining/) | When one GPU is not enough — how do you split a model across many? |
| Ch.5 | [InferenceOptimization/](./InferenceOptimization/) | How do you serve thousands of requests efficiently without blowing the latency SLA? |
| Ch.6 | [ServingFrameworks/](./ServingFrameworks/) | vLLM vs TGI vs TensorRT-LLM vs llama.cpp — which one and when? |

### Infrastructure

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.7 | [NetworkingAndClusterArchitecture/](./NetworkingAndClusterArchitecture/) | What happens between the GPUs — bandwidth, latency, and collective operations? |
| Ch.8 | [CloudAIInfrastructure/](./CloudAIInfrastructure/) | AWS vs GCP vs Azure vs bare-metal cloud — how do you model the real cost? |

### Operations

| Chapter | File | Core Question |
|---------|------|---------------|
| Ch.9 | [MLOpsAndExperimentManagement/](./MLOpsAndExperimentManagement/) | How do you run training jobs that survive preemption, track experiments, and version models? |
| Ch.10 | [ProductionAIPlatform/](./ProductionAIPlatform/) | What does the complete stack look like when it's running in production for real users? |

---

## Reading Paths

### "I need to know if I can self-host this model"
→ Ch.1 → Ch.2 → Ch.3 → Ch.8

*Goal: hardware selection + VRAM sizing + quantization tradeoff + cloud cost model.*

### "I need to serve this model at scale"
→ Ch.5 → Ch.6 → Ch.7 → Ch.10

*Goal: inference optimization + serving framework selection + cluster networking + production architecture.*

### "I need to scale training beyond one GPU"
→ Ch.1 → Ch.2 → Ch.4 → Ch.7 → Ch.9

*Goal: GPU fundamentals + memory budgets + parallelism strategy + networking + MLOps.*

### "I want the full picture end to end"
→ Ch.1 → Ch.2 → Ch.3 → Ch.4 → Ch.5 → Ch.6 → Ch.7 → Ch.8 → Ch.9 → Ch.10

---

## Story Arc — How the Concepts Chain Together

```
START HERE
    │
    ▼
Step 0: UNDERSTAND THE HARDWARE
        Ch.1 — GPU Architecture

        Key insight: A GPU is not a fast CPU. It is a massively parallel
        matrix-multiply machine with a narrow memory bandwidth bottleneck.
        Every subsequent decision — quantization, parallelism, serving
        strategy — is an attempt to either feed that machine faster
        or do more work per memory access.
    │
    ▼
Step 1: COUNT YOUR MEMORY BEFORE YOU START
        Ch.2 — Memory & Compute Budgets

        Key insight: Most deployment failures are not code failures —
        they are VRAM failures. A 7B parameter model in FP16 needs
        ~14 GB just for weights. Add optimizer states, activations,
        and the KV cache and you are at 40–80 GB before writing
        a single line of training code.
    │
    ▼
Step 2: SQUEEZE THE MODEL
        Ch.3 — Quantization & Precision

        Key insight: INT4 quantization cuts memory by 8× vs FP32, with
        1–3% perplexity degradation on most models. BF16 training is
        almost always better than FP16. FP8 is the emerging standard
        for H100-class hardware.
    │
    ▼
Step 3: SCALE OUT THE TRAINING
        Ch.4 — Parallelism & Distributed Training

        Key insight: Data parallelism is trivial to implement but hits
        a wall at the per-GPU memory limit. Once the model no longer
        fits on one GPU, you need tensor or pipeline parallelism —
        and the communication cost becomes the dominant constraint.
    │
    ▼
Step 4: OPTIMISE THE INFERENCE PATH
        Ch.5 — Inference Optimization
        Ch.6 — Serving Frameworks

        Key insight: Training throughput and inference throughput have
        opposite optima. Training wants large batches; inference wants
        low latency. The KV cache, continuous batching, and PagedAttention
        are the mechanisms that reconcile these pressures.
    │
    ▼
Step 5: MODEL THE NETWORK
        Ch.7 — Networking & Cluster Architecture

        Key insight: NVLink within a node gives 600 GB/s. PCIe gives
        64 GB/s. InfiniBand between nodes gives 400 Gb/s. The ratio
        of compute time to communication time determines whether your
        distributed training scales or stalls.
    │
    ▼
Step 6: PICK YOUR CLOUD
        Ch.8 — Cloud AI Infrastructure

        Key insight: The meaningful comparison is not $/GPU-hour but
        $/useful-token. Spot instances cut costs by 60–80% but require
        fault-tolerant training. The right answer is almost never
        pure on-demand.
    │
    ▼
Step 7: OPERATIONALISE EVERYTHING
        Ch.9 — MLOps & Experiment Management
        Ch.10 — Production AI Platform

        Key insight: A training run without checkpointing is a liability.
        A serving cluster without auto-scaling is a guess. The operational
        layer is what separates a demo from a product.
```

---

## Chapter Detail — What Each Note Covers

### Ch.1 — GPU Architecture Fundamentals

**Core question:** What does a GPU actually do, and how do its specs translate to AI workloads?

**Concepts covered:**
- CUDA cores vs Tensor Cores vs SM (Streaming Multiprocessor) structure
- Memory hierarchy: registers → L1/L2 cache → HBM (VRAM) → system RAM
- Memory bandwidth vs compute throughput — why AI workloads are bandwidth-bound
- The Roofline Model: the graphical map of compute vs memory limits
- Arithmetic intensity: FLOPs per byte — where your operation lives on the roofline
- Key GPU generations: V100 (32 GB HBM2) → A100 (80 GB HBM2e) → H100 (80 GB HBM3) → consumer (RTX 4090 24 GB)
- GPU specs you actually need to read: TFLOPS (BF16), memory bandwidth (TB/s), VRAM capacity, NVLink bandwidth
- Why a matrix multiply maps naturally to GPU parallelism

**Notebook:** Roofline model calculator — given your GPU's specs and your operation's arithmetic intensity, visualise whether you are compute-bound or memory-bound. Compare representative operations: embedding lookup, attention, linear layer, convolution.

**InferenceBase angle:** Which GPU can run Llama-3-8B inference at 50 tokens/sec? Model the answer before ordering hardware.

---

### Ch.2 — Memory & Compute Budgets

**Core question:** How much VRAM does this model need — and will it fit?

**Concepts covered:**
- Parameter count → memory size: FP32 (4 B/param), FP16/BF16 (2 B/param), INT8 (1 B/param), INT4 (0.5 B/param)
- Inference memory = weights + KV cache + activations
- Training memory = weights + gradients + optimizer states (Adam = 3–4× weights in FP32)
- Mixed-precision training: FP16/BF16 forward + FP32 master weights + loss scaling
- KV cache growth: `2 × layers × heads × head_dim × seq_len × batch_size × bytes_per_element`
- Memory offloading: ZeRO-Offload, CPU offloading, disk offloading
- Flash Attention: recompute from Q/K/V instead of materialising the full $n^2$ attention matrix
- Model parallelism as a memory solution (preview of Ch.4)

**Notebook:** Interactive VRAM budget calculator — input model architecture (layers, heads, hidden dim, seq len, batch size, dtype) and get: inference VRAM, training VRAM with Adam, KV cache size as sequence length grows.

**InferenceBase angle:** Llama-3-8B (8B params, 32 layers, 32 heads, 4096 hidden dim) in BF16 = 16 GB weights + KV cache. Fit analysis across A10G (24 GB), A100 (40 GB / 80 GB), consumer RTX 4090 (24 GB).

---

### Ch.3 — Quantization & Precision

**Core question:** How do you shrink a model without destroying its quality?

**Concepts covered:**
- Floating point formats: FP32, FP16, BF16, FP8 (E4M3 / E5M2), INT8, INT4 — tradeoffs in range, precision, hardware support
- Why BF16 is preferred over FP16 for training (larger exponent range → less overflow)
- Post-training quantization (PTQ): apply after training, no retraining needed; fast but lossy
- Quantization-aware training (QAT): simulate quantization during training; better quality, expensive
- Weight-only quantization vs activation quantization vs KV cache quantization
- GPTQ: layer-wise PTQ with Hessian-based weight adjustment — the dominant format for LLM compression
- AWQ (Activation-aware Weight Quantization): identify salient weights, protect them from quantization
- GGUF / llama.cpp: CPU-friendly format, supports mixed-precision per tensor group
- Perplexity as a quantization quality metric
- The compression tradeoff table: INT4 saves 4–8× VRAM at 1–3% perplexity cost for most models

**Notebook:** NumPy quantization simulator — demonstrate FP32 → INT8 → INT4 rounding error on a weight tensor; plot the distribution of quantization error; compute effective precision loss.

**InferenceBase angle:** Llama-3-8B in GPTQ INT4 → 4.5 GB VRAM. Fits on a single RTX 3080 Ti (12 GB). Does the quality hold for document information extraction tasks?

---

### Ch.4 — Parallelism & Distributed Training

**Core question:** When one GPU is not enough — how do you split a model across many?

**Concepts covered:**
- Why one GPU is not enough: scaling laws, model size wall, memory wall
- **Data parallelism (DP/DDP):** replicate the full model on each GPU; split the batch; all-reduce gradients
- **Tensor parallelism (TP):** split individual weight matrices across GPUs (Megatron-style column/row splitting)
- **Pipeline parallelism (PP):** assign consecutive transformer layers to different GPUs; GPU bubbles
- **Sequence parallelism:** split along the sequence dimension for attention; reduces activation memory
- **ZeRO (Zero Redundancy Optimizer):** stages 1/2/3 — partition optimizer states, gradients, parameters across GPUs; compare to DP
- Gradient accumulation: simulate larger batch sizes without extra VRAM
- Communication collectives: all-reduce, all-gather, reduce-scatter — which parallelism uses which
- Frameworks: PyTorch DDP, FSDP (ZeRO-3 equivalent), DeepSpeed, Megatron-LM
- 3D parallelism: DP × TP × PP combined in large-scale training (GPT-4 class runs)

**Notebook:** Parallelism strategy simulator — model the effective throughput, communication overhead, and pipeline bubble fraction for a given model size, GPU count, and parallelism configuration.

**InferenceBase angle:** Fine-tuning Llama-3-8B on 4× A100s: which strategy is best? ZeRO-2 with DDP vs tensor parallelism vs FSDP — expected VRAM and training time for each.

---

### Ch.5 — Inference Optimization

**Core question:** How do you serve thousands of requests efficiently without blowing the latency SLA?

**Concepts covered:**
- The inference compute graph: prefill phase vs decode phase — different bottlenecks
- **KV cache:** store past-token key/value projections; avoid recomputing them at every decode step
- Why naive inference is inefficient: static batching GPU utilisation patterns
- **Continuous batching (iteration-level scheduling):** retire finished sequences, insert new ones mid-batch; the key vLLM innovation
- **PagedAttention:** store KV cache in non-contiguous memory pages (like virtual memory); eliminates fragmentation; enables large concurrent batch sizes
- **Speculative decoding:** small draft model generates candidates; large verifier model approves many tokens per forward pass
- **Flash Attention:** fused kernel that avoids materialising the full $O(n^2)$ attention matrix; cuts memory from $O(n^2)$ to $O(n)$; 2–4× faster on long sequences
- Throughput vs latency tradeoff curves: batch size ↑ → throughput ↑, latency ↑
- Prefill chunking: break long prompts into chunks to avoid blocking the decode queue
- Prefix caching: cache the KV states of a shared system prompt across all users

**Notebook:** Continuous batching throughput simulator — model requests arriving at a Poisson rate, simulate naive static batching vs continuous batching GPU utilisation; plot latency distribution and GPU idle fraction.

**InferenceBase angle:** At 10,000 document requests/day (avg 2,000 input tokens, 300 output tokens), how many A100s does InferenceBase need under each batching strategy?

---

### Ch.6 — Serving Frameworks

**Core question:** vLLM vs TGI vs TensorRT-LLM vs llama.cpp — which one and when?

**Concepts covered:**
- The serving framework landscape: what each layer provides (runtime, scheduler, API server)
- **vLLM:** PagedAttention, continuous batching, OpenAI-compatible API; Python-first; best for general LLM serving
- **Text Generation Inference (TGI):** Hugging Face; continuous batching, flash attention, tensor parallelism; tight HF Hub integration
- **TensorRT-LLM:** NVIDIA-specific; ahead-of-time compilation into optimised TRT engines; highest throughput on NVIDIA hardware; complex setup
- **llama.cpp:** CPU + Metal + CUDA; GGUF format; minimal dependencies; best for local/edge/developer use
- **Ollama:** user-friendly wrapper over llama.cpp; model management; not for high-concurrency production
- **Triton Inference Server:** NVIDIA multi-framework model server; protocol-agnostic (gRPC / HTTP); used in large-scale deployments with heterogeneous models
- OpenAI-compatible API layer: why all major frameworks converge on the same REST interface
- Benchmarking frameworks: `lm-evaluation-harness`, `llmperf`, `genai-perf`
- SLA definition: TTFT (time to first token), TBT (time between tokens), P50/P99 latency, throughput (tokens/sec)

**Notebook:** Framework comparison scorecard — given a model, hardware spec, and SLA, score each framework on ease of setup, expected throughput, memory efficiency, and ecosystem fit. No live GPU required: use published benchmark numbers from official leaderboards.

**InferenceBase angle:** Three-way benchmark for Llama-3-8B on A100 80GB: vLLM vs TGI vs TensorRT-LLM. InferenceBase SLA: TTFT < 500 ms, P99 latency < 2 s. Which framework wins, and what is the setup cost?

---

### Ch.7 — Networking & Cluster Architecture

**Core question:** What happens between the GPUs — bandwidth, latency, and collective operations?

**Concepts covered:**
- Intra-node communication: PCIe (64 GB/s bidirectional) vs NVLink Gen 4 (900 GB/s for H100 NVSwitch)
- Inter-node communication: Ethernet (100 Gb/s) vs InfiniBand NDR (400 Gb/s) vs RoCE (RDMA over Ethernet)
- RDMA (Remote Direct Memory Access): copy GPU memory to remote GPU memory bypassing the CPU
- NCCL (NVIDIA Collective Communications Library): the abstraction layer over all of the above
- Collective operations: **all-reduce** (sum gradients), **all-gather** (assemble shards), **reduce-scatter** (ZeRO's building block), **broadcast**, **point-to-point**
- Ring all-reduce algorithm: each GPU sends and receives in a ring; scales with bandwidth, not with GPU count
- Bandwidth-latency tradeoff: large messages → bandwidth-bound; small messages → latency-bound; implication for gradient accumulation step size
- Effective bandwidth vs theoretical bandwidth: busbw, algbw measurements
- Fat-tree topology: standard datacenter network; bisection bandwidth for all-to-all traffic
- Communication/computation overlap: async gradient communication during backward pass

**Notebook:** Ring all-reduce simulator — implement the ring all-reduce algorithm in NumPy; model end-to-end communication time for a given gradient tensor size, GPU count, and link bandwidth. Plot scaling efficiency vs GPU count.

**InferenceBase angle:** InferenceBase adds a second 8× A100 node for fine-tuning. NVLink within each node, InfiniBand between nodes. Model all-reduce time for a full fine-tuning gradient sync. Does communication dominate compute?

---

### Ch.8 — Cloud AI Infrastructure

**Core question:** AWS vs GCP vs Azure vs bare-metal cloud — how do you model the real cost?

**Concepts covered:**
- Cloud GPU instance families:
  - **AWS:** p3 (V100), p4d/p4de (A100), p5 (H100), g5 (A10G); SageMaker training jobs
  - **GCP:** a2 (A100), a3 (H100), g2 (L4); TPU v4/v5 (matrix multiply ASICs); Vertex AI
  - **Azure:** NDv4 (A100), NDm A100, NDv5 (H100); Azure ML
  - **Bare-metal cloud:** Lambda Labs, CoreWeave, RunPod, Vast.ai — no cloud markup, GPU-specific
- On-demand vs Reserved vs Spot/Preemptible: cost structures and interruption rates
- Total cost of inference: $/GPU-hour → $/1M tokens (accounting for utilisation, quantization, batching)
- Storage considerations: EBS (throughput-limited for large checkpoints), EFS/FSx for Lustre (shared filesystem for multi-node training), GCS/S3 (checkpoint storage)
- Networking costs between regions and availability zones — a frequently overlooked bill item
- When to self-host vs when to use a managed API: the economic crossover point
- TPUs vs GPUs: matrix multiply architecture differences; JAX vs PyTorch on TPUs; when TPUs win (large-scale training of transformer-only models)
- Multi-cloud and hybrid strategies: avoid vendor lock-in at the compute layer

**Notebook:** Cloud cost model for InferenceBase — given model specs, request volume, and SLA, compute the 30-day cost on: (1) OpenAI API, (2) AWS p4d on-demand, (3) AWS p4d spot, (4) Lambda Labs on-demand, (5) CoreWeave reserved. Plot the crossover points.

**InferenceBase angle:** At what monthly request volume does self-hosting become clearly cheaper than the OpenAI API? The answer is in this chapter.

---

### Ch.9 — MLOps & Experiment Management

**Core question:** How do you run training jobs that survive preemption, track experiments, and version models?

**Concepts covered:**
- Why MLOps is not optional: reproducibility, auditability, and the lost-training-run tax
- **Experiment tracking:** MLflow (open source, self-hosted), Weights & Biases (managed, richer UI) — what to log: hyperparameters, loss curves, hardware metrics, eval scores, model artifacts
- **Checkpointing strategies:** save every N steps; save best-by-metric; sharded checkpointing for large models (PyTorch `DistributedCheckpoint`, DeepSpeed checkpoint)
- **Fault-tolerant training:** NCCL timeout handling, elastic training (PyTorch `torchrun` with `--rdzv_backend`), automatic job restart on preemption
- **Model versioning and registries:** MLflow Model Registry, HuggingFace Hub, W&B Artifacts — tag, stage (staging → production), lineage
- **Data versioning:** DVC (Data Version Control) — version large datasets without storing them in git; remote storage backends
- **Resource scheduling:**
  - **SLURM:** HPC-standard job scheduler; node allocation, job queues, partitions; `sbatch` scripts; multi-node distributed jobs
  - **Kubernetes + Kubeflow:** cloud-native; `PyTorchJob` CRD; autoscaling; heterogeneous hardware; harder to set up, better for multi-tenant clusters
  - **Ray Train:** Python-native distributed training abstraction; simpler than raw DDP; integrates with Ray Serve for serving
- **CI/CD for ML:** evaluation gates before promoting a model; automated regression testing on model quality; integration with GitHub Actions / GitLab CI
- Monitoring training health: loss spikes, gradient norms, GPU utilisation, dead neurons

**Notebook:** MLflow experiment tracker simulation — log a hyperparameter sweep (learning rate, batch size, weight decay) with synthetic loss curves; demonstrate model registration, staging, and artifact lineage. Runs entirely locally with `mlflow` and `pandas`.

**InferenceBase angle:** A fine-tuning run on 4× A100s got preempted at step 8,000 of 20,000 on a spot instance. How do you recover? Checkpoint strategy + job restart config + cost of the interrupted run.

---

### Ch.10 — Production AI Platform — The Full Stack

**Core question:** What does the complete stack look like when it's running in production for real users?

**Concepts covered:**
- Synthesising all previous chapters into a coherent reference architecture
- **Request lifecycle:** client → load balancer → API gateway → request queue → serving cluster (vLLM) → GPU → token stream → client
- **Auto-scaling inference:** Kubernetes HPA on GPU utilisation metric; KEDA for queue-depth-driven scaling; scale-to-zero for low-traffic periods
- **Latency SLA enforcement:** P50 / P95 / P99 targets; queue depth monitoring; request timeout and circuit breaking; graceful degradation (fall back to smaller model)
- **Multi-model serving:** route different request types to different model variants (8B vs 70B vs fine-tuned); A/B testing at the routing layer
- **Observability stack:** GPU metrics (DCGM Exporter → Prometheus → Grafana), LLM-specific metrics (TTFT, TBT, token_throughput), application traces (OpenTelemetry), log aggregation (Loki / ELK)
- **Cost vs performance optimisation loop:** continuous benchmarking → quantization → batch size tuning → instance type review → repeat
- **Security considerations:** prompt injection at the infrastructure level (input sanitisation, output validation), model access control, audit logging
- **Build vs buy decision framework:** when to use a managed LLM inference platform (Anyscale, Fireworks, Together AI, Replicate) vs fully self-hosted vs hybrid
- **Capacity planning:** traffic forecasting → GPU count → cost projection; headroom rules; burst handling

**Notebook:** End-to-end production cost and latency model — given a request rate (RPS), model specs, GPU type, and batching strategy, compute: required GPU count, P99 latency estimate, monthly cost, and utilisation headroom. The InferenceBase final architecture decision.

**InferenceBase angle:** The full InferenceBase production stack blueprint: serving 10,000 req/day → 50,000 req/day growth plan. Hardware, framework, cloud, MLOps tooling, observability, and total cost. This chapter closes the loop on everything that came before.

---

## What Each Chapter Contains

Every chapter in this track follows the same structure (mirroring the ML and AI tracks):

```
notes/AIInfrastructure/
├── GPUArchitecture/
│   ├── GPUArchitecture.md     ← Technical deep-dive: concepts, diagrams, interview checklist
│   └── notebook.ipynb         ← Calculator / simulator that makes the concepts concrete
├── MemoryAndComputeBudgets/
│   ├── MemoryAndComputeBudgets.md
│   └── notebook.ipynb
│  ... (10 chapters total)
```

### Chapter README structure

```
# Ch.N — [Topic Name]

## Core Idea (3–4 sentences, plain English)

## The InferenceBase Angle
(one paragraph: how does this chapter's concept hit the startup scenario?)

## The Concepts
(key ideas, equations where applicable — every term explained inline)

## How It Works — Step by Step
(numbered walkthrough or decision flow in Mermaid/ASCII)

## The Key Diagram
(Mermaid or ASCII art — minimum 1)

## The Numbers That Matter
(concrete benchmarks, reference specs, order-of-magnitude estimates)

## What Can Go Wrong
(3–5 failure modes, one sentence each)

## Interview Checklist
| Must know | Likely asked | Trap to avoid |

## Bridge to the Next Chapter
```

### Notebook structure

```
[markdown] Chapter title + one-liner
[markdown] ## Core Idea
[markdown] ## The InferenceBase Setup
[code]     Define model/hardware parameters (user fills these in)
[markdown] ## The Math / Simulation
[code]     Implement the calculator, model, or simulator
[code]     Visualise the result
[markdown] ## Sensitivity Analysis
[code]     Sweep a key variable — what changes?
[markdown] ## Exercises
[code]     2–3 prompts for the reader to explore independently
```

---

## Interview Checklist — The Track in 90 Seconds

| Topic | Must Know | Common Trap |
|-------|-----------|-------------|
| GPU architecture | Tensor Cores do INT/FP matrix multiply; CUDA cores do scalar FP | "More CUDA cores = faster" — no, bandwidth is usually the real limit |
| Memory | 7B model in BF16 = 14 GB weights; add KV cache, activations, optimizer states | Forgetting KV cache growth with sequence length and batch size |
| Quantization | INT4 saves ~8× VRAM vs FP32; GPTQ/AWQ are leading PTQ methods | Quantization is not free: perplexity degrades, especially for reasoning tasks |
| Parallelism | Data parallelism = gradient sync; Tensor parallelism = weight sharding | "More GPUs = linear speedup" — communication overhead breaks this |
| Inference optimization | KV cache avoids recomputing past tokens; continuous batching keeps GPU busy | Confusing prefill (prompt processing) with decode (generation) — very different bottlenecks |
| Serving frameworks | vLLM is the dominant open-source choice for production GPU serving | "Just use Ollama in production" — Ollama is for local dev, not high-concurrency serving |
| Networking | All-reduce is the critical collective for DDP; NVLink >> PCIe for intra-node | Forgetting inter-node bandwidth is a hard limit on multi-node scaling |
| Cloud | $/GPU-hour is not the right metric — $/token at your SLA is | On-demand GPU instances for training = leaving 60–80% of budget on the table |
| MLOps | Checkpoint early and often; fault-tolerant training is not optional on spot | "We'll add checkpointing later" — the later never comes, you lose the run |
| Production | TTFT and P99 latency are the user-visible metrics; throughput is the cost metric | Optimising for throughput at the cost of P99 latency — users notice, accountants don't |

---

## Connections to Other Tracks

| This chapter | Connects to |
|---|---|
| Ch.1 (GPU Architecture) | ML Ch.18 (Transformers) — the operations that run on those tensor cores |
| Ch.2 (Memory Budgets) | AI CostAndLatency.md — the VRAM side of the same cost model |
| Ch.3 (Quantization) | AI FineTuning.md — QLoRA is quantization + LoRA combined |
| Ch.4 (Parallelism) | AI FineTuning.md — FSDP and DeepSpeed are used for large fine-tuning runs |
| Ch.5 (Inference Opt.) | AI CostAndLatency.md — KV cache and batching are the mechanism behind the cost numbers |
| Ch.6 (Serving) | AI ReActAndSemanticKernel.md — agents call LLMs via exactly these serving APIs |
| Ch.10 (Production) | MultimodalAI LocalDiffusionLab — the same serving patterns apply to diffusion models |

---

> **Status:** Roadmap complete. Chapters are built in order. See individual chapter folders for current state.
