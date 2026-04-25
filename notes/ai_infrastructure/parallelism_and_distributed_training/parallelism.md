# Ch.4 — Parallelism & Distributed Training  (Training Focus - Not Blocking Launch)

> **The story.** Training a large model on a single GPU is slow. Training GPT-3 (175B params) on one V100 would take 355 years. **Data parallelism** (Goyal et al., Facebook, **2017**) split batches across GPUs — if 1 GPU trains on 32 examples, 8 GPUs train on 256 → 8× faster. But memory was still the bottleneck. **ZeRO** (Rajbhandari et al., Microsoft, **2019**) sharded optimizer states across GPUs, cutting per-GPU memory by 4–8× → train 10× larger models. **FSDP** (Zhao et al., Meta, **2023**) took this further: shard parameters themselves, not just optimizer states → train 1T-param models on 128 GPUs. **LoRA** (Hu et al., Microsoft, **2021**) sidestepped the problem: freeze base model, only train small adapter weights → fine-tune 65B models on single consumer GPUs.
>
> **Where you are in the curriculum.** Ch.1-5 solved the inference problem (12k req/day, <2s latency). This chapter tackles training: **can we fine-tune Llama-3-8B to improve document extraction accuracy?** Ch.2 showed full fine-tuning needs 104GB → requires expensive A100s. This chapter introduces ZeRO-2 (shard optimizer across GPUs) and LoRA (train only adapters) → fine-tune on RTX 4090 budget.
> <!-- TODO: add notation sentence here -->

---

## 0 · The Challenge — Where We Are

## Animation

> 🎬 *Animation placeholder — needle-builder agent will generate this.*


> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1-3: Inference works (12k req/day, 1.2s latency, 96.2% accuracy) ✅
- ✅ Ch.5: Optimized to 22k req/day, 680ms latency ✅
- ⚡ **Quality** note: 96.2% accuracy is above 95% threshold, but CEO sees opportunity for improvement

**What's (potentially) blocking us**:

⚡ **Training not blocking launch, but needed for future improvement**

**Current situation**: 2 weeks post-launch, CEO reviewing analytics

```
Launch week analytics:
✅ Throughput: 12,000 req/day (target hit!)
✅ Latency: 1.2s p95 (target hit!)
✅ Cost: $1,095/month (under budget!)
✅ Accuracy: 96.2% (above 95% target!)

CEO: "Great launch! But I see an opportunity. Our competitors claim 98%+ accuracy.
     Can we fine-tune Llama-3-8B on our proprietary invoice dataset (50,000 PDFs)
     to push from 96.2% → 98%?"

Engineer: "Let me calculate training memory requirements:
 
 Ch.2 showed full fine-tuning needs:
   16GB params (FP16)
 + 64GB optimizer states (Adam FP32)
 + 16GB gradients (FP16)
 + 8GB activations
 = 104GB total
 
 RTX 4090 only has 24GB → cannot fit! ❌
 
 Options:
 1. Rent A100 80GB × 2 with ZeRO-2 sharding ($6,000/month for 2 weeks training)
 2. Use LoRA (Low-Rank Adaptation): freeze base model, only train 0.5% of params
    → 16GB params (frozen) + 2GB adapter + 6GB optimizer = 24GB total ✅ (fits!)
 
CEO: "What's the trade-off with LoRA?"

Engineer: "LoRA is parameter-efficient but slightly less expressive than full fine-tuning.
          Typically gets 90-95% of full fine-tuning quality. For document extraction,
          should push us from 96.2% → 97.5%+ (vs 98% with full fine-tuning).
          
          Key benefit: $1,095/month RTX 4090 × 2 weeks = $500 training cost
          vs $6,000 for A100s."

CEO: "Do LoRA. If it works, we save $5,500. If quality improvement is insufficient,
     we can always do full fine-tuning later."
```

**Problems** (for future fine-tuning, not blocking launch):
1. ❌ **Full fine-tuning too expensive**: 104GB → need A100 80GB × 2 ($6k/month)
2. ⚡ **LoRA untested**: Team has no experience with parameter-efficient fine-tuning
3. ⚡ **Training infrastructure unknown**: Need multi-GPU data parallelism, gradient accumulation, checkpointing
4. ⚡ **Quality uncertainty**: Will LoRA match full fine-tuning quality on document extraction?
5. ⚡ **Deployment complexity**: How to swap base model + adapters without downtime?

**Business impact**:
- **98% accuracy = competitive edge**: Competitors claim 98%+ → we need parity
- **$5,500 training cost savings**: LoRA on RTX 4090 ($500) vs full fine-tuning on A100 ($6k)
- **Fast iteration**: LoRA trains in 2 days vs 2 weeks for full fine-tuning → faster experiments

**What this chapter unlocks**:

🚀 **Cost-efficient fine-tuning with LoRA + ZeRO-2**:
1. **LoRA setup**: Freeze Llama-3-8B, add low-rank adapters to attention layers (rank=16 → 0.5% trainable params)
2. **Training memory**: 16GB frozen params + 2GB adapter + 6GB optimizer = 24GB ✅ (fits RTX 4090!)
3. **Data parallelism (DDP)**: Distribute batches across 2× RTX 4090 → 2× faster training
4. **ZeRO-2 fallback**: If LoRA insufficient, shard optimizer states → full fine-tune on 4× RTX 4090
5. **Quality validation**: Train on 50k PDFs, test on 5k holdout → measure accuracy improvement

⚡ **Expected outcomes**:
- **Training cost**: $1,095/month × 2 weeks = $500 (vs $6k A100 option)
- **Training time**: 2 days on 2× RTX 4090 (vs 2 weeks full fine-tuning)
- **Quality improvement**: 96.2% → 97.8% accuracy ✅ (+1.6 points, close to 98% target)
- **Deployment**: Hot-swap LoRA adapters without reloading base model → zero downtime

**Constraint status after Ch.4**:
- #1 (Cost): ✅ **MAINTAINED** ($1,095/month + $500 one-time training)
- #2 (Latency): ✅ **MAINTAINED** (LoRA adds <5ms inference overhead)
- #3 (Throughput): ✅ **MAINTAINED** (12k req/day unchanged)
- #4 (Memory): ✅ **TRAINING FITS** (24GB for LoRA, or 96GB for full fine-tuning with 4× GPU + ZeRO-2)
- #5 (Quality): ✅ **IMPROVED!** (96.2% → 97.8% with LoRA fine-tuning)
- #6 (Reliability): ✅ **MAINTAINED** (adapter swaps don't disrupt service)

**Critical note**: **Training is NOT blocking launch**. Ch.4 enables future quality improvements, but the system is already production-ready at 96.2% accuracy.

---

## 1 · Core Idea

**Data Parallelism** = replicate model on N GPUs, split batch across GPUs:
- Each GPU computes gradients on its micro-batch
- All-reduce gradients across GPUs
- All GPUs update model weights (same final weights)

**Model Parallelism** = split model layers across GPUs (needed for 70B+ models that don't fit on 1 GPU)

**ZeRO** = shard optimizer states + gradients + parameters across GPUs → reduce per-GPU memory without full model parallelism

**LoRA** = freeze base model, add low-rank adapter matrices → train 0.1–1% of parameters

---

## 2 · Running Example

**Goal**: Fine-tune Llama-3-8B on 50,000 proprietary invoice PDFs to improve accuracy from 96.2% → 98%.

**Memory constraint**: RTX 4090 (24GB) cannot fit full fine-tuning (104GB).

**Solution**: LoRA (Low-Rank Adaptation)
- Freeze 8B parameters (16GB)
- Add 42M trainable adapter parameters (84MB)
- Total training memory: 16GB frozen + 2GB adapter + 6GB optimizer = 24GB ✅

**Result**: Train on 2× RTX 4090 in 2 days → accuracy improves to 97.8% ✅ (vs 98% target).

---

## 3 · Data Parallelism (DDP)

Standard PyTorch distributed training:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group (one process per GPU)
dist.init_process_group(backend='nccl', world_size=2, rank=rank)

# Wrap model in DDP
model = DistributedDataParallel(model, device_ids=[rank])

# Training loop (each GPU processes different batch)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Compute local gradients
    # DDP automatically all-reduces gradients across GPUs here
    optimizer.step()  # Update weights (same on all GPUs)
```

**Memory**: Each GPU stores full model (16GB) + optimizer (64GB) + gradients (16GB) = 96GB per GPU ❌

**Speedup**: 2× GPUs → 2× throughput (if batch size doubles), but still memory-constrained.

---

## 4 · ZeRO-2: Shard Optimizer States

**Problem**: Adam optimizer stores momentum + variance = 2× param memory (64GB for 8B model).

**ZeRO-2 solution**: Each GPU stores only its shard of optimizer states:
- GPU 0: Stores optimizer for layers 0-15 (32GB)
- GPU 1: Stores optimizer for layers 16-31 (32GB)
- During backward pass: gradients are reduced, each GPU updates its shard
- Before forward pass: all-gather updated parameters

**Memory savings**: 64GB optimizer → 32GB per GPU (2× GPUs) ✅

```python
from deepspeed import zero

# Enable ZeRO-2 in DeepSpeed config
config = {
    "zero_optimization": {
        "stage": 2,  # Shard optimizer + gradients
        "offload_optimizer": False,  # Keep on GPU
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=config,
)

# Training loop (ZeRO handles sharding automatically)
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**Memory** (4× RTX 4090 with ZeRO-2):
- 16GB params (replicated on each GPU)
- 16GB optimizer per GPU (64GB total, sharded)
- 4GB gradients per GPU (16GB total, sharded)
- **Total per GPU: 16 + 16 + 4 = 36GB** → fits on A100 40GB, not RTX 4090 24GB ❌

→ Need gradient checkpointing or LoRA to fit on RTX 4090.

---

## 5 · LoRA (Low-Rank Adaptation)

**Key insight**: Model updates during fine-tuning are low-rank (most weight changes lie in a small subspace).

Instead of updating $W \in \mathbb{R}^{d \times k}$, learn low-rank decomposition:
$$W' = W + \Delta W = W + BA$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$ (e.g., $r=16$).

**Parameters**:
- Original: $d \times k = 4096 \times 4096 = 16M$ params per attention layer
- LoRA: $d \times r + r \times k = 4096 \times 16 + 16 \times 4096 = 131K$ params (99.2% reduction!)

**Llama-3-8B with LoRA (rank=16)**:
- Frozen base: 8,030M params (16GB)
- Trainable adapters: 42M params (84MB)
- Total training memory: 16GB frozen + 2GB adapter (including optimizer) = **18GB** ✅ (fits RTX 4090!)

```python
from peft import LoraConfig, get_peft_model

# Apply LoRA to attention layers only
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
)

model = get_peft_model(base_model, lora_config)

print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")
print(f"Total params: {model.num_parameters():,}")
# Output:
# Trainable params: 42,467,328  (0.53% of total)
# Total params: 8,030,261,248
```

---

## 6 · Fine-Tuning Llama-3-8B with LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2. Apply LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 3. Load invoice extraction dataset
dataset = load_dataset("invoices_proprietary", split="train[:50000]")

# 4. Training config (2× RTX 4090, DDP)
training_args = TrainingArguments(
    output_dir="./llama3-invoice-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 4 × 4 × 2 GPUs = 32
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    warmup_steps=100,
)

# 5. Train (Hugging Face Trainer handles DDP automatically)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# 6. Save adapter weights only (84MB vs 16GB base model)
model.save_pretrained("./llama3-invoice-lora-adapters")
```

**Training time**: 2 days on 2× RTX 4090 (vs 2 weeks full fine-tuning on A100s).

---

## 7 · Evaluating LoRA Fine-Tuning Quality

Test on 5,000-PDF holdout set:

| Model | Accuracy | Training Cost | Training Time |
|-------|----------|---------------|---------------|
| Base Llama-3-8B (zero-shot) | 96.2% | $0 | 0 |
| LoRA fine-tuned (rank=16) | 97.8% | $500 | 2 days |
| Full fine-tuning (projected) | 98.1% | $6,000 | 2 weeks |

**Result**: LoRA achieves 95% of full fine-tuning quality at 8% of cost ✅

**Failure analysis** (cases where LoRA < full fine-tuning):
- Complex multi-page invoices with tables spanning pages (LoRA: 92%, full: 95%)
- Handwritten invoice amounts (LoRA: 88%, full: 93%)
- Foreign language invoices (Spanish, French) (LoRA: 94%, full: 97%)

**Conclusion**: LoRA sufficient for most use cases. If quality needs push to 98%+, consider full fine-tuning later.

---

## 8 · Deploying LoRA Adapters in Production

**Key advantage**: Adapters are tiny (84MB) → can hot-swap without reloading base model (16GB).

```python
from peft import PeftModel

# Load base model once (16GB)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
)

# Load adapter A (invoice extraction, 84MB)
model_invoices = PeftModel.from_pretrained(base_model, "./adapters/invoices")

# Serve 10,000 invoice requests...

# Hot-swap to adapter B (receipt extraction, 84MB) — no base model reload!
model_receipts = PeftModel.from_pretrained(base_model, "./adapters/receipts")

# Serve 5,000 receipt requests...
```

**Memory**: 16GB base (loaded once) + 84MB adapter (swapped in <1s) ✅

**Use case**: Multi-tenant serving (different adapters for different customers, same base model).

---

## 9 · When to Use Each Strategy

| Scenario | Strategy | Why |
|----------|----------|-----|
| Fine-tune 8B model on 24GB GPU | LoRA (rank=16) | Fits in 18GB, 0.5% trainable params |
| Fine-tune 8B model on 4× 24GB GPUs | ZeRO-2 + gradient checkpointing | Shard optimizer, recompute activations |
| Fine-tune 70B model on 8× A100 80GB | ZeRO-3 + tensor parallelism | Shard params + optimizer + gradients |
| Train from scratch (13B+ model) | FSDP (Fully Sharded Data Parallel) | Most memory-efficient for large-scale training |
| Multi-task learning (10 tasks) | LoRA per task | 10× 84MB adapters vs 10× 16GB models |

**InferenceBase choice**: LoRA for invoice fine-tuning → 97.8% accuracy ✅, $500 cost ✅

---

## 10 · The Key Diagram

### Training Memory: Full Fine-Tuning vs LoRA

```
Full Fine-Tuning (104GB — does NOT fit RTX 4090):
├─ Parameters (FP16): 16GB
├─ Optimizer states (FP32): 64GB  ← 60% of memory!
├─ Gradients (FP16): 16GB
└─ Activations: 8GB

LoRA Fine-Tuning (18GB — fits RTX 4090 ✅):
├─ Parameters (frozen, FP16): 16GB (read-only)
├─ Adapter params (FP16): 84MB
├─ Adapter optimizer (FP32): 1.5GB (only for 42M adapter params)
└─ Adapter gradients + activations: 500MB

Memory savings: 104GB → 18GB (83% reduction!)
```

---

## 8 · What Can Go Wrong

- **LoRA rank too low (r=4)** — under-parameterized, cannot capture task complexity → poor quality
- **LoRA rank too high (r=128)** — defeats the purpose, memory approaches full fine-tuning
- **Forgetting to freeze base model** — if base weights are trainable, optimizer memory explodes
- **Not using gradient accumulation** — RTX 4090 can only fit batch=4; need accumulation to reach effective batch=32
- **Assuming LoRA = same quality as full fine-tuning** — typically 90-95% of quality; measure on your task

---

## The Hyperparameter Dial

LoRA has three independent dials. Getting them wrong either wastes memory or degrades quality.

### Dial 1 — LoRA Rank $r$

| Rank $r$ | Adapter params (Llama-3-8B) | Extra VRAM | Accuracy (typical) | Notes |
|---------|----------------------------|------------|--------------------|-------|
| 4 | ~21M | +0.1 GB | ~92% | Too low for complex tasks |
| 8 | ~42M | +0.2 GB | ~95% | Good for simple classification |
| 16 | ~84M | +0.3 GB | ~97% | **Sweet spot for most tasks** |
| 32 | ~168M | +0.6 GB | ~98% | Only marginal gain over 16 |
| 64 | ~336M | +1.3 GB | ~98.5% | Approaching full fine-tuning cost |

> 💡 Start with r=16 and evaluate. Moving from 16→32 rarely justifies the memory cost; 16→8 is a safe fallback when VRAM is tight.

### Dial 2 — ZeRO Stage

ZeRO (Zero Redundancy Optimizer) shards different optimizer components across GPUs:

| ZeRO Stage | What is sharded | Memory per GPU | Communication overhead | Use when |
|------------|-----------------|----------------|------------------------|----------|
| 0 (none) | Nothing | Full copy | None | Single GPU |
| 1 | Optimizer states | ~0.5× | Low | 2–4 GPUs, bandwidth-heavy |
| 2 | Gradients + optimizer | ~0.25× | Medium | 4–8 GPUs |
| 3 | Params + grads + optimizer | ~0.12× | High | 8+ GPUs; requires NVLink |

> ⚠️ ZeRO-3 over PCIe (no NVLink) often runs slower than ZeRO-2 with gradient checkpointing — the all-gather communication dominates.

### Dial 3 — Gradient Accumulation Steps

Effective batch size = `per_device_batch` × `gradient_accumulation_steps`. On RTX 4090 (24 GB):

| Config | Memory/step | Effective batch | Quality |
|--------|-------------|-----------------|---------|
| per_batch=4, accum=1 | 18 GB | 4 | Underfit |
| per_batch=4, accum=8 | 18 GB | 32 | Good |
| per_batch=4, accum=16 | 18 GB | 64 | Near-full training quality |
| per_batch=8, accum=8 | ~22 GB | 64 | Risk of OOM |

---

## Code Skeleton

```python
# Educational: minimal LoRA from scratch
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Low-rank adaptation of a linear layer.
    Adds trainable A (down-project) and B (up-project) matrices while freezing W.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank  # LoRA scaling factor
        self.A = nn.Linear(in_features, rank, bias=False)  # down-project
        self.B = nn.Linear(rank, out_features, bias=False)  # up-project
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)  # B starts at 0 so ΔW = 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scale  # ΔW·x = B·A·x · (α/r)
```

```python
# Production: LoRA fine-tuning with HuggingFace PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def finetune_with_lora(base_model_id: str, dataset, output_dir: str,
                        rank: int = 16, alpha: float = 32) -> str:
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="bfloat16",
                                                  device_map="auto")
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # attention layers
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # ~0.5% of total

    training_args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch = 32
        num_train_epochs=3, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy="epoch"
    )
    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)
    trainer.train()
    model.save_pretrained(output_dir)
    return output_dir
```

---

## Where This Reappears

| Chapter | How LoRA / parallelism concepts appear |
|---------|----------------------------------------|
| **Ch.2 — Memory Budgets** | LoRA training VRAM formula (params + gradients + optimizer states) builds on the parameter memory model from Ch.2 |
| **Ch.5 — Inference Optimization** | Adapter serving at inference time: LoRA adapters from this chapter are loaded into the base model at request time |
| **Ch.6 — vLLM & Serving** | vLLM supports LoRA hot-swap: multiple adapters loaded simultaneously, base model shared — enabled by the adapter architecture here |
| **Multi-agent AI track** | Specialized agents (code agent, retrieval agent) can use task-specific LoRA adapters — adapter routing uses the hot-swap pattern from Ch.6 |
| **Fine-tuning (AI track)** | QLoRA = LoRA over INT4-quantized base model; combines Ch.3 quantization with this chapter's LoRA pattern |

---

## 11.5 · Progress Check — What We've Accomplished

🎉 **COST-EFFICIENT FINE-TUNING ENABLED! LoRA trains on RTX 4090 budget ✅**

**Unlocked capabilities**:
- ✅ **LoRA setup**: Freeze 8B base, train 42M adapter params (0.5% of model)
- ✅ **Training memory**: 18GB (fits RTX 4090 24GB) ✅
- ✅ **Training cost**: $500 (vs $6k A100 option) → 92% savings ✅
- ✅ **Quality improvement**: 96.2% → 97.8% accuracy (+1.6 points) ✅
- ✅ **Hot-swap adapters**: 84MB adapters load in <1s, no base model reload

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MAINTAINED** | $1,095/month + $500 training (under budget) |
| #2 LATENCY | ✅ **MAINTAINED** | LoRA adds <5ms overhead (negligible) |
| #3 THROUGHPUT | ✅ **MAINTAINED** | 12k req/day unchanged |
| #4 MEMORY | ✅ **TRAINING FITS** | 18GB LoRA (vs 104GB full fine-tuning) |
| #5 QUALITY | ✅ **IMPROVED!** | 96.2% → 97.8% (+1.6 points, close to 98% target) |
| #6 RELIABILITY | ✅ **MAINTAINED** | Adapter swaps don't disrupt service |

**What we can solve now**:

✅ **Fine-tune without blowing budget**:
```
Before Ch.4:
CEO: "Can we fine-tune to improve from 96.2% → 98%?"

Engineer: "Full fine-tuning needs 104GB → require A100 80GB × 2 = $6,000/month
           for 2 weeks = $3,000 training cost."

CEO: "That's half our monthly budget just for training. Not approved."

After Ch.4:
Engineer: "LoRA alternative:
 - Freeze base model (16GB)
 - Train tiny adapters (84MB, 0.5% of params)
 - Total memory: 18GB (fits RTX 4090!)
 - Training cost: $1,095/month × 2 weeks = $500
 - Quality: 96.2% → 97.8% (vs 98.1% full fine-tuning)
 
 We get 95% of full fine-tuning quality at 17% of cost."

CEO: "Approved. If 97.8% isn't enough, we revisit full fine-tuning later."

Result: ✅ $500 training cost vs $3,000 A100 option → $2,500 saved!
```

✅ **Multi-tenant serving with adapters**:
```
Future opportunity:
Customer A: Needs invoice extraction (adapter A, 84MB)
Customer B: Needs receipt extraction (adapter B, 84MB)
Customer C: Needs purchase order extraction (adapter C, 84MB)

Instead of:
3× separate 16GB models = 48GB VRAM (needs 2× RTX 4090)

Use:
1× 16GB base model + hot-swap 84MB adapters = 16.3GB VRAM ✅ (fits 1× RTX 4090)

Result: ✅ 3× revenue capacity without additional hardware!
```

✅ **Fast experimentation**:
```
Before Ch.4:
Full fine-tuning: 2 weeks per experiment → 1 experiment/month

After Ch.4:
LoRA: 2 days per experiment → 10 experiments/month ✅

Result: ✅ 10× faster iteration for quality improvements!
```

**What's still blocking**:

- ⚡ **Serving framework unknown**: Need vLLM/TGI to deploy LoRA adapters efficiently → **Ch.6**
- ⚡ **No multi-GPU redundancy**: Single RTX 4090 = single point of failure → **Ch.7**
- ⚡ **No production observability**: How do we monitor adapter performance? → **Ch.9**

**Next chapter**: [Inference Optimization](../inference_optimization) (already completed) → [Serving Frameworks](../ServingFrameworks)

**Key interview concepts from this chapter**:

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Data parallelism replicates model, splits batch; model parallelism splits model layers | What's the difference between data and model parallelism? | Confusing the two — data parallelism is for speedup, model for memory |
| ZeRO-2 shards optimizer states across GPUs → 2-4× memory savings per GPU | How does ZeRO reduce memory? | Thinking it compresses weights — it shards optimizer/gradients |
| LoRA freezes base, trains low-rank adapters (r=16 typical) → 0.1-1% trainable params | Can you fine-tune 70B model on consumer GPU? | Saying "no" — LoRA + quantization makes it possible! |
| LoRA quality = 90-95% of full fine-tuning (task-dependent) | Is LoRA always better than full fine-tuning? | Not mentioning quality trade-off — LoRA is parameter-efficient, not quality-optimal |
| Gradient accumulation: batch=4 × 8 accumulation steps = effective batch=32 | How to train with large batch size on small GPU? | Forgetting gradient accumulation (most common solution) |

---

## 12 · Bridge to Chapter 5

Ch.4 enabled cost-efficient fine-tuning, but didn't address production inference bottlenecks. Ch.5 (Inference Optimization) returns to the inference path: continuous batching (eliminate queue wait spikes), PagedAttention (double batch size via better memory management), and speculative decoding (30% faster generation). These optimizations push throughput from 12k → 22k req/day and reduce tail latency under spiky traffic from 8.7s → 1.8s.

## Illustrations

![Parallelism — Data parallelism (DDP), ZeRO optimizer sharding, LoRA adapter architecture, memory comparison](img/Parallelism.png)


## 5 · Key Diagrams

> Add 2–3 diagrams showing the key data flows or architectural boundaries here.


## 6 · The Hyperparameter Dial

> List 3–5 dials (batch size, precision, parallelism strategy, etc.) and their
> effect on the latency/throughput/memory triangle.


## 7 · Code Skeleton

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

