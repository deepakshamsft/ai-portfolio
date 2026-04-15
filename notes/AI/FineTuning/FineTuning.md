# Fine-Tuning, PEFT & LoRA — Adapting Models Without Retraining From Scratch

> **The decision of *when* to fine-tune is more important than *how* to fine-tune.** Most applications that reach for fine-tuning too early could have solved their problem with better prompt engineering or RAG — at a fraction of the cost and complexity. This document covers when fine-tuning is the right call, and how to do it efficiently with parameter-efficient methods.

---

## 1 · Core Idea

**Full fine-tuning** retrains all parameters of a pretrained model on a new dataset. For a 7B parameter model, this requires ~140 GB VRAM and hours of GPU time. For a 70B model, it requires a cluster.

**PEFT (Parameter-Efficient Fine-Tuning)** adapts a model by training only a small number of additional parameters while keeping the original weights frozen. The main PEFT method in production is **LoRA**.

```
Full fine-tuning:  retrain all W  (7B params, ~140 GB VRAM, hours)
LoRA fine-tuning:  freeze W, train ΔW = A·B  (0.1–1% of params, ~16–24 GB VRAM, minutes)
```

---

## 2 · Should You Fine-Tune? — Decision Tree

```
Is the model failing to follow the correct output format?
    └─ YES → Use structured output mode or prompt engineering first.
              Fine-tuning is overkill for format.

Is the model missing domain-specific facts (recent events, private data)?
    └─ YES → Use RAG. Fine-tuning memorises facts poorly and they go stale.

Is the model failing despite correct context in the prompt?
    └─ YES → Is this a reasoning failure or a style/behaviour failure?
              Reasoning failure → better model, CoT prompting, or ReAct
              Style/behaviour failure → fine-tuning is the right call ✓

Is the model correct but too slow or too expensive for production?
    └─ YES → Distillation (fine-tune a smaller model to mimic a larger one) ✓

Is the task so specialised that no amount of prompting helps?
    └─ YES → Fine-tuning ✓
```

### When fine-tuning is worth it

| Use case | Rationale |
|---|---|
| Consistent output style/persona | Style is about weight-level behaviour, not knowledge — prompt engineering can't fully lock it in |
| Legal / medical domain formatting | Very specific structural requirements that prompting only partially meets |
| Low-latency, high-volume production | Fine-tuned smaller model beats prompting a larger model on cost and speed |
| Distillation from GPT-4 to a 7B model | Teach a small model to mimic the larger model's output quality on your specific task |
| Code generation for an internal DSL | Domain-specific language not in the training data; RAG doesn't help with syntax |

---

## 3 · Math — LoRA

**The key insight:** model adaptation doesn't require changing all weights. Most of the task-relevant adaptation projects into a low-rank subspace of the weight matrix.

For a pretrained weight matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$, LoRA represents the update as:

$$\mathbf{W}' = \mathbf{W} + \Delta\mathbf{W} = \mathbf{W} + \mathbf{B}\mathbf{A}$$

where $\mathbf{A} \in \mathbb{R}^{r \times k}$, $\mathbf{B} \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$ is the **rank** — the key hyperparameter.

| Symbol | Meaning |
|---|---|
| $\mathbf{W}$ | Frozen pretrained weight — never updated |
| $\mathbf{A}$ | Trainable low-rank matrix — initialised with Gaussian noise |
| $\mathbf{B}$ | Trainable low-rank matrix — initialised to zeros |
| $r$ | Rank — controls capacity of the adaptation (typical: 4–64) |
| $\alpha$ | Scaling factor: `ΔW` is scaled by `α/r` (typical: 16–32) |

**Initialisation:** $\mathbf{B}$ starts at zero so $\Delta\mathbf{W} = \mathbf{B}\mathbf{A} = 0$ at the start of training. This means LoRA-adapted models behave identically to the base model at initialisation — training starts from a stable point.

**Parameter count reduction:**

```
Original W:  d × k parameters
LoRA:        (d × r) + (r × k) = r × (d + k) parameters

For d=4096, k=4096, r=16:
Original :   16,777,216 params
LoRA     :   16  × (4096 + 4096) = 131,072 params  →  0.78% of original
```

**At inference time:** merge $\mathbf{W}' = \mathbf{W} + \mathbf{B}\mathbf{A}$ back into the original matrix — zero inference overhead compared to the base model.

---

## 4 · QLoRA — Quantisation + LoRA

**QLoRA** combines LoRA with 4-bit quantisation of the frozen base model weights, enabling fine-tuning of large models on a single consumer GPU.

```
Standard LoRA on 7B model:   ~14 GB VRAM (fp16 frozen base + bf16 adapters)
QLoRA on 7B model:           ~6 GB VRAM   (4-bit frozen base + bf16 adapters)
QLoRA on 70B model:          ~48 GB VRAM  (fine-tunable on 2× A100 40GB)
```

The quantisation introduces a small accuracy trade-off compared to full fp16 LoRA, but the quality gap is negligible for most tasks. QLoRA is the standard method for fine-tuning open-source models.

---

## 5 · Step by Step — Fine-Tuning with LoRA

```
1. Choose a base model
   └─ Instruct-tuned (not base) — SFT+RLHF makes fine-tuning more sample-efficient
   └─ Smallest model that can solve the task without fine-tuning at T=0 (measure first)

2. Build a training dataset
   └─ Format: {"prompt": "...", "completion": "..."}  (Alpaca format)
   └─ 500–2000 high-quality examples outperform 10,000 mediocre ones
   └─ Include negative examples (what NOT to output) — dramatically reduces the most common errors
   └─ Hold out 10% for evaluation

3. Set LoRA hyperparameters
   └─ r = 16 (start here; increase to 32–64 if underfitting)
   └─ alpha = 32  (typically 2×r)
   └─ target_modules: ["q_proj", "v_proj"]  (apply LoRA to attention query and value by default)
   └─ dropout = 0.05

4. Train
   └─ Optimiser: AdamW + cosine LR schedule + warmup
   └─ Learning rate: 2e-4 (LoRA adapters); frozen base doesn't update
   └─ Batch size: as large as VRAM allows; gradient accumulation to simulate larger batches
   └─ Epochs: 1–3 (overfitting risk is real with small datasets)
   └─ Monitor: training loss + eval loss + eval task metric

5. Evaluate
   └─ Run EvaluatingAISystems.md metrics on a held-out test set
   └─ Compare against the untuned base model on the same prompts

6. Merge or serve with adapter
   └─ Merge: W' = W + BA → zero inference overhead
   └─ Adapter: serve base model + load adapter at request time → swap adapters per user/tenant
```

---

## 6 · Code Skeleton

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

# ── Load base model (example: Llama 3 8B Instruct) ──────────────────────────
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,          # QLoRA: quantise to 4-bit
    device_map="auto"
)

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # rank
    lora_alpha=32,               # scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,044,937,216 || trainable%: 0.17%

# ── Dataset ───────────────────────────────────────────────────────────────────
# Format: list of {"prompt": str, "completion": str}
train_data = Dataset.from_list([
    {"prompt": "Classify the district as high-value or low-value: MedInc=8.3, AveRooms=6.4...",
     "completion": "high-value"},
    # ... 500+ more examples
])

# ── Training ──────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    fp16=False, bf16=True,      # bf16 for adapter weights with 4-bit base
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    max_seq_length=512,
)
trainer.train()
```

---

## 7 · What Can Go Wrong

- **Catastrophic forgetting.** If the fine-tuning dataset is narrow, the model may lose general capabilities. Mitigation: include a small sample (~5%) of general-purpose examples mixed into the training data ("data mixing").
- **Overfitting to format, not behaviour.** The model learns to produce the right-looking output for training examples but fails to generalise the underlying reasoning. Sign: near-zero training loss but poor eval task metric. Fix: more diverse examples or higher dropout.
- **Dataset contamination.** Training examples that are too similar to eval examples make metrics look better than they are. Deduplicate across train and eval splits.
- **Wrong `target_modules`.** Only applying LoRA to attention layers (`q_proj`, `v_proj`) is standard; for some tasks, applying it to FFN layers (`up_proj`, `down_proj`) significantly helps. Ablate both.
- **Forgetting to test the untuned baseline.** Always compare against the untuned model on the same prompts. Many engineers discover that fine-tuning wasn't necessary after they measure the baseline.

---

## 8 · PizzaBot Connection

> See [AIPrimer.md](../AIPrimer.md) for the full system definition.

The PizzaBot decision tree applied:

| Question from decision tree | PizzaBot answer | Verdict |
|---|---|---|
| Is the model failing to follow the correct output format? | Order confirmations consistently drift from the JSON schema | Prompt engineering first (JSON mode). Fine-tune only if structured output mode fails. |
| Is the model missing domain-specific facts? | Menu changes weekly. New items, seasonal specials, price updates. | **RAG, not fine-tuning.** Retraining weekly is impractical. RAG re-index takes minutes. |
| Is this a style/behaviour failure? | The bot occasionally sounds terse; Mamma Rosa's brand needs warmth. | **Fine-tune candidate.** Tone is weight-level behaviour. RAG can't fix it. |
| Is the model correct but too slow/expensive? | 10k orders/day at GPT-4o prices → ~$150/day in agent calls alone. | **Distillation candidate.** Fine-tune a 7B model on GPT-4o traces for the order-placement path. |

**Practical split:** use RAG for all factual content (menu, allergens, locations), fine-tune for the conversational persona layer.

---

## 9 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The LoRA decomposition: W' = W + BA and why B is initialised to zero | When would you choose fine-tuning over RAG? | Saying fine-tuning teaches the model new facts — it teaches new behaviour; RAG handles new facts |
| What rank `r` controls and its effect on parameter count | What is QLoRA and what does quantisation trade off? | Confusing full fine-tuning with LoRA — they have completely different VRAM requirements |
| The decision tree: when fine-tuning beats RAG and when it doesn't | How do you prevent catastrophic forgetting during fine-tuning? | Saying fine-tuning is too complicated for a production team — QLoRA on a single A100 is now routine |
| Target modules: why q_proj and v_proj are the default | What is the difference between LoRA and adapter methods? | Confusing inference overhead — merged LoRA has zero; unmerged adapters add a small forward pass |
| **DPO vs SFT:** SFT trains on (prompt, chosen\_response) pairs — teaches format and style but provides no signal about what to avoid. DPO trains on (prompt, chosen, rejected) triples — directly optimises to increase log-probability of chosen relative to rejected, without a separate reward model or RL loop. More stable than RLHF | "What is Direct Preference Optimisation and how does it differ from RLHF?" | "DPO replaces pretraining" — DPO only adjusts relative preference between responses the model already generates reasonably; SFT first, then DPO is the standard recipe |
| **Data requirements for fine-tuning:** quality matters more than quantity. LoRA fine-tuning for a specific task can work with 50–500 carefully curated examples; quality filtering (dedup, length filter, human review) consistently outperforms naive scaling of low-quality data | "How much data do you need to fine-tune an LLM with LoRA?" | "More fine-tuning data always helps" — noisy or contradictory training examples cause catastrophic forgetting or performance degradation; quality first, quantity second |

---

## 10 · Bridge

Fine-Tuning showed how to adapt model behaviour when prompting and RAG aren't enough. `SafetyAndHallucination.md` covers the reliability risks that all three approaches — prompting, RAG, and fine-tuning — must mitigate before a system can be trusted in production.

> *Fine-tuning changes the distribution of outputs. Evaluation (EvaluatingAISystems.md) is what tells you if the new distribution is actually better.*
