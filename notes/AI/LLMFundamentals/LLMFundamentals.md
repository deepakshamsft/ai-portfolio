# LLM Fundamentals — What a Language Model Actually Is

> **Read this before anything else in the AI track.** CoTReasoning, RAG, ReAct, and every agent framework in these notes assume you know what an LLM is under the hood. This document builds that foundation from the transformer (Ch.17) through to the models you call via API today.

---

## 1 · Core Idea

A **large language model** is a transformer decoder (Ch.17) trained to predict the next token given all previous tokens, on internet-scale text. That single objective — next-token prediction — produces a model that appears to reason, retrieve facts, write code, and generate plans. None of those behaviours were explicitly programmed. They emerge from scale.

```
Training objective:   maximise P(token_t | token_1, token_2, ..., token_{t-1})
Training data:        ~10–100 trillion tokens scraped from the web, books, code
Training compute:     10²³–10²⁵ FLOP  (millions of GPU-hours)
Result:               a model with 7B–1T parameters that can perform most language tasks
```

Three stages turn a raw next-token predictor into the assistant you actually use:

```
Stage 1: Pretraining        Raw transformer on internet text → learns language + world knowledge
Stage 2: SFT                Fine-tuned on (instruction, good response) pairs → follows instructions
Stage 3: RLHF / DPO         Aligned with human preferences → helpful, harmless, honest
```

Each stage is covered in detail below.

---

## 2 · Tokenisation

The model never sees raw text. Text is first broken into **tokens** — subword units — using a byte-pair encoding (BPE) vocabulary.

### How BPE Works

```
Start with character-level vocabulary: [a, b, c, ..., z, space, ...]

1. Count all adjacent character pairs in the training corpus
2. Merge the most frequent pair into a new token: "t" + "h" → "th"
3. Repeat until vocabulary reaches target size (32k–100k tokens)
```

**Result:** common words become single tokens (`the`, `model`, `training`). Rare or technical words split (`trans` + `former`, `to` + `ken` + `isation`). Code tokens are often single characters.

### What You Need to Know About Tokens

| Fact | Why it matters |
|---|---|
| ~1 token ≈ 0.75 English words | Convert words → tokens for cost estimation |
| One token ≈ 4 bytes | 1M tokens ≈ 4 MB of text |
| The same text tokenises differently across models | Never assume GPT-4's token count matches Claude's |
| Code is token-dense | `self.attention_weights[layer_idx]` may be 6–10 tokens |
| Numbers tokenise byte-by-byte | `12345` → `[123, 45]` in some vocabularies — arithmetic is hard |

### The Context Window

The context window is the maximum number of **tokens** the model can process in a single forward pass — both input (prompt + retrieved chunks + history) and output (generated tokens).

| Model class | Context window |
|---|---|
| GPT-3.5 (2022) | 4k tokens |
| GPT-4 (2023) | 8k / 32k |
| Claude / Gemini (2024) | 200k / 1M |
| LLaMA 3 (2024) | 128k |

Larger context windows do not mean unlimited memory. Empirically, models show **lost-in-the-middle** degradation: information at the beginning and end of a long context is recalled more reliably than information buried in the middle.

---

## 3 · Sampling — Temperature, Top-p, Top-k

The model outputs a probability distribution over the vocabulary at each step. **Sampling parameters** control how you select the next token from that distribution.

### Temperature

$$p'_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Temperature $T$ | Effect |
|---|---|
| $T → 0$ | Deterministic: always pick the highest-probability token (greedy) |
| $T = 1$ | Sample from the unmodified distribution |
| $T > 1$ | Distribution flattens — more randomness, less coherent |

**Rule of thumb:** factual retrieval → low T (0.0–0.3); creative generation → higher T (0.7–1.0); code → 0.0–0.2.

### Top-p (Nucleus Sampling)

Instead of sampling from all tokens, select from the smallest set of tokens whose cumulative probability exceeds $p$:

```
Sort tokens by probability descending: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, ...]
top_p = 0.9 → keep [0.40, 0.25, 0.15, 0.10] (cumsum = 0.90) → sample only these four
```

Top-p dynamically adjusts the candidate set per token — large when the distribution is flat (uncertain), small when one token dominates (confident). Almost all production usage combines temperature + top-p.

### Top-k

Keep only the k highest-probability tokens and renormalise. Less adaptive than top-p; rarely preferred in practice.

---

## 4 · The Three Training Stages

### Stage 1 — Pretraining

A standard transformer decoder (Ch.17 causal mask) is trained on a massive corpus with the cross-entropy loss over next-token prediction. No human labels — the text itself is the supervision.

**What it learns:** grammar, syntax, world knowledge, reasoning patterns, code idioms, basic arithmetic, multilingual text — anything that appears frequently enough in the training data.

**What it doesn't learn:** to be helpful, to follow instructions, or to prefer honest over fluent answers.

A pretrained model responds to `"What is the capital of France?"` by continuing the text in a plausible direction — which might be `"?"` or `"A: Paris"` or `"Who is the king of France?"` depending on what it has seen. It does not reliably answer the question.

### Stage 2 — Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model on a curated dataset of `(instruction, response)` pairs written by human annotators.

```
Input:   "Summarise this document in three bullet points: [doc]"
Target:  "• Point 1\n• Point 2\n• Point 3"
Loss:    Cross-entropy on the target tokens only (not the input)
```

SFT teaches the model to follow instruction format and stay on task. Even a few thousand high-quality examples significantly improves instruction-following.

**The risk:** the model learns what annotators wrote, not what is correct. If annotators tend to produce verbose, confident answers, the model does too.

### Stage 3 — RLHF / DPO (Alignment)

The goal: move the model's outputs toward what humans actually prefer — more helpful, less harmful, more honest.

**RLHF (Reinforcement Learning from Human Feedback):**

```
1. Sample two completions for the same prompt
2. Human annotator picks the preferred one
3. Train a reward model R(prompt, completion) on these preference pairs
4. Fine-tune the SFT model to maximise R using PPO (policy gradient RL)
   + KL penalty to stay close to the original SFT model
```

**DPO (Direct Preference Optimization):** skips the reward model entirely. Directly fine-tunes the model on preference pairs with a loss that increases the probability of the preferred response and decreases the probability of the rejected one. Simpler, more stable, now preferred over RLHF in most open-source work.

**What RLHF/DPO gives you:** a model that says "I don't know" when it doesn't know, declines harmful requests, and structures answers for human convenience rather than for statistical fluency.

**The sycophancy trap:** RLHF optimises for human *approval*, which is not the same as human *benefit*. Models learn to agree with the user's framing even when it's wrong. This is why you can sometimes "convince" a model to change a correct answer by pushing back.

---

## 5 · Emergent Capabilities

Several capabilities of LLMs were not explicitly trained for and appeared qualitatively at sufficient scale:

| Capability | Approximate threshold |
|---|---|
| In-context learning (few-shot) | ~7B parameters |
| Chain-of-thought reasoning | ~100B parameters |
| Multi-step arithmetic | ~540B parameters |
| Theory of mind (passing Sally-Anne test) | GPT-4 class |

**"Emergent"** does not mean magical. These capabilities exist in the training data — it's that the model needs sufficient capacity to compress and reconstruct the reasoning patterns latent there.

---

## 6 · What "Model Size" Actually Means

```
Parameters = weights in all attention and FFN matrices
           = num_layers × (12 × d_model²)   for a standard transformer

7B model:   7 × 10⁹ parameters × 2 bytes (fp16) = 14 GB VRAM minimum
13B model:  ~26 GB
70B model:  ~140 GB  (requires 2× A100 80GB)
GPT-4:      estimated 1.8T parameters in a mixture-of-experts architecture
```

**Inference cost scales with parameter count, context length, and batch size.** A 70B model at 128k context costs roughly 50× more to run than a 7B model at 4k context. This is why RAG and agentic applications use smaller, instruction-tuned models wherever possible.

---

## 7 · Key Distinctions Every Engineer Gets Asked

| Pair | Distinction |
|---|---|
| **Base model vs instruct/chat model** | Base: raw next-token predictor. Instruct: SFT+RLHF applied — follows instructions. Always use instruct for applications. |
| **Parameters vs context window** | Parameters = learned knowledge. Context window = working memory for one inference call. |
| **Temperature vs top-p** | Temperature rescales the whole distribution. Top-p truncates it. Use both. |
| **RLHF vs DPO** | RLHF trains a separate reward model; DPO doesn't. DPO is simpler and now standard. |
| **Tokens vs words** | Tokens are model-native; words are human-native. 1 word ≈ 1.3 tokens on average for English prose. |
| **Hallucination vs confabulation** | Hallucination: factually wrong output. Confabulation: specifically, a fluent-sounding fabrication of a plausible but non-existent fact (citation, statistic, API name). Same mechanism, different vocabulary. |

---

## 8 · PizzaBot Connection

> See [AIPrimer.md](../AIPrimer.md) for the full system definition.

| Concept | Where it shows up in PizzaBot |
|---|---|
| **Temperature** | `temperature=0` for the JSON order confirmation (no hallucinated prices). `temperature=0.8` for the "surprise me with a pizza" recommendation path. |
| **BPE tokenisation** | `"pepperoni"` → 3 tokens; `"gluten-free"` → 3 tokens. Matters when estimating the cost of embedding the menu corpus and when counting context window usage per turn. |
| **Context window** | A long SMS session (20+ back-and-forth turns) plus 5 retrieved chunks plus the system prompt approaches 8k tokens. This is why the agent must summarise older turns rather than pass the full history. |
| **Lost-in-the-middle** | If the allergen chunk is injected in the middle of a long context, the model may miss the gluten flag. Keep critical safety facts (allergens) near the system prompt or in a separate grounding block. |
| **RLHF sycophancy** | User: "I was told the Margherita is £8 today." Model (RLHF-aligned) may agree even if the RAG corpus says £13.99. Mitigation: grounding constraint + fact-check against retrieved price. |

---

## 9 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The three training stages (pretraining → SFT → RLHF/DPO) and what each adds | What is DPO and why is it better than RLHF? | Saying RLHF is "just reinforcement learning" — the reward model and KL penalty are the critical additions |
| Temperature: effect of T<1 vs T>1 | What is top-p and why is it preferred over top-k? | Saying temperature = "creativity dial" — technically it's distribution sharpening/flattening |
| What a context window is and what lost-in-the-middle means | Why can't LLMs do reliable multi-digit arithmetic? | Confusing model parameters (knowledge) with context window (working memory) |
| What BPE tokenisation produces and why it matters for cost | Why do larger models exhibit emergent capabilities? | Saying the model "understands" or "thinks" — anthropomorphic framing fails in interviews |
| Base model vs instruct model — what SFT adds | What is sycophancy and why does RLHF cause it? | Saying fine-tuning changes what the model "knows" — SFT/RLHF changes behaviour, not stored knowledge |
| **PEFT overview:** Parameter-Efficient Fine-Tuning methods add or modify a small number of parameters while freezing the base model. **LoRA** inserts low-rank matrices into attention projections (~0.1% of parameters, merges at inference for zero latency); **prefix tuning** prepends learnable tokens to every layer's KV sequence (increases KV cache size at inference); **prompt tuning** learns only input embeddings (smallest footprint) | "Compare LoRA, prefix tuning, and prompt tuning" | "PEFT methods are interchangeable" — LoRA modifies weights and merges at inference with no latency cost; prefix tuning increases the effective sequence length and has a constant KV cache overhead at inference; choosing between them depends on deployment constraints |
| **Instruction following vs base model:** a base model generates the statistically likely continuation of any text — useful for perplexity benchmarks, not for assistants. SFT on instruction-response pairs shifts the output distribution toward helpful, formatted answers | "What does instruction fine-tuning actually teach the model?" | "Fine-tuning on instructions teaches the model new knowledge" — SFT teaches *format and tone*, not new facts; knowledge comes from pretraining data; new facts require RAG or continual pretraining |

---

## 10 · Bridge

LLM Fundamentals established the model: a scaled, aligned next-token predictor with a finite context window and probabilistic sampling. The next document — `CoTReasoning.md` — shows how you exploit that predictor to produce step-by-step reasoning chains, and how those chains become the planning substrate for an agentic loop.

> *The model is the brain. It predicts tokens. Everything in the AI track — CoT, RAG, ReAct, Semantic Kernel — is about how you wire inputs and outputs around that single mechanical act.*

## Illustrations

![LLM fundamentals — BPE tokenisation, sampling, training stages, and the context window](img/LLM%20Fundamentals.png)

## Illustrations

![LLM fundamentals — BPE tokenisation, sampling, training stages, and the context window](img/LLM%20Fundamentals.png)
