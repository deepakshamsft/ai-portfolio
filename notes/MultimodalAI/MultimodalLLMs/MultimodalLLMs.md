# Multimodal LLMs — When Language Models Can See

> After reading this you will understand how a vision encoder is connected to a language model, what the Q-Former and projection layer do, and how LLaVA, BLIP-2, and GPT-4V differ architecturally.

## 1 · Core Idea

A Multimodal Large Language Model (MLLM) extends a text LLM to accept image (and optionally audio/video) tokens alongside text tokens. The strategy is almost always:

1. **Vision encoder** (ViT, typically frozen): image → sequence of visual tokens
2. **Projection / adapter**: visual token dimension → LLM token dimension
3. **LLM decoder** (GPT/LLaMA): generates text autoregressively, attending over both visual and text tokens

The hard part is bridging the **modality gap**: a ViT token looks nothing like a word embedding, so naively concatenating them doesn't work. The two main solutions are:
- **Linear projection** (LLaVA): a single learnable matrix aligns dimensions
- **Q-Former** (BLIP-2): a transformer module with learnable query tokens that compresses visual features into a fixed number of tokens before injection into the LLM

## 2 · Running Example

PixelSmith v6: given a photo of a digit + a question ("What digit is this?"), answer in natural language.

The notebook implements a mini MLLM: a pretrained ViT (from torchvision) → linear projection → a tiny GPT-style decoder trained to answer simple questions about MNIST images.

## 3 · The Math

### Vision Encoder Output

A ViT-B/16 processes a 224×224 image into $N = 196$ patch tokens + 1 CLS token = 197 tokens, each of dimension $d_v = 768$.

$$\mathbf{V} = \text{ViT}(x) \in \mathbb{R}^{197 \times 768}$$

### Linear Projection (LLaVA-style)

Map visual tokens to LLM embedding dimension $d_L$ (e.g., 4096 for LLaMA-7B):

$$\mathbf{V}' = \mathbf{V} \mathbf{W}_p + \mathbf{b}_p, \quad \mathbf{W}_p \in \mathbb{R}^{768 \times 4096}$$

LLaVA-1.5 uses a two-layer MLP with the same structure. The projected tokens are then concatenated with text token embeddings and fed into the LLM:

$$\text{input tokens} = [\mathbf{V}'_1, \ldots, \mathbf{V}'_{197}, t_1, \ldots, t_L]$$

where $t_i$ are text token embeddings.

### Q-Former (BLIP-2)

The Q-Former (Querying Transformer) uses $N_q = 32$ **learnable query tokens** $\mathbf{Q} \in \mathbb{R}^{32 \times d_q}$ that attend over the 197 visual tokens:

$$\text{Q-Former output} = \text{CrossAttn}(\mathbf{Q},\, \mathbf{V}) \in \mathbb{R}^{32 \times d_q}$$

The 32 output tokens (rather than 197) are projected to the LLM dimension. This achieves:
- **Compression**: 197 visual tokens → 32 tokens (6× fewer tokens for the LLM to process)
- **Filtering**: the query tokens learn to extract task-relevant visual features only

### Visual Instruction Tuning (LLaVA)

LLaVA trains on (image, instruction, response) triplets using a standard next-token prediction loss:

$$\mathcal{L} = -\sum_{t} \log p_\theta(r_t \mid r_{<t}, \mathbf{V}', \mathbf{q})$$

where $r_t$ is the response token, $\mathbf{V}'$ is the projected visual embedding, and $\mathbf{q}$ is the instruction. Only the projection layer and the LLM are trained; the ViT is frozen.

### Interleaved Text and Images (Flamingo-style)

For interleaved documents (text...image...text...image...):

```
[text tokens] [visual tokens] [text tokens] [visual tokens] [text tokens...]
```

Flamingo (DeepMind, 2022) uses "gated cross-attention" dense layers inserted every $k$ LLM layers to inject visual features. The gating prevents early training instability.

## 4 · How It Works — Step by Step

### LLaVA-1.5 Inference

1. Image → CLIP ViT-L/14 → 576 visual tokens (each 1024-dim)
2. 576 × 1024 → two-layer MLP → 576 × 4096 (LLaMA-2 7B dimension)
3. Tokenise instruction: "Describe this image" → token IDs → LLaMA-2 embedding layer → `[N_text × 4096]`
4. Concatenate: `[576 visual tokens | N_text text tokens]`
5. LLaMA-2 decoder autoregressively generates response tokens
6. Each response token can attend to all 576 visual tokens + all previous tokens

### BLIP-2 Inference

1. Image → ViT-g/14 (1.2B params, frozen) → 256 visual tokens
2. Q-Former (12 layers, 32 query tokens) → 32 compressed visual tokens
3. Linear projection → 32 × 4096 (LLM dimension)
4. FlanT5-XL or OPT-6.7B receives the 32 visual tokens as a soft prompt prefix
5. Decode response

Why fewer tokens? The LLM's KV-cache memory scales quadratically with sequence length. 32 tokens instead of 256 saves 64× memory at the visual prefix.

## 5 · The Key Diagrams

```
LLaVA Architecture:

  224×224 image
      │
  [ViT-L/14, FROZEN]
      │
  576 × 1024 visual tokens
      │
  [Linear Projection / 2-layer MLP, TRAINED]
      │
  576 × 4096 visual tokens (in LLaMA-2 embed space)
      │         ┌──────────────────────────┐
      └────────▶│  LLaMA-2 7B (TRAINED)   │◀──── "Question: What's in the photo?"
                │  [32 transformer layers] │
                └──────────────────────────┘
                              │
                     "A cat sitting on..."


BLIP-2 Architecture:

  image → [ViT-g, FROZEN] → 256 tokens
                               │
                          [Q-Former, TRAINED]  ← 32 learnable query tokens
                               │
                           32 × d_q tokens (compressed)
                               │
                          [Linear, TRAINED]
                               │
                         32 × d_LLM tokens
                               │
                     [FlanT5 / OPT, FROZEN or TRAINED]
                               │
                          text response
```

## 6 · What Changes at Scale

| Model | Vision encoder | Connector | LLM backbone | Open? |
|-------|---------------|-----------|-------------|-------|
| LLaVA-1.0 | CLIP ViT-L/14 | Linear | Vicuna-7B/13B | Yes |
| LLaVA-1.5 | CLIP ViT-L/14-336 | 2-layer MLP | LLaMA-2 7B/13B | Yes |
| BLIP-2 | ViT-g/14 | Q-Former | FlanT5-XL/OPT-6.7B | Yes |
| InstructBLIP | ViT-g/14 | Q-Former + instruct | FlanT5/Vicuna | Yes |
| GPT-4V | Undisclosed | Undisclosed | GPT-4 | No |
| Gemini 1.5 Pro | Undisclosed | Undisclosed | Gemini | No |
| Llama-3.2 Vision | CLIP-ViT | Cross-attention | LLaMA-3.2 | Yes |
| Qwen2-VL | NaViT (variable res) | MLP | Qwen2-7B | Yes |

Key trend: higher resolution vision encoders (336px → 448px → variable), larger LLMs, and more diverse training data (charts, OCR, medical images).

## 7 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "MLLMs understand spatial relationships perfectly" | Spatial reasoning (left/right, counting) is still a known weakness; CoT prompting helps |
| "Freezing the ViT is optimal" | LLaVA-1.5 and newer models often fine-tune the ViT end-to-end for higher accuracy |
| "Q-Former always outperforms linear projection" | LLaVA-1.5 with a simple MLP outperformed many Q-Former models at 7B scale; simplicity can win |
| "More visual tokens = always better" | Longer context → slower generation, higher memory; there's a quality/cost trade-off |
| "Multimodal models can read small text in images" | OCR capability varies widely; models trained on document datasets (DocVQA) are much better at this |

## 8 · Interview Checklist

### Must Know
- General MLLM recipe: vision encoder → alignment layer → LLM
- Difference between LLaVA (linear projection, 576 tokens) and BLIP-2 (Q-Former, 32 tokens)
- Visual instruction tuning: freeze ViT, train projection + LLM on (image, instruction, answer) triples

### Likely Asked
- *"How would you add vision to LLaMA-3?"* — Attach a CLIP or SigLIP ViT, project visual tokens to LLaMA's embed dimension with an MLP, fine-tune on instruction-following visual QA data
- *"What is the Q-Former and when would you use it?"* — A cross-attention transformer that compresses many visual tokens into few learnable query outputs; use when the LLM has short context limits or when visual compression is needed
- *"Why freeze the ViT during initial training?"* — Prevents catastrophic interference; the ViT's features are already strong; frozen ViT lets you focus the compute budget on learning the alignment

### Trap to Avoid
- Don't say MLLMs "see" images the way humans do — they process a sequence of numerical patch embeddings; their spatial understanding is learned from training data, not built-in.

## 9 · What's Next

[GenerativeEvaluation.md](../GenerativeEvaluation/GenerativeEvaluation.md) — how do you measure the quality of generated images and video? FID, IS, CLIP Score, and human preference models.
