# CLIP — Contrastive Language-Image Pretraining

> **After reading this note** you will understand how CLIP learns to align image and text embeddings without any manual labels, what the InfoNCE contrastive loss is doing geometrically, how zero-shot classification works without a training set, and why CLIP's text encoder is the component inside Stable Diffusion that converts your prompt into a conditioning signal.

---

## 1 · Core Idea

**CLIP** (Contrastive Language-Image Pretraining, OpenAI 2021) trains two encoders — a ViT image encoder and a transformer text encoder — jointly on 400 million image-text pairs scraped from the internet. The training objective is beautifully simple: **make the embedding of each image similar to the embedding of its paired caption, and dissimilar to the embeddings of all other captions in the same batch**. No class labels. No manual annotations. Just the signal that a photograph of a dog and the text "a photo of a dog" belong together.

The result is a shared embedding space where you can directly compare images and text using cosine similarity. This single capability unlocks: zero-shot image classification, semantic image search, and — most importantly for this track — the text conditioning mechanism inside every latent diffusion model.

---

## 2 · Running Example — PixelSmith v2

```
Goal:    Build a text-image search index: given a text query, return the most 
         semantically matching image from a collection.

Process: Encode each image → 512-dim embedding (stored in index)
         For a query, encode the text → 512-dim embedding
         Rank images by cosine similarity to the text embedding

Before CLIP (Ch.1+2): we have image embeddings but no text embeddings
After CLIP (this chapter): image and text live in the SAME space → direct comparison
```

---

## 3 · The Math

### 3.1 Dual Encoder Architecture

CLIP has two separate encoders that share no weights:

$$\mathbf{v}_i = \text{ImageEncoder}(I_i) / \|\text{ImageEncoder}(I_i)\|_2 \quad \in \mathbb{R}^d$$
$$\mathbf{t}_i = \text{TextEncoder}(T_i) / \|\text{TextEncoder}(T_i)\|_2 \quad \in \mathbb{R}^d$$

Both outputs are **$\ell_2$-normalised** so that cosine similarity reduces to dot product:

$$\text{cosine\_sim}(\mathbf{v}_i, \mathbf{t}_j) = \mathbf{v}_i \cdot \mathbf{t}_j$$

The similarity matrix over a batch of $N$ pairs forms an $N \times N$ matrix $S$ where $S_{ij} = \mathbf{v}_i \cdot \mathbf{t}_j$.

### 3.2 InfoNCE Contrastive Loss

CLIP uses the **InfoNCE** (Noise Contrastive Estimation) loss, scaled by a learnable temperature $\tau$:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ij}/\tau)} + \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ji}/\tau)} \right]$$

**Decomposing this:**
- First term: for image $i$, find its caption among all $N$ captions (image → text direction)
- Second term: for text $i$, find its image among all $N$ images (text → image direction)
- The diagonal $S_{ii}$ entries are the matching pairs; off-diagonal are negatives

**Geometrically:** InfoNCE loss pushes matching (image, text) pairs close together and pushes all $N-1$ non-matching pairs apart. With $N = 32{,}768$ (OpenAI's CLIP batch size), each sample has 32,767 negatives — very hard negatives that force the model to learn fine-grained distinctions.

### 3.3 Temperature Scaling

The temperature $\tau$ is a learnable scalar initialised to $\log(1/0.07) \approx 2.66$:

$$\tau^{-1} \approx 14.3 \text{ (effective)}$$

**Effect:** low $\tau$ sharpens the softmax — the model must be more confident about the correct match. CLIP learns $\tau \approx 0.01$–$0.1$ — a very sharp distribution.

### 3.4 Zero-Shot Classification

Given $K$ class names $\{c_1, \ldots, c_K\}$, construct text prompts: "a photo of a {class}". Encode each → $\mathbf{t}_k$. For a new image $I$, the predicted class is:

$$\hat{y} = \arg\max_k \; \mathbf{v} \cdot \mathbf{t}_k$$

No gradient updates to CLIP. No labelled training data for the new task. The model transfers because it learned general visual-semantic alignment, not dataset-specific patterns.

---

## 4 · How It Works — Step by Step

**Step 1: Batch assembly**
```
Sample N image-text pairs from the 400M dataset
Each pair: (image of ocean sunset, "golden hour at the beach")
```

**Step 2: Encode**
```
Image encoder (ViT-B/32): image → 768-dim → linear projection → 512-dim → L2 norm
Text encoder (GPT-like transformer): tokens → 512-dim (from [EOS] token) → L2 norm
```

**Step 3: Similarity matrix**
```
S = V @ T.T   shape (N, N)
S[i,j] = cosine similarity between image i and text j
S[i,i] = matching pair (positive)
S[i,j≠i] = non-matching pair (negative)
```

**Step 4: Symmetric cross-entropy**
```
Row-wise softmax of S → loss for each image finding its text
Column-wise softmax of S → loss for each text finding its image
Average both losses
```

**Step 5: Temperature scaling**
```
S_scaled = S / τ
Lower τ → sharper distribution → harder to distinguish → higher loss gradient
```

**Step 6: Backprop through both encoders**
```
Gradients flow back into both ViT and text transformer
Both learn to embed matching pairs close, non-matching pairs far
```

---

## 5 · The Key Diagrams

### CLIP Architecture

```
  TEXT INPUT                          IMAGE INPUT
  "a photo of a cat"                  [JPEG of a cat]
        │                                   │
  ┌─────▼──────────────┐          ┌────────▼────────────┐
  │  Text Transformer   │          │  ViT Image Encoder  │
  │  (GPT-like, 12L)    │          │  (ViT-B/32 or L/14) │
  │                     │          │                     │
  │  [EOS] hidden state │          │  [CLS] hidden state │
  │  → linear → 512-dim │          │  → linear → 512-dim │
  └─────────┬──────────┘          └──────────┬──────────┘
            │   L2 norm                        │   L2 norm
            ▼                                  ▼
     text embedding t              image embedding v
       (512-dim unit vector)         (512-dim unit vector)
            │                                  │
            └──────────────┬───────────────────┘
                           │
               cosine similarity = v · t
               (same space → directly comparable)
```

### InfoNCE Loss — Batch Similarity Matrix

```
                      Text embeddings
           t₁    t₂    t₃    t₄   ...  tₙ
         ┌────────────────────────────────┐
  v₁     │ 0.92  0.11  0.08  0.05  ...│  ← image 1 should match t₁
  v₂     │ 0.09  0.88  0.12  0.07  ...│  ← image 2 should match t₂
  v₃     │ 0.06  0.13  0.91  0.04  ...│  ← image 3 should match t₃
  v₄     │ 0.04  0.08  0.05  0.89  ...│  ← image 4 should match t₄
  ...    │ ...                        │
         └────────────────────────────┘
         
  Goal: push diagonal entries → 1.0
        push off-diagonal entries → 0.0
  
  Cross-entropy loss treats each row as a classification problem
  (N-way classification with one correct answer per row)
```

---

## 6 · What Changes at Scale

| Factor | OpenAI CLIP (2021) | State of the art (2024) |
|--------|-------------------|------------------------|
| Training data | 400M image-text pairs (WIT) | LAION-5B (5 billion pairs) |
| Batch size | 32,768 | 65,536–262,144 |
| Image encoder | ViT-B/32, ViT-L/14 | ViT-G/14, ViT-bigG |
| Embedding dim | 512–768 | 1024–1280 |
| Text encoder | GPT-style, 63M params | GPT-style, 124M–340M |
| Zero-shot ImageNet top-1 | 76.2% (ViT-L/14@336) | ~80%+ (EVA-CLIP) |

**Hard negatives matter:** CLIP's quality scales with batch size because larger batches provide more hard negatives. A batch of 32K gives 32,767 negatives per sample. Methods like ARCA and CoCa further improve by mining hard negatives from the full dataset.

**The text encoder in Stable Diffusion:** SD v1-v2 uses CLIP's text encoder (frozen) to convert prompts to 77 × 768 cross-attention conditioning tokens. SD v2 switches to OpenCLIP ViT-H/14. SDXL uses two text encoders: OpenCLIP ViT-bigG (1280-dim) + CLIP ViT-L (768-dim) concatenated → 2048-dim conditioning.

---

## 7 · Common Misconceptions

**"CLIP is a generative model"**
CLIP is a discriminative/contrastive model — it scores similarity but does not generate images or text. Generation is handled by diffusion models in later chapters.

**"CLIP understands spatial relationships"**
CLIP struggles with spatial and compositional reasoning. "A red circle above a blue square" and "A blue square above a red circle" often produce very similar embeddings because CLIP's pretraining emphasises object identity over spatial layout. This is why ControlNet (Ch.8) exists.

**"Zero-shot means CLIP was not trained on ImageNet"**
CLIP was not trained *with ImageNet labels*, but it was trained on internet data that certainly contains ImageNet images. The claim is that classification is performed *without fine-tuning on labelled ImageNet data*, not that the model is truly naive about those images.

**"The similarity score is a probability"**
After L2 normalisation, the dot product gives cosine similarity in $[-1, 1]$. It is not a probability. To get probabilities you must apply softmax across a set of candidate texts.

---

## 8 · Interview Checklist

### Must Know
- What are the two encoders in CLIP and what do they output?
- What is the InfoNCE loss — what are the positives and negatives?
- How does zero-shot classification work with CLIP?

### Likely Asked
- "Why does CLIP use large batch sizes?"
  → More negatives per sample → harder negatives → sharper representations
- "How is CLIP's text encoder used in Stable Diffusion?"
  → Frozen CLIP text encoder converts prompt to 77 × 768 token embeddings → fed as cross-attention keys/values inside the U-Net denoiser at every layer
- "What does CLIP's embedding space geometry look like?"
  → All embeddings are on the unit hypersphere (`L2 norm = 1`); matching pairs are close (high cosine sim); unrelated pairs are near-orthogonal

### Trap to Avoid
- Confusing CLIP (contrastive training, no generation) with DALL-E (generative, uses CLIP as a component)
- Saying CLIP fine-tunes on task data — zero-shot means no fine-tuning
- Forgetting temperature: without the learned $\tau$ the gradients are too weak for early training
- **SigLIP vs CLIP:** SigLIP replaces the softmax-normalised InfoNCE loss with per-pair sigmoid binary cross-entropy — no normalisation over the batch; each (text, image) pair is simply scored as matching or not. Works well at smaller batch sizes and removes hard-negative dependency. Trap: "SigLIP makes batch size irrelevant" — smaller batches still mean fewer negatives per sample; SigLIP is less sensitive but not batch-size-independent
- **CLIP for retrieval:** embed query with the appropriate encoder and perform cosine-similarity search over a pre-indexed CLIP embedding corpus; supports image-to-image, text-to-image, and cross-modal retrieval from the same index. Trap: "CLIP embeddings can be compared with raw dot product" — CLIP embeddings are L2-normalised; use cosine similarity (equivalently, dot product on unit sphere); unnormalised dot product gives wrong rankings if vectors are not on the unit sphere

---

## 9 · What's Next

→ **[DiffusionModels.md](../DiffusionModels/DiffusionModels.md)** — CLIP gives PixelSmith a shared image-text space, enabling semantic search (v2). The PixelSmith v3 upgrade requires generation: given noise, produce a plausible image. Diffusion models provide this. You need to understand the forward noising process, the reverse denoising process, and the U-Net architecture before you can understand how CLIP's text encoder slots into Stable Diffusion as a conditioning signal.

## Illustrations

![CLIP - dual encoders, contrastive matrix, zero-shot, shared space](img/CLIP.png)
