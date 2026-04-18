# Vision Transformers — How Images Become Sequences

> **The story.** For most of deep-learning history, vision meant CNNs — LeCun's LeNet (1989) through AlexNet (2012) through ResNet (2015). The transformer ([ML Ch.18](../../ML/ch18-transformers/)) was for text. In **October 2020** **Alexey Dosovitskiy** and colleagues at Google Brain published *"An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"* — the **Vision Transformer (ViT)** — with one audacious move: split the image into 16×16 patches, treat each as a token, throw it into a vanilla transformer encoder, and skip convolutions entirely. With enough data (JFT-300M), ViT matched and then beat the best CNNs on ImageNet. **Swin Transformer** (Microsoft, 2021) added hierarchical windows; **DINO** (Caron et al., Meta, 2021) showed self-supervised ViTs learn surprisingly clean object segmentations; and ViT became the visual backbone of **CLIP**, **Stable Diffusion**, **DINOv2**, **SAM**, and every modern multimodal LLM.
>
> **Where you are in the curriculum.** [MultimodalFoundations](../MultimodalFoundations/) showed why every modality needs to become tokens. This chapter shows how *images* become tokens. After this you understand the visual half of the [CLIP](../CLIP/) embedding space, the encoder inside [Latent Diffusion](../LatentDiffusion/), and the vision tower of every [Multimodal LLM](../MultimodalLLMs/) in the track.

---

## 1 · Core Idea

A standard transformer expects a sequence of token embeddings as input. An image is a 2-D spatial grid of pixels, not a sequence. The Vision Transformer (ViT) resolves this mismatch with one elegant trick: **split the image into fixed-size patches and treat each patch as a token**. A 224×224 image divided into 16×16 patches yields 196 tokens. Each patch is flattened and linearly projected into a $d$-dimensional embedding vector, then processed by an ordinary transformer encoder with no convolutions at all.

This design has two major consequences: 
1. ViT can process images with the same architecture that processes text — enabling true multimodal models. 
2. At scale (large datasets + large models), ViT outperforms CNNs because attention can model long-range spatial dependencies that convolution's fixed receptive field cannot.

---

## 2 · Running Example — PixelSmith v1

```
Input: (3, 224, 224) normalised image tensor (built in Ch.1)
Process: Split into 196 patches of size 16×16
 → each patch: 16 × 16 × 3 = 768 values
 → linear projection: 768 → 768 (the hidden dimension dmodel = 768 for ViT-B)
Output: 196 patch embeddings of shape (768,) + 1 [CLS] token
 → full sequence shape (197, 768) ready for transformer encoder
```

---

## 3 · The Math

### 3.1 Patch Embedding

Divide the image $I \in \mathbb{R}^{3 \times H \times W}$ into $N$ non-overlapping patches:

$$N = \frac{H \times W}{P^2}$$

where $P$ is the patch size (typically 16 or 32 pixels). Each patch $p_i \in \mathbb{R}^{3 P^2}$ is flattened and linearly projected:

$$\mathbf{z}_i = \mathbf{E} \cdot p_i + \mathbf{b}$$

where $\mathbf{E} \in \mathbb{R}^{d \times 3P^2}$ is the learned patch embedding matrix and $d$ is the model hidden dimension.

**Equivalently**, this can be implemented as a 2-D convolution with kernel size $P$, stride $P$:

```python
patch_embed = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=P, stride=P)
# Input: (N, 3, 224, 224) → Output: (N, d, 14, 14)
# Flatten spatial dims → (N, d, 196) → transpose → (N, 196, d)
```

### 3.2 CLS Token and Positional Encoding

A learnable `[CLS]` token is prepended to the sequence. The classification (or embedding) output is taken from this token's final representation.

$$\mathbf{z}_0 = [\mathbf{x}_{cls} ; \mathbf{z}_1 + \mathbf{e}_1 ; \mathbf{z}_2 + \mathbf{e}_2 ; \ldots ; \mathbf{z}_N + \mathbf{e}_N]$$

where $\mathbf{e}_i \in \mathbb{R}^d$ are **learned positional embeddings** (one per patch position + one for CLS). Unlike sinusoidal positional encoding in NLP transformers, ViT uses learnable 1-D position embeddings that index the patch grid linearly (left-to-right, top-to-bottom).

### 3.3 Transformer Encoder

The patch sequence passes through $L$ standard transformer encoder layers:

$$\mathbf{y}_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{y}_\ell)) + \mathbf{y}_\ell$$

where MSA = Multi-head Self-Attention, LN = Layer Normalisation, MLP = 2-layer feedforward. Pre-norm (LN before MSA/MLP) is standard in ViT.

### 3.4 Self-Attention Over Patches

Each patch attends to every other patch. The attention weight $A_{ij}$ between patch $i$ and patch $j$:

$$A_{ij} = \frac{\exp \left(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}\right)}{\sum_m \exp \left(\mathbf{q}_i^\top \mathbf{k}_m / \sqrt{d_k}\right)}$$

**Key insight:** this is $O(N^2)$ in the number of patches. For 196 patches this is 38,416 pairs — very manageable. But for 512×512 images with P=16 (1024 patches), quadratic cost becomes significant — motivating efficient attention variants in high-resolution diffusion.

### 3.5 ViT Variants

| Model | Layers $L$ | Hidden dim $d$ | Heads | Params | Patch size |
|-------|-----------|----------------|-------|--------|-----------|
| ViT-Ti/16 | 12 | 192 | 3 | 6M | 16 |
| ViT-S/16 | 12 | 384 | 6 | 22M | 16 |
| ViT-B/16 | 12 | 768 | 12 | 86M | 16 |
| ViT-L/16 | 24 | 1024 | 16 | 307M | 16 |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 14 |

CLIP uses ViT-B/32 (patch size 32, coarser but faster) or ViT-L/14 (finer patches, better quality).

---

## 4 · How It Works — Step by Step

**Step 1: Input tensor**
```
(3, 224, 224) image tensor — normalised to ImageNet stats
```

**Step 2: Patch extraction**
```
Divide into P×P patches → N = (224/16)² = 196 patches
Each patch: (3, 16, 16) → flatten → (768,)
```

**Step 3: Linear projection**
```
Each 768-dim patch vector projected through E ∈ ℝ^(d × 768)
→ 196 patch embeddings, each (d,) = (768,) for ViT-B
```

**Step 4: Prepend CLS token**
```
Total sequence: 1 + 196 = 197 tokens × 768 dims
```

**Step 5: Add positional embeddings**
```
Learned position embedding e_i ∈ ℝ^768 added to each position
This is the ONLY spatial information the model has — it's not baked into the architecture
```

**Step 6: Transformer encoder (×12 layers for ViT-B)**
```
Self-attention: each of 197 tokens attends to all 197 tokens
Cross-patch attention is what lets the model relate distant image regions
```

**Step 7: Output**
```
CLS token final representation → 768-dim image embedding
All 197 token representations → used for dense tasks (segmentation, detection)
```

---

## 5 · The Key Diagrams

### ViT Architecture Overview

```
Input image (3, 224, 224)
 │
 ▼
┌─────────────────────────────────────┐
│ Patch Embedding (Conv2d, P=16) │
│ (3, 224, 224) → (196, 768) │
└───────────────┬─────────────────────┘
 │
 Prepend [CLS]
 │
 ┌───────▼──────────┐
 │ + Pos Embeddings │ (learnable, 197 × 768)
 └───────┬──────────┘
 │
 ┌───────▼──────────┐
 │ Transformer L1 │ LN → MSA → Add
 │ │ LN → MLP → Add
 └───────┬──────────┘
 │
 ... × 12 layers (ViT-B)
 │
 ┌───────▼──────────┐
 │ Transformer L12 │
 └───────┬──────────┘
 │
 ┌───────▼──────────┐
 │ Layer Norm │
 └───────┬──────────┘
 │
 CLS output: (768,) ← image embedding (used by CLIP, etc.)
```

### Attention Pattern — Local vs Global

```
CNN (ResNet): ViT self-attention:
 
 ┌───┬───┬───┐ Every patch can attend to every other patch.
 │ * │ │ │ 
 ├───┼───┼───┤ Early layers: attend mostly to nearby patches
 │ │ R │ │ vs. (similar to CNN receptive field)
 ├───┼───┼───┤ 
 │ │ │ │ Deep layers: attend globally — patch at top-left
 └───┴───┴───┘ attends to patch at bottom-right
 
 Hard boundary: No boundary — pure data-driven attention
 max = kernel size anywhere in the image
```

---

## 6 · What Changes at Scale

| Dimension | Small / Laptop | Production |
|-----------|---------------|-----------|
| Patch size | P=32 (fewer tokens, faster) | P=14 or P=16 (more detail) |
| Resolution | 224×224 → 196 patches | 512×512 at P=16 → 1024 patches (10× compute) |
| Attention | Full $O(N^2)$ | Window attention / linear attention for high-res |
| Positional encoding | Learnable 1-D | Interpolated for arbitrary resolution; RoPE |
| Class token | Standard | Sometimes removed; global average pooling used instead |
| Pretraining data | ImageNet (1.3M) | LAION-5B (5 billion image-text pairs for CLIP) |

**The resolution problem:** diffusion models need high-resolution images. ViT at 512×512 with P=16 has 1024 tokens — quadratic attention is 1024² ≈ 1M operations per head. This is why Stable Diffusion runs the U-Net in **latent space** (64×64 latents, not 512×512 pixels) — Chapter 7.

---

## 7 · Common Misconceptions

**"ViT uses convolutions for the attention mechanism"**
No. The *patch embedding* step can be implemented as a single non-overlapping convolution, but this is just a linear projection. There are no convolutions anywhere in the transformer encoder itself.

**"Positional encoding in ViT is the same as in BERT"**
BERT uses learnable 1-D positional embeddings for token positions in text. ViT uses learnable 1-D positional embeddings for patch positions. The mechanism is identical — but the *meaning* differs: in ViT, positions encode 2-D spatial location linearised as row-major order. At inference on a different resolution, these positions must be bicubically interpolated.

**"ViT is strictly better than CNN for all image tasks"**
ViT overtakes CNN at scale (~100M+ parameters, ~100M+ training examples). For small datasets or limited compute, CNNs (ResNet, ConvNeXt) often outperform ViT because the inductive biases (local connectivity, translation equivariance) that CNNs build in, ViT must learn from data.

**"The CLS token is a special architectural feature"**
It is a learnable parameter prepended to every input sequence. BERT popularised it for classification. Its purpose: act as an aggregator that attends to all patch tokens and collects a summary representation. Some modern models use global average pooling of all patch tokens instead.

---

## 8 · Interview Checklist

### Must Know
- How does ViT convert an image to a sequence? (split into $P \times P$ patches, flatten, linear project)
- Why does ViT struggle at low data regimes compared to CNN?
- What does the CLS token output represent?

### Likely Asked
- "A ViT-B/16 receives a 224×224 image. How many tokens does the transformer encoder process?"
 → $196 \text{ patches} + 1 \text{ CLS} = 197$
- "How would you adapt ViT to process a 512×512 image without retraining positional embeddings?"
 → Bicubic interpolation of the learned position embeddings from 14×14 to 32×32 grid
- "CLIP uses ViT-L/14 as its image encoder. Why L/14 and not B/16?"
 → Smaller patch size (14 vs 16) → more patches → finer detail → better zero-shot performance

### Trap to Avoid
- Forgetting that attention in ViT is over patches, not pixels — patch count $N = (H/P)^2$, not $H \times W$
- Conflating the CLS token (learned summary) with the image as a whole — the CLS embedding is what CLIP stores in its lookup table
- Stating that ViT has "no inductive biases" — it has translational invariance through tied patch weights, just not local connectivity
- **Swin vs plain ViT:** Swin introduces a hierarchical feature pyramid with shifted-window attention (local, $O(n)$ not $O(n^2)$); each stage halves spatial resolution and doubles channels like a CNN. Preferred for dense prediction (detection, segmentation). Trap: "Swin is always better than ViT" — for global understanding tasks (CLIP, large-scale classification), plain ViT with full attention often matches or exceeds Swin; Swin's advantage is on spatial-resolution-sensitive tasks
- **ViT data requirements:** ViT-B trained from scratch on ImageNet-1k underperforms ResNet-50 with equal training time because ViT has no spatial inductive biases; it needs JFT-300M scale or strong augmentation (DeiT) to compensate. Trap: "ViT always beats ResNet" — only with sufficient scale and data; DeiT and MAE address this for the constrained-data regime

---

## 9 · What's Next

→ **[CLIP.md](../CLIP/CLIP.md)** — ViT gives us a 768-dimensional image embedding. CLIP gives us the critical missing piece: a **paired text embedding** in the same space. By training a ViT image encoder and a transformer text encoder jointly with contrastive loss on 400 million image-text pairs, CLIP learns that the embedding of "a photograph of a cat" is close to the embedding of an actual photograph of a cat — enabling zero-shot image classification, semantic image retrieval, and (crucially) the text conditioning mechanism inside Stable Diffusion.

## Illustrations

![Vision Transformers - patchify, ViT architecture, receptive field vs CNN, scale curve](img/Vision%20Transformers.png)
