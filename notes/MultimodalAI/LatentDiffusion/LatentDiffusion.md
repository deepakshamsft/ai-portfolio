# Latent Diffusion — Compress, Diffuse, Decode

> After reading this you will understand why Stable Diffusion works in a compressed latent space, what the VAE does, and how the three components (VAE + U-Net + CLIP) assemble into a single pipeline.

## 1 · Core Idea

Diffusing directly on 512×512 pixels costs ~262 000 dimensions per image. Stable Diffusion instead:

1. **Encodes** the image to a 64×64×4 latent with a VAE encoder (8× spatial compression)
2. **Diffuses** in that 16 384-dimensional latent space (16× cheaper per step)
3. **Decodes** the denoised latent back to pixels with the VAE decoder

Same theory as DDPM; only the domain changes. This is why SD can run on consumer hardware.

## 2 · Running Example

PixelSmith v4: given a text prompt, generate a 512×512 image using the `diffusers` library and a local SD checkpoint (or SDXL-Turbo for instant results).

You can run this on a CPU in under 3 minutes with SDXL-Turbo's 4-step schedule.

## 3 · The Math

### VAE: Encoder → Latent → Decoder

The VAE encoder maps an image $x$ to a Gaussian distribution in latent space:

$$q_\phi(z | x) = \mathcal{N}(\mu_\phi(x),\; \sigma^2_\phi(x)\,\mathbf{I})$$

Training objective (ELBO):

$$\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x)\,\|\,\mathcal{N}(0, \mathbf{I}))$$

The first term is pixel reconstruction; the second regularises the latent space to be roughly unit-Gaussian. This is what makes sampling from latent space meaningful.

At inference: encode to $\mu_\phi(x)$ (no sampling), diffuse, decode $z \to x$.

### The Latent Rescaling Trick

Raw latent activations have variance ≠ 1. Stable Diffusion multiplies the VAE output by a **scaling factor** $s = 0.18215$ before feeding into the diffusion U-Net:

$$z_{\text{scaled}} = s \cdot \text{VAE\_encode}(x)$$

This rescales latents to unit variance so the DDPM noise schedule works correctly. Forgetting this is a common source of blurry outputs.

### Cross-Attention for Text Conditioning

In SD, conditioning is not via label embedding addition (Ch.5) but via **cross-attention layers** inside the U-Net:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where:
- $Q$ = image feature map (flattened spatial positions, projected)
- $K$ = CLIP text embeddings (each token), projected
- $V$ = CLIP text embeddings, projected

Each spatial position in the U-Net attends over all text tokens. This is how "a red cat" makes the model attend to "red" at fur pixels and "cat" at shape pixels.

### Full SD Pipeline

```
Input text  ───▶  CLIP Text Encoder  ───▶  text_embeds (77×768)
                                               │
                                         cross-attention
                                               │
Input noise ───▶  [DDIM 20 steps] ◀──── U-Net (in latent space)
                        │
                        ▼
                   denoised z
                        │
                   VAE Decoder
                        │
                        ▼
                   512×512 image
```

## 4 · How It Works — Step by Step

### SD Inference

1. **Tokenise** prompt → CLIP tokenizer → token IDs
2. **Encode** with CLIP text encoder → `text_embeds` tensor (shape: `[batch, 77, 768]`)
3. **Sample** random latent noise `z_T ~ N(0, I)` shape `[1, 4, 64, 64]`
4. **Denoise** for N steps using the U-Net, which receives `(z_t, t, text_embeds)` and outputs `eps_pred`
5. **Decode** denoised `z_0` with VAE decoder → `[1, 3, 512, 512]` pixel image
6. **Rescale** pixel values from `[-1, 1]` to `[0, 255]`

### Training SD (for reference)

1. Take a real image + caption pair
2. VAE-encode the image to `z_0`, scale by 0.18215
3. Sample timestep `t`, add noise: `z_t = sqrt(ab_t)*z_0 + sqrt(1-ab_t)*eps`
4. CLIP-encode the caption to `text_embeds`
5. U-Net predicts `eps` given `(z_t, t, text_embeds)` via cross-attention
6. Loss: MSE between predicted and actual `eps`

The VAE is **frozen during diffusion training** — only the U-Net is updated.

## 5 · The Key Diagrams

```
SD Architecture — Dimensions at Each Stage:

┌──────────────────────────────────────────────────────────────────┐
│ Image space (pixel U-Net, e.g. DDPM on MNIST)                   │
│   28×28×1  ──────────────────────────────── 28×28×1             │
│   (784 dim)                                                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Latent space (Stable Diffusion 1.x)                              │
│  512×512×3  ──[VAE enc]──▶  64×64×4  ──[U-Net]──▶  64×64×4     │
│  (786 432 dim)              (16 384 dim)                          │
│                                 │                                  │
│                           [VAE dec]                                │
│                                 │                                  │
│                           512×512×3                                │
└──────────────────────────────────────────────────────────────────┘

VAE compression ratio: 786 432 / 16 384 = 48×
                       (8× spatial × 3 channels → 4 channels = ×48 net)
```

## 6 · What Changes at Scale

| Model | VAE latent dim | U-Net params | Text encoder | Steps (typical) |
|-------|---------------|-------------|-------------|----------------|
| SD 1.5 | 64×64×4 | 860M | CLIP ViT-L/14 (123M) | 20–50 |
| SD 2.1 | 64×64×4 | 865M | CLIP-based OpenCLIP | 20–50 |
| SDXL | 128×128×4 | 2.6B | Two CLIP encoders | 20–30 |
| SDXL-Turbo | 128×128×4 | 2.6B + ADD | Same | 1–4 |
| SD 3.5 | 128×128×16 | 8B (DiT) | Three encoders | 20–50 |
| Flux | 128×128×16 | 12B (MMDiT) | T5-XXL + CLIP | 20–50 |

The trend is: larger latent channels (4→16), larger U-Net or switch to Diffusion Transformer (DiT), stronger text encoder (CLIP→T5).

## 7 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "SD's VAE is trained jointly with the diffusion model" | No — the VAE is trained separately; SD fine-tunes only the U-Net |
| "You can resize images freely with SD" | SD 1.x was trained at 512×512; going to 768 causes artefacts. SDXL was trained at 1024×1024 |
| "The CLIP encoder in SD is the same as OpenAI CLIP" | SD 1.x uses the OpenAI CLIP ViT-L/14 text encoder (frozen). SD 2.x uses OpenCLIP |
| "Latent diffusion is only about speed" | Also about quality: pixel-space models struggle at high resolution; latent models can condition cross-attention on spatial features more efficiently |
| "The scaling factor 0.18215 is arbitrary" | It is empirically determined so the latent variance ≈ 1.0 under a unit-Gaussian prior, matching the DDPM assumption |

## 8 · Interview Checklist

### Must Know
- The three components of Stable Diffusion: **VAE** (compress/decompress), **U-Net** (diffuse in latent), **CLIP** (condition on text)
- Why latent space: ~48× cheaper diffusion without meaningful quality loss
- Cross-attention mechanism for text conditioning: Q from image features, K/V from text tokens

### Likely Asked
- *"What is the latent scaling factor and why is it needed?"* — 0.18215; rescales VAE output to unit variance to match DDPM's N(0,I) prior
- *"How does SDXL improve on SD 1.5?"* — 2× larger latent spatial resolution (128×128), 3× more U-Net params, two CLIP encoders concatenated, trained on aspect-ratio bucketing
- *"What is a Diffusion Transformer (DiT)?"* — Replace U-Net with a pure Transformer; patches of the latent are tokens; SD 3.5 and Flux use this architecture

### Trap to Avoid
- Don't say "the VAE is trained as part of SD" — it's pre-trained separately. The SD training only updates the denoising U-Net. During inference the VAE decoder is also not updated.

## 9 · What's Next

[TextToImage.md](../TextToImage/TextToImage.md) — beyond basic text-to-image: prompt engineering, img2img, inpainting, and ControlNet for spatially guided generation.

## Illustrations

![Latent diffusion - pipeline, pixel vs latent shape, compute savings, VAE tradeoff](img/Latent%20Diffusion.png)
