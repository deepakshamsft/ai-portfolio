# Text-to-Image — Prompts, img2img, Inpainting, ControlNet

> **The story.** **DALL·E** (Ramesh et al., OpenAI, **January 2021**) was the first text-to-image model to capture mainstream attention — a 12B-parameter discrete autoregressive model that could draw "an armchair shaped like an avocado." **DALL·E 2** (April 2022) replaced the autoregressive backbone with diffusion + CLIP guidance and crossed the line into *photorealistic*. **Imagen** (Google, May 2022) and **Parti** (Google, June 2022) appeared within weeks. The opening of the floodgates was **Stable Diffusion 1.4** (Stability AI, **August 2022**) under a permissive open licence — within months, **AUTOMATIC1111**'s WebUI, **ComfyUI**, **LoRAs**, **ControlNet**, **inpainting** workflows, and Civitai's model marketplace built an entire creative-tooling ecosystem around the open weights. **SDXL** (2023), **SD3** (2024), and **FLUX** (Black Forest Labs, 2024) carried the open lineage forward. The user-facing prompt-engineering, img2img, and inpainting workflows you use in 2026 are all surface area on this same architecture.
>
> **Where you are in the curriculum.** [LatentDiffusion](../latent_diffusion) gave you the model; [GuidanceConditioning](../guidance_conditioning) gave you the steering wheel. This chapter is the user-facing pipeline: how prompt token choice and order affect [CLIP](../clip) embeddings, why img2img is just "start denoising from a noisy version of an existing image," how inpainting masks the U-Net's loss, and how ControlNet imposes structural constraints. After this you can ship the [PixelSmith](../README.md) studio's headline feature.

![Text-to-image flow animation](img/text-to-image-flow.gif)

*Flow: text is tokenized and encoded, then the denoiser iteratively transforms noise into an image consistent with the prompt.*

---

## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge needs <5% unusable generations to compete with freelancers (who deliver on the first try after seeing the brief).

**Current blocker at Chapter 8**: CFG (Ch.7) improved prompt adherence but **cannot guarantee composition**. "Product at 45-degree angle on white background" still fails 60% of time. Text alone isn't precise enough for spatial constraints.

**What this chapter unlocks**: **ControlNet** — condition diffusion on edge maps, depth maps, pose skeletons. Designer sketches rough layout → ControlNet enforces structure → **95% first-try success rate**. Also: inpainting (edit specific regions), img2img (transform reference photos), prompt weighting.

---

### The 6 Constraints — Snapshot After Chapter 8

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | ⚡ **~3.8/5.0** | ControlNet improves composition quality |
| #2 Speed | <30 seconds | ✅ **~18s** | ControlNet adds minimal overhead (~2s per image) |
| #3 Cost | <$5k hardware | ✅ **$2.5k laptop** | Unchanged from Ch.6 |
| #4 Control | <5% unusable | ✅ **~3% unusable** | ControlNet guarantees structure, hits target! |
| #5 Throughput | 100+ images/day | ⚡ **~80 images/day** | Team of 2, still ramping up workflow |
| #6 Versatility | 3 modalities | ⚡ **Text→Image production-ready** | Img2img, inpainting, ControlNet all deployed |

---

### What's Still Blocking Us After This Chapter?

**Video demand**: Clients now want **15-second social media video ads**, not just static images. Images = solved. Video = unsolved.

**Next unlock (Ch.9)**: **Text-to-Video** — extend diffusion to temporal dimension. AnimateDiff adds motion modules to SD → generate 16-frame 512×512 clips.

---

## 1 · Core Idea

Text-to-image is the user-facing layer of latent diffusion. The core pipeline from Ch.7 stays intact; this chapter is about the *inputs and outputs* you can control:

- **Prompt engineering** — how token choice and token order affect CLIP embeddings
- **Negative prompts** — expanding CFG to subtract unwanted concepts
- **img2img** — start from a partially-noised real image instead of pure noise
- **Inpainting** — diffuse only within a user-drawn mask
- **ControlNet** — inject structural guidance (edges, depth, skeleton) as a spatial conditioning signal

## 2 · Running Example

**PixelSmith v5 — VisualForge ControlNet product-positioning brief.** The e-commerce team needs product shots at a specific angle (45°, three-quarter view) that current text-only generation can't reliably produce. ControlNet + edge maps solves the positioning problem.

```
Brief type: Product-on-white with precise camera angle
Input: Sketch/edge map of the product at 45° angle + text prompt
Prompt: "Mango leather crossbody bag, three-quarter view, white background, studio lighting"
ControlNet condition: Canny edge map (from a reference 3D render or sketch)
Result: Generated image respects both the angle (from edge map) and the brand description (from text)
```

> 📖 **Educational proxy:** ControlNet img2img math is demonstrated on a simplified domain to show the latent interpolation. Production ControlNet on VisualForge briefs uses the same `diffusers` API shown in §5.

## 3 · The Math

### Prompt Engineering: Weighted Token Embeddings

CLIP text encoder output is an average over token positions. Each token occupies one position in the 77-length context. Practical implications:

- **Order matters slightly**: earlier tokens get slightly more attention in the bidirectional encoder, but effect is small
- **Repeated tokens = linearly stronger signal**: `a beautiful beautiful landscape` ≈ `(a beautiful landscape:1.2)` via prompt weighting in Automatic1111
- **Token budget (77)**: long prompts are truncated silently — put most important concepts early

**Prompt weighting** in practice multiplies the embedding vector:

$$\mathbf{e}_{\text{weighted}} = w \cdot \mathbf{e}_{\text{token}}$$

This is just scalar multiplication of the embedding vector before the diffusion U-Net receives it via cross-attention.

### Negative Prompts

Extend the CFG equation with a negative prompt $c^-$:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c^+) - \epsilon_\theta(x_t, c^-))$$

With standard CFG: $c^- = \emptyset$ (null token). With a negative prompt: $c^- = \text{CLIP}(\text{"blurry, watermark, bad anatomy"})$.

This requires **three** U-Net calls per step (uncond, positive, negative). In practice, uncond and negative are combined into a single "negative embedding" batch:

$$\hat{\epsilon} = \epsilon_\theta(x_t, c^-) + w \cdot (\epsilon_\theta(x_t, c^+) - \epsilon_\theta(x_t, c^-))$$

Only **two** calls per step: one for positive, one for negative (negative acts as the new baseline).

### img2img: Start from Partial Noise

Unlike text-to-image which starts from $x_T \sim \mathcal{N}(0, I)$, img2img:

1. Takes a real image $x_0$
2. Adds noise for $t_{\text{start}} < T$ steps: $x_{t_\text{start}} = \sqrt{\bar{\alpha}_{t_\text{start}}}x_0 + \sqrt{1-\bar{\alpha}_{t_\text{start}}}\epsilon$
3. Runs the reverse process from $t_{\text{start}}$ to 0

**Strength** parameter $s \in [0, 1]$:
- $s = 1.0$: start from pure noise (identical to text-to-image)
- $s = 0.5$: start halfway (500 out of 1000 steps); image structure is preserved, details are changed
- $s = 0.1$: minimal noise; only small details are changed

$$t_\text{start} = \lfloor s \cdot T \rfloor$$

### Inpainting: Mask-Constrained Generation

At every denoising step, the known pixels (outside the mask) are re-noised and composited back:

$$x_t^{\text{masked}} = \text{mask} \odot x_t + (1-\text{mask}) \odot \big(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon\big)$$

This forces the unmasked region to remain consistent with the original while the masked region is freely generated by the model.

### ControlNet: Spatial Conditioning

ControlNet (Zhang et al. 2023) adds a control signal (Canny edges, depth map, skeleton, etc.) as an additional input to the U-Net. Architecture:

1. **Clone** the U-Net encoder (freeze the original)
2. The **cloned encoder** receives `(z_t + control_signal)` and produces a sequence of feature residuals
3. These residuals are **added** to the corresponding skip connections in the main U-Net

$$\text{skip}_l^{\text{ControlNet}} = \text{skip}_l^{\text{original}} + \text{residual}_l^{\text{control}}$$

The main U-Net decoder then assembles the spatially-guided features into the output.

Key property: ControlNet requires **no retraining of the main U-Net**. Only the cloned encoder's weights are trained on (image, control, prompt) triplets.

## 4 · How It Works — Step by Step

### img2img Workflow

```
Input image x₀ (e.g., rough sketch)
 │
 VAE encode → z₀
 │
 Add noise for t_start steps → z_{t_start}
 │
 Diffuse from t_start → 0 (same DDIM/DPM schedule)
 (cross-attention with CLIP text embedding throughout)
 │
 VAE decode → output image
```

Practical use cases:
- Sketch → realistic rendering (strength=0.7)
- Style transfer (strength=0.5)
- Day → night (strength=0.4)
- Quality improvement: upscale + img2img at low strength

### ControlNet Workflow

```
Prompt ──────────────────────────────────────────┐
 ▼
Control signal (Canny edges) → Cloned Encoder → residuals
 ↓
Noise z_T → Main U-Net (encoder) ─────────────────┤
 (add residuals to skips)
 Main U-Net (decoder)
 │
 VAE decode
 │
 512×512 output image
```

---

## 5 · Production Example — VisualForge in Action

**Brief type: ControlNet product-at-angle with edge-map conditioning**

The VisualForge e-commerce team needs products photographed at the same 45° angle across the entire spring collection for visual consistency. ControlNet enforces this constraint without a physical photo studio.

```python
# Production: ControlNet Canny for VisualForge product-positioning brief
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import torch
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# VisualForge brief: 45° three-quarter view of spring-collection bag
reference_sketch = Image.open("vf_sketch_45deg.png")  # 3D render or hand sketch
canny = CannyDetector()
edge_map = canny(reference_sketch, low_threshold=100, high_threshold=200)

image = pipe(
    "Mango leather crossbody bag, three-quarter view, white background, studio lighting, product photography",
    negative_prompt="distorted, blurry, deformed, people, logo, watermark, cluttered",
    image=edge_map,
    controlnet_conditioning_scale=1.0,  # 1.0 = strict edge adherence; 0.5 = loose
    num_inference_steps=20,
    guidance_scale=7.5,
).images[0]
image.save("vf_product_45deg.png")
```

**VisualForge ControlNet constraint scorecard:**

| Metric | Target | Result |
|--------|--------|--------|
| Angle consistency (45° view) | >90% of batch | 94% ✅ (ControlNet enforces geometry) |
| Background compliance (white) | >95% | 96% ✅ |
| Creative quality score | ≥4.0/5.0 | 4.3/5.0 ✅ |
| Batch time (50 images) | <30 min | ~17 min ✅ |

> 💡 `controlnet_conditioning_scale=1.0` enforces strict edge adherence. For stylistic freedom while maintaining pose, use 0.6–0.8.

---

## 6 · The Key Diagrams

```
img2img vs. text-to-image vs. inpainting:

 text-to-image: N(0,I) ─── [DDIM 20 steps] ──▶ image
 (full 1000 steps of noise)

 img2img (s=0.5): real_img ─[add noise 500 steps]─▶ z_t=500
 ─── [DDIM 10 steps] ──▶ image
 (structure preserved; details changed)

 inpainting: real_img + mask
 at each step: paste original pixels outside mask
 at each step: free diffusion inside mask


Strength dial in img2img:
 s=0.1 ████░░░░░░ (10% noise, minor changes)
 s=0.5 █████░░░░░ (50% noise, moderate changes)
 s=0.9 █████████░ (90% noise, major changes)
 s=1.0 ██████████ (100% noise, identical to text-to-image)
```

## 7 · What Changes at Scale

- **SDXL** separates prompt into `prompt` (coarse, for base model) and `prompt_2` (fine, for refiner model), allowing two-stage generation at full resolution
- **IP-Adapter** extends the idea of ControlNet to use a reference *image* as the conditioning signal instead of a structural map — enables style or identity transfer
- **LoRA** (Low-Rank Adaptation): fine-tune SD on 5–20 subject images by adding tiny rank-decomposition matrices to U-Net attention layers. Captures person/object identity without full fine-tuning cost
- **Textual Inversion**: optimise a new token embedding vector (not the weights) to represent a novel concept. Simpler but less capable than LoRA

## 8 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Negative prompts erase concepts" | They steer *away* from the negative concept using CFG; they don't remove tokens from the model's vocabulary |
| "Higher strength = always better in img2img" | High strength (>0.9) ignores the input image almost entirely; lower strength preserves structure |
| "ControlNet requires a new base SD model" | It works as an add-on to any SD checkpoint; the base model is not modified |
| "Prompt length doesn't matter" | 77 token limit is real — SD truncates silently at 77 bpe tokens |
| "Inpainting always generates seamless results" | Hard edges in the mask cause seam artefacts; soft/feathered masks help; latent-space inpainting is smoother than pixel-space |

## 9 · Interview Checklist

### Must Know
- img2img = add partial noise then denoise; strength parameter controls how much of the original is preserved
- Negative prompts extend CFG: replace unconditional baseline with negatively-conditioned baseline
- ControlNet architecture: frozen original encoder + trainable cloned encoder whose residuals are injected as skip connection additions

### Likely Asked
- *"How would you make SD generate images in a specific person's style?"* — LoRA fine-tune on ~15 images (~15 min on a consumer GPU)
- *"What is the difference between ControlNet and img2img?"* — ControlNet injects a spatial map (edges, depth) as structural guidance at every attention layer; img2img starts from a partially-noised version of an image
- *"Why do people use negative prompts like 'lowres, bad anatomy'?"* — These are common LAION dataset artifacts; subtracting their embeddings pushes the output away from low-quality image cluster in latent space

### Trap to Avoid
- Don't confuse **classifier guidance** (Ch.5, needs pretrained classifier) with **classifier-free guidance** (Ch.5) and with the **negative prompt extension** of CFG (this chapter). They all use the word "guidance" but are different mechanisms.

---

## 10 · Progress Check — What Have We Unlocked?

### Before This Chapter
- **Constraint #4 (Control)**: ⚡ <15% unusable, text alone cannot guarantee composition
- **VisualForge Status**: "Product at 45-degree angle" fails 60% of time

### After This Chapter
- **Constraint #4 (Control)**: ✅ **~3% unusable** → ControlNet guarantees structure, target hit!
- **Constraint #5 (Throughput)**: ⚡ **~80 images/day** → Team of 2 designers, workflow ramping up
- **VisualForge Status**: Designer sketches layout → ControlNet (Canny edge) + text prompt → 95% first-try success

---

### Key Wins

1. **ControlNet structural control**: Sketch edge map → guarantees composition, eliminates 60% failure rate
2. **Inpainting**: Edit specific regions (e.g., change product color) without regenerating entire image
3. **Img2img**: Transform reference photos with strength parameter (0.3 = subtle, 0.8 = major changes)
4. **Prompt weighting**: `(blue:1.3)` emphasizes blue, `(red:0.7)` de-emphasizes red

---

### What's Still Blocking Production?

**Video capability**: Clients now request **15-second social media video ads** (product rotating, zoom effects). Static images = solved. Video generation = unsolved.

**Next unlock (Ch.9)**: **Text-to-Video** — extend diffusion to temporal dimension. AnimateDiff adds temporal attention layers → 16-frame clips at 512×512.

---

## 11 · What's Next

[TextToVideo.md](../text_to_video/text-to-video.md) — add a temporal dimension. Generating consistent motion is the key unsolved challenge in T2V.

## Illustrations

![Text-to-image - txt2img, img2img strength, inpainting, ControlNet stack](img/Text%20to%20Image.png)
