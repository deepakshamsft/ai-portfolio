# Multimodal AI — How to Read This Collection

> This document is your **entry point and reading map**. It explains the conceptual arc across all notes, defines the running example that threads through every chapter, shows how each document connects to the others, and prescribes reading paths based on your goal.

---

## The Central Story in One Paragraph

Modern generative AI systems — the ones that turn a text prompt into a photorealistic image, generate a video from a sentence, or answer questions about a photograph — are all built on one foundational idea: **any signal (pixels, audio frames, video clips, text tokens) can be projected into the same high-dimensional embedding space, and a transformer (or diffusion model) can learn the joint distribution over those signals**. To build that intuition from the ground up, you need four layers of knowledge: (0) how raw signals become tensors and then tokens (Foundations + Vision Transformers), (1) how meaning is aligned across modalities without labels (CLIP and contrastive learning), (2) how models generate entirely new signals they were never shown — not just predict them (Diffusion Models and Latent Diffusion), and (3) how those generation capabilities are extended to video, conditioned on structure, and baked into language models (Text-to-Video + Multimodal LLMs). The documents in this collection cover each layer in depth, and they deliberately cross-reference the AI track (embeddings, transformers, fine-tuning) because the layers are not independent.

---

## The Running Example — PixelSmith

Every note in this track is anchored to a single growing system: **PixelSmith**, a local AI-powered creative studio you are building from scratch.

```
PixelSmith v1 (after Foundations):
  Input: raw image file → Output: pixel tensor, patch embeddings

PixelSmith v2 (after CLIP):
  Input: text query → Output: ranked images by semantic similarity

PixelSmith v3 (after Diffusion Models):
  Input: noise → Output: generated image (DDPM from scratch)

PixelSmith v4 (after Latent Diffusion):
  Input: text prompt → Output: generated image (Stable Diffusion, locally)

PixelSmith v5 (after ControlNet / img2img):
  Input: text prompt + sketch → Output: controlled generated image

PixelSmith v6 (after Multimodal LLMs):
  Input: photograph + question → Output: natural language answer
```

The key constraint: **PixelSmith must run on a stock developer laptop** — no A100, no cloud GPU budget. This forces every chapter to confront the same question production engineers face: *how do you get serious generative AI to run where you actually are?*

---

## How We Got Here — A Short History of Multimodal & Generative AI

Image generation went from "blurry 32×32 digits" to "photoreal 4K video" in about a decade. The ordering of our chapters mirrors the ordering of the breakthroughs that made it possible.

| Era | Year | Breakthrough | Why it set up the next chapter |
|---|---|---|---|
| **CNN era** | 1998 | **LeNet-5** (LeCun) | Convolutions as the vision primitive. Dominated vision for 20 years. |
| | 2012 | **AlexNet** — ImageNet by a landslide | Deep learning won vision. Started the representation-learning era. |
| | 2014 | **VGG / GoogLeNet / Inception**; **ResNet** (2015) | Showed that depth + skip connections scale. |
| **First generative wave** | 2013 | **VAE** (Kingma & Welling) | First tractable deep latent-variable model. The encoder/decoder duality reappears in Stable Diffusion. → [LatentDiffusion](./LatentDiffusion/). |
| | 2014 | **GAN** (Goodfellow et al.) | First genuinely convincing image generation. Defined the bar diffusion had to clear. |
| | 2015 | **U-Net** (Ronneberger) — segmentation architecture | The exact architecture later used as the diffusion denoiser. |
| | 2015 | **StyleGAN2, BigGAN** (2018–2019) | Peak GAN quality — but notoriously unstable to train. Made the field hungry for something better. |
| **Attention + vision** | 2017 | **Transformer** (Vaswani et al.) | Made sequence length a compute problem, not an architecture problem. |
| | 2020 Oct | **Vision Transformer (ViT)** (Dosovitskiy et al., Google) | Images as sequences of patches. Convolutions were no longer required. → [VisionTransformers](./VisionTransformers/). |
| **Alignment via contrast** | 2021 Jan | **CLIP** (OpenAI, Radford et al.) | 400M image-text pairs + contrastive loss → text and images share one space. The backbone of almost every text-to-image system since. → [CLIP](./CLIP/). |
| | 2021 Jan | **DALL·E 1** (OpenAI) — discrete VAE + autoregressive transformer | Text-to-image as a *token sequence* problem. Outpaced by diffusion within a year. |
| **Diffusion revolution** | 2015 | **Diffusion probabilistic models** (Sohl-Dickstein et al.) | Mathematically elegant but slow. Ignored for five years. |
| | 2020 Jun | **DDPM** (Ho, Jain, Abbeel) | Showed diffusion could match GANs on quality *and* stability. → [DiffusionModels](./DiffusionModels/). |
| | 2020 Oct | **DDIM** (Song et al.) | Deterministic, ~10× fewer sampling steps. → [Schedulers](./Schedulers/). |
| | 2021 May | **Classifier-Free Guidance** (Ho & Salimans) | The "guidance scale" knob every user now turns. → [GuidanceConditioning](./GuidanceConditioning/). |
| | 2022 Apr | **DALL·E 2 / unCLIP** (OpenAI) | CLIP + diffusion, at scale. Mainstream moment. |
| | 2022 May | **Imagen** (Google) | Proved a frozen text encoder (T5) + diffusion beats training a joint model. |
| | 2022 Aug | **Stable Diffusion / Latent Diffusion** (Rombach et al.) — **open weights** | Diffusion in VAE latent space, runnable on a consumer GPU. Democratised everything. → [LatentDiffusion](./LatentDiffusion/). |
| | 2022 Oct | **DPM-Solver / DPM-Solver++** (Lu et al.) | 10–20-step sampling with no quality loss. |
| **Control era** | 2023 Feb | **ControlNet** (Zhang et al.) | Structural conditioning (pose, edges, depth) without retraining the base model. → [TextToImage](./TextToImage/). |
| | 2023 | **LoRA for diffusion**, **IP-Adapter**, **InstantID** | Tiny adapters replaced full fine-tuning. |
| **Multimodal LLMs** | 2023 Mar | **GPT-4V (Vision)**; **LLaVA** (Liu et al.) — instruction-tuned vision-language | Vision encoders projected into an LLM's token space. → [MultimodalLLMs](./MultimodalLLMs/). |
| | 2023–2024 | **BLIP-2 / Q-Former**, **Gemini**, **Claude 3**, **GPT-4o** | Native-multimodal became the default for frontier models. |
| **Video era** | 2023 | **AnimateDiff**, **Stable Video Diffusion** | Temporal layers bolted onto image diffusion. |
| | 2024 Feb | **Sora** (OpenAI) — diffusion transformer over spacetime patches | Text-to-video reached minute-long, coherent clips. → [TextToVideo](./TextToVideo/). |
| | 2024–2025 | **Runway Gen-3**, **Veo**, **Kling**, **Luma Dream Machine** | Video generation became a product category. |
| **Local / practical era** | 2023–2026 | **ComfyUI**, **AUTOMATIC1111**, **SDXL / Flux.1**, quantised UNets (fp8, nf4), **LCM / Lightning distillation** | Generation collapsed from minutes on an A100 to seconds on a laptop. → [LocalDiffusionLab](./LocalDiffusionLab/). |
| **Evaluation era** | 2017 → 2024 | **FID / IS → CLIPScore → HPSv2 → human preference benchmarks** | As quality saturated, evaluation shifted from pixel-distribution metrics to preference alignment. → [GenerativeEvaluation](./GenerativeEvaluation/). |

**The through-line:** each chapter corresponds to a specific obstacle that was removed. Pixels were too high-dimensional → VAE + latent diffusion. Sampling was too slow → DDIM / DPM-Solver. Training was too unstable → diffusion replaced GANs. Text and images lived in separate spaces → CLIP. Base models were too rigid → ControlNet / LoRA. Images weren't enough → MLLMs + video diffusion. Cloud GPUs were too expensive → quantisation + distillation. Reading the chapters in order is reading this list of obstacles falling.

---

## The Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTIMODAL AI SYSTEM                                 │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     GENERATION LAYER                                    │ │
│  │                                                                          │ │
│  │   Diffusion Models · Latent Diffusion · Schedulers                      │ │
│  │   Text-to-Image · Text-to-Video · ControlNet                            │ │
│  │   [DiffusionModels.md] [LatentDiffusion.md] [TextToImage.md]            │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                 │                                             │
│          ┌──────────────────────┴───────────────────┐                       │
│          │                                           │                       │
│  ┌───────▼───────────────────┐     ┌────────────────▼───────────────────┐  │
│  │  ALIGNMENT LAYER           │     │  UNDERSTANDING LAYER                │  │
│  │                             │     │                                     │  │
│  │  How language and vision    │     │  How models answer questions        │  │
│  │  share the same space       │     │  about images and video             │  │
│  │                             │     │                                     │  │
│  │  [CLIP.md]                  │     │  [MultimodalLLMs.md]                │  │
│  │  [GuidanceConditioning.md]  │     │  [TextToVideo.md]                   │  │
│  └───────────────────────────┘     └────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     REPRESENTATION LAYER                                │ │
│  │                                                                          │ │
│  │   How raw signals become tokens the model can process                   │ │
│  │   [MultimodalFoundations.md] [VisionTransformers.md]                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Document Map

### Foundation Notes (read before the core notes)

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [MultimodalFoundations.md](./MultimodalFoundations/MultimodalFoundations.md) | What multimodal AI is; how images, audio, and video become tensors; the representation problem; the modality gap | How does a photograph become a matrix? Why can't we just feed pixels to a transformer? What is a modality gap? |
| [VisionTransformers.md](./VisionTransformers/VisionTransformers.md) | ViT: splitting images into patches, positional encoding, self-attention over patches; how ViT differs from CNNs and why it won | What is a patch embedding? How does ViT handle position? Why did attention beat convolution at scale? |

### Alignment Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [CLIP.md](./CLIP/CLIP.md) | Contrastive Language-Image Pretraining; dual-encoder architecture; InfoNCE loss; zero-shot classification without any labelled data | How does a model learn that a photo of a cat matches the text "a cat"? What is contrastive loss? What is zero-shot transfer? |
| [GuidanceConditioning.md](./GuidanceConditioning/GuidanceConditioning.md) | Classifier guidance, classifier-free guidance (CFG), text conditioning via cross-attention; what the guidance scale actually does; negative prompts | Why does guidance scale 7.5 produce sharper images than 1.0? What does a negative prompt actually do mechanically? |

### Generation Core Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [DiffusionModels.md](./DiffusionModels/DiffusionModels.md) | The math of DDPM: the forward noising process, the reverse denoising process, score matching, noise schedules; why diffusion beat GANs | What is the forward process? What does the U-Net actually predict — the image or the noise? Why is diffusion more stable than GAN training? |
| [Schedulers.md](./Schedulers/Schedulers.md) | DDPM vs DDIM vs DPM-Solver; how to generate in 4 steps instead of 1,000; deterministic sampling; the speed/quality trade-off | Why does DDIM need fewer steps? What changes when you switch from DDPM to DPM-Solver? What is a sampler doing geometrically? |
| [LatentDiffusion.md](./LatentDiffusion/LatentDiffusion.md) | Why pixel-space diffusion is too slow; VAEs as a compression layer; the Stable Diffusion architecture (VAE + U-Net + CLIP text encoder); latent space geometry | What is a VAE? Why run diffusion in latent space instead of pixel space? How does text reach the U-Net in Stable Diffusion? |

### Application Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [TextToImage.md](./TextToImage/TextToImage.md) | End-to-end text-to-image pipeline; prompt engineering for images; img2img; inpainting; ControlNet for structural conditioning | How does prompt weight syntax work? What is ControlNet's conditioning signal? How does inpainting avoid repainting the whole image? |
| [TextToVideo.md](./TextToVideo/TextToVideo.md) | Extending diffusion to the temporal dimension; the consistency problem; overview of video generation (Sora, CogVideo, AnimateDiff) | What makes video harder than images? How does Sora model spacetime? What is AnimateDiff doing differently from full video models? |
| [MultimodalLLMs.md](./MultimodalLLMs/MultimodalLLMs.md) | Connecting vision encoders to LLM decoders; LLaVA, BLIP-2, GPT-4V, Gemini; visual instruction tuning; the projection layer | How does GPT-4V "see"? What is a Q-Former? How do you fine-tune an LLM to accept image tokens? |

### Evaluation Note

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [GenerativeEvaluation.md](./GenerativeEvaluation/GenerativeEvaluation.md) | How do you measure the quality of a generated image or video? FID, IS, CLIP score, LPIPS, human preference models; the alignment problem in evaluation | What does FID actually measure? Why is CLIP score better for text-image alignment than FID? Why is human evaluation still the gold standard? |

### Capstone

| File | Purpose |
|------|---------|
| [LocalDiffusionLab.md](./LocalDiffusionLab/LocalDiffusionLab.md) | Hands-on capstone: train a DDPM from scratch on MNIST (runs in ~5 minutes on CPU), visualise every step of the diffusion process, then run Stable Diffusion locally using `diffusers` + a turbo/LCM checkpoint. Step-by-step output viewable in the notebook. |

---

## Reading Paths

### Path A — "I just want to understand how Stable Diffusion works"

```
MultimodalFoundations → VisionTransformers → CLIP → DiffusionModels → LatentDiffusion
```

### Path B — "I want to run image generation locally from scratch"

```
MultimodalFoundations → DiffusionModels → Schedulers → LatentDiffusion → LocalDiffusionLab
```

### Path C — "I'm already familiar with diffusion, I want to extend to video and multimodal LLMs"

```
CLIP → GuidanceConditioning → TextToImage → TextToVideo → MultimodalLLMs
```

### Path D — "I need to evaluate model outputs for a project"

```
GenerativeEvaluation (read alone — it is largely self-contained)
```

### Full Sequential Path (recommended)

```
MultimodalFoundations
  └─▶ VisionTransformers
        └─▶ CLIP
              └─▶ DiffusionModels
                    └─▶ GuidanceConditioning
                          └─▶ Schedulers
                                └─▶ LatentDiffusion
                                      └─▶ TextToImage
                                            └─▶ TextToVideo
                                                  └─▶ MultimodalLLMs
                                                        └─▶ GenerativeEvaluation
                                                              └─▶ LocalDiffusionLab (capstone)
```

---

## How This Track Connects to the AI Track and ML Track

| Concept from this track | Prerequisite from other tracks |
|------------------------|-------------------------------|
| Patch embeddings in ViT | Transformer architecture → [ML Ch.18 — Transformers](../ML/ch18-transformers/README.md) |
| InfoNCE contrastive loss | Embedding training objectives → [RAGAndEmbeddings.md](../AI/RAGAndEmbeddings/RAGAndEmbeddings.md) |
| CLIP text encoder inside Stable Diffusion | Tokenisation + transformer encoder → [LLMFundamentals.md](../AI/LLMFundamentals/LLMFundamentals.md) |
| CFG conditioning via cross-attention | Attention mechanics → [ML Ch.18 — Transformers](../ML/ch18-transformers/README.md) |
| VAE (encoder-decoder architecture) | Neural network layers + backprop → [ML Ch.4](../ML/ch04-neural-networks/README.md) + [ML Ch.5](../ML/ch05-backprop-optimisers/README.md) |
| Fine-tuning LLaVA on visual instructions | Fine-tuning concepts → [FineTuning.md](../AI/FineTuning/FineTuning.md) |
| FID as a distribution-level metric | Evaluation concepts → [EvaluatingAISystems.md](../AI/EvaluatingAISystems/EvaluatingAISystems.md) |

---

## The Build Plan

> This section tracks the chapter-by-chapter build of the Multimodal AI notes library. Each chapter lives under `notes/MultimodalAI/` in its own folder, containing a `.md` note and a Jupyter notebook. The running example (PixelSmith) progresses through each chapter.

### Chapter Structure

Every note follows this template (same order as the ML and AI tracks):

```
# [Topic] — [Subtitle]

> Blockquote: what you'll understand after reading this

## 1 · Core Idea                   ← 2–4 sentences, plain English
## 2 · Running Example             ← how PixelSmith uses this concept
## 3 · The Math                    ← key equations, every symbol annotated
## 4 · How It Works — Step by Step ← numbered steps or flow diagram
## 5 · The Key Diagrams            ← Mermaid / ASCII diagrams
## 6 · What Changes at Scale       ← how this works in production systems
## 7 · Common Misconceptions       ← what people get wrong
## 8 · Interview Checklist         ← Must Know / Likely Asked / Trap to Avoid
## 9 · What's Next                 ← forward pointer to the next note
```

### Chapter Status

| # | Chapter | Folder | Status |
|---|---------|--------|--------|
| 1 | Multimodal Foundations | `MultimodalFoundations/` | Complete |
| 2 | Vision Transformers | `VisionTransformers/` | Complete |
| 3 | CLIP | `CLIP/` | Complete |
| 4 | Diffusion Models | `DiffusionModels/` | Complete |
| 5 | Guidance & Conditioning | `GuidanceConditioning/` | Complete |
| 6 | Schedulers | `Schedulers/` | Complete |
| 7 | Latent Diffusion | `LatentDiffusion/` | Complete |
| 8 | Text-to-Image | `TextToImage/` | Complete |
| 9 | Text-to-Video | `TextToVideo/` | Complete |
| 10 | Multimodal LLMs | `MultimodalLLMs/` | Complete |
| 11 | Generative Evaluation | `GenerativeEvaluation/` | Complete |
| 12 | Local Diffusion Lab (capstone) | `LocalDiffusionLab/` | Complete |

---

## Hardware Expectations

> No chapter in this track requires a GPU. Every notebook runs on a stock developer laptop.

| Chapter | What runs locally | Typical time on CPU |
|---------|------------------|---------------------|
| 1–3 (Foundations, ViT, CLIP) | Numpy / PyTorch tensor operations | < 30 seconds |
| 4–6 (Diffusion, Guidance, Schedulers) | DDPM training on MNIST | < 5 minutes |
| 7–8 (Latent Diffusion, Text-to-Image) | SDXL-Turbo or LCM via `diffusers` | 30–90 seconds / image |
| 9 (Text-to-Video) | Theory + inspection of pretrained models | No local generation |
| 10 (Multimodal LLMs) | LLaVA-1.5-7B via `ollama` or `llama.cpp` | 10–60 seconds / response |
| 11 (Evaluation) | FID / CLIP score on pre-generated samples | < 2 minutes |
| 12 (Local Diffusion Lab) | Full pipeline: scratch DDPM + local SD | < 10 minutes total |

---

## The PixelSmith Running Example — Master Design

### What PixelSmith Is

PixelSmith is a local AI creative studio. Think of it as a minimal, from-scratch recreation of the core inference pipeline behind tools like DALL-E or Stable Diffusion — built piece by piece as you work through the notes. At the end of the track, the notebook in `LocalDiffusionLab/` ties every component together into a runnable end-to-end pipeline.

### Why This Running Example Works

| Property | Why it matters |
|----------|---------------|
| Fully local | You never need an API key or a cloud account — every chapter runs offline |
| Progressive | Each chapter adds exactly one new concept to the same growing system |
| Grounded | Every abstraction (patch embeddings, latent space, noise schedule) is demonstrated with real tensor operations you can inspect |
| Honest about constraints | A stock machine cannot run Stable Diffusion XL in full float32, so the notes explain *why*, not just *what* |

### What PixelSmith Is Not

PixelSmith is **not** a production image editor, a fine-tuning service, or a LoRA training pipeline. It is a teaching artefact — the simplest system that lets you verify you understand each concept with code you wrote or can read line-by-line.

---

## Prerequisite Check

Before starting Chapter 1, you should be comfortable with:

| Prerequisite | Where to build it if needed |
|-------------|----------------------------|
| What a transformer is and how attention works | [ML Ch.17 — From Sequences to Attention](../ML/ch17-sequences-to-attention/README.md) then [ML Ch.18 — Transformers](../ML/ch18-transformers/README.md) |
| What embeddings are and why cosine similarity matters | [RAGAndEmbeddings.md](../AI/RAGAndEmbeddings/RAGAndEmbeddings.md) |
| Basic PyTorch tensor operations (`torch.Tensor`, `.view()`, matrix multiply) | [ML Ch.4 — Neural Networks](../ML/ch04-neural-networks/README.md) |
| What a convolutional layer does (for comparison with ViT) | [ML Ch.7 — CNNs](../ML/ch07-cnns/README.md) |

You do **not** need: prior experience with image generation, a GPU, or familiarity with `diffusers` / `huggingface_hub` before you start.
