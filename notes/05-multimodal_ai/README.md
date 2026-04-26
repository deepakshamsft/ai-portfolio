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

Image generation went from "blurry 32×32 digits" to "photoreal 4K video" in about a decade. **The detailed timeline now lives in each chapter's own prelude** — every chapter opens with a *"The story"* blockquote that names the papers, dates, and dramatic tensions behind that breakthrough.

**The through-line in one paragraph.** Each chapter corresponds to a specific obstacle that was removed. CNNs dominated vision until [ViT](ch02_vision_transformers) (Dosovitskiy et al., Oct 2020) showed patches + attention scaled better. Text and images lived in separate spaces until [CLIP](ch03_clip) (Radford et al., Jan 2021) aligned them with 400M web pairs. GANs (Goodfellow, 2014) were unstable until [DDPM](ch04_diffusion_models) (Ho et al., Jun 2020) replaced them with stable denoising. Pixel-space diffusion was too expensive until [LatentDiffusion / Stable Diffusion](ch06_latent_diffusion) (Rombach et al., Aug 2022) moved to VAE latents and went open-source. Sampling was slow until [DDIM](ch05_schedulers) (Song et al., Oct 2020) and DPM-Solver (Lu et al., 2022). Models were uncontrollable until [classifier-free guidance](ch07_guidance_conditioning) (Ho & Salimans, Jul 2022) and [ControlNet](ch08_text_to_image) (Zhang, Feb 2023). Images weren't enough → [MLLMs](ch10_multimodal_llms) (BLIP-2 Jan 2023, GPT-4V Sep 2023) and [video](ch09_text_to_video) (AnimateDiff 2023, Sora Feb 2024). Cloud GPUs were too expensive → [LocalDiffusionLab](ch13_local_diffusion_lab) via quantisation + LCM distillation. As quality saturated, [evaluation](ch12_generative_evaluation) shifted from FID (Heusel 2017) to human preference (HPSv2, 2023).

---

## Popular and Powerful Models by Modality (Apr 21, 2026 snapshot)

This is a practical field snapshot (not a leaderboard). It highlights model families most commonly cited for quality and adoption.

| Modality | Popular and powerful model families |
|----------|-------------------------------------|
| Text-to-Image | Midjourney latest, FLUX.1 family (BFL), SDXL/SD3.x ecosystem, Imagen 3 |
| Image Editing / Control | ControlNet-style pipelines, FLUX control/edit variants, SD inpainting stacks |
| Text-to-Video | Sora, Veo 2, Runway latest Gen family, Kling latest, open stacks such as HunyuanVideo and CogVideoX |
| Vision-Language (MLLM) | GPT-4o family, Gemini 2.x family, Claude 3.7+ vision, Qwen2.5-VL, Pixtral, Llama vision variants |
| Speech-to-Text (ASR) | Whisper large-v3 family, distil-whisper variants, Canary family |
| Text-to-Speech (TTS) | ElevenLabs latest voices, gpt-4o-mini-tts, Coqui XTTS v2, MMS-TTS, Kokoro-82M |
| Music / SFX Generation | Suno latest, Udio latest, Stable Audio family, MusicGen family |
| Cross-modal Embeddings / Retrieval | CLIP descendants, SigLIP family, EVA-CLIP style vision-text encoders |

### Local CPU quick-win picks (recommended)

If your goal is "few lines of code + visible output on stock hardware":

1. Text-to-speech: `facebook/mms-tts-eng` or `Kokoro-82M`
2. Speech-to-text: `distil-whisper` or smaller `Whisper` checkpoints
3. Image retrieval demo: CLIP/SigLIP embeddings on a small local image set

For this reason, the new [AudioGeneration](ch11_audio_generation/README.md) chapter uses MMS TTS as the first demonstration.

---

## GPU-Only Supplement Notebooks

Each MultimodalAI chapter now includes a dedicated GPU-only notebook named `notebook_supplement.ipynb`.

- Purpose: keep optional GPU-first experiments separate from the main chapter walkthrough.
- Scope: one `notebook_supplement.ipynb` per chapter under `notes/MultimodalAI/<Chapter>/`.
- Safety guard: every `notebook_supplement.ipynb` checks for GPU availability in the first code cell and exits early with a clear message if no compatible GPU is detected.

### Requirements

- Hardware: NVIDIA GPU with drivers installed.
- Runtime: Python kernel with either:
  - `torch` with CUDA support, or
  - `nvidia-smi` available on `PATH` for fallback detection.

If neither check confirms a GPU, the notebook stops immediately by design.

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
| [MultimodalFoundations.md](ch01_multimodal_foundations/multimodal-foundations.md) | What multimodal AI is; how images, audio, and video become tensors; the representation problem; the modality gap | How does a photograph become a matrix? Why can't we just feed pixels to a transformer? What is a modality gap? |
| [VisionTransformers.md](ch02_vision_transformers/vision-transformers.md) | ViT: splitting images into patches, positional encoding, self-attention over patches; how ViT differs from CNNs and why it won | What is a patch embedding? How does ViT handle position? Why did attention beat convolution at scale? |

### Alignment Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [CLIP.md](ch03_clip/clip.md) | Contrastive Language-Image Pretraining; dual-encoder architecture; InfoNCE loss; zero-shot classification without any labelled data | How does a model learn that a photo of a cat matches the text "a cat"? What is contrastive loss? What is zero-shot transfer? |
| [GuidanceConditioning.md](ch07_guidance_conditioning/guidance-conditioning.md) | Classifier guidance, classifier-free guidance (CFG), text conditioning via cross-attention; what the guidance scale actually does; negative prompts | Why does guidance scale 7.5 produce sharper images than 1.0? What does a negative prompt actually do mechanically? |

### Generation Core Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [DiffusionModels.md](ch04_diffusion_models/diffusion-models.md) | The math of DDPM: the forward noising process, the reverse denoising process, score matching, noise schedules; why diffusion beat GANs | What is the forward process? What does the U-Net actually predict — the image or the noise? Why is diffusion more stable than GAN training? |
| [Schedulers.md](ch05_schedulers/schedulers.md) | DDPM vs DDIM vs DPM-Solver; how to generate in 4 steps instead of 1,000; deterministic sampling; the speed/quality trade-off | Why does DDIM need fewer steps? What changes when you switch from DDPM to DPM-Solver? What is a sampler doing geometrically? |
| [LatentDiffusion.md](ch06_latent_diffusion/latent-diffusion.md) | Why pixel-space diffusion is too slow; VAEs as a compression layer; the Stable Diffusion architecture (VAE + U-Net + CLIP text encoder); latent space geometry | What is a VAE? Why run diffusion in latent space instead of pixel space? How does text reach the U-Net in Stable Diffusion? |

### Application Notes

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [TextToImage.md](ch08_text_to_image/text-to-image.md) | End-to-end text-to-image pipeline; prompt engineering for images; img2img; inpainting; ControlNet for structural conditioning | How does prompt weight syntax work? What is ControlNet's conditioning signal? How does inpainting avoid repainting the whole image? |
| [TextToVideo.md](ch09_text_to_video/text-to-video.md) | Extending diffusion to the temporal dimension; the consistency problem; overview of video generation (Sora, CogVideo, AnimateDiff) | What makes video harder than images? How does Sora model spacetime? What is AnimateDiff doing differently from full video models? |
| [MultimodalLLMs.md](ch10_multimodal_llms/multimodal-llms.md) | Connecting vision encoders to LLM decoders; LLaVA, BLIP-2, GPT-4V, Gemini; visual instruction tuning; the projection layer | How does GPT-4V "see"? What is a Q-Former? How do you fine-tune an LLM to accept image tokens? |
| [AudioGeneration](ch11_audio_generation/README.md) | CPU-first text-to-speech quick win using a compact pretrained model and a minimal notebook flow | How can you generate speech locally without a GPU? What is the shortest path from text prompt to playable waveform? |

### Evaluation Note

| File | Purpose | Key Questions Answered |
|------|---------|------------------------|
| [GenerativeEvaluation.md](ch12_generative_evaluation/generative-evaluation.md) | How do you measure the quality of a generated image or video? FID, IS, CLIP score, LPIPS, human preference models; the alignment problem in evaluation | What does FID actually measure? Why is CLIP score better for text-image alignment than FID? Why is human evaluation still the gold standard? |

### Capstone

| File | Purpose |
|------|---------|
| [LocalDiffusionLab.md](ch13_local_diffusion_lab/local-diffusion-lab.md) | Hands-on capstone: train a DDPM from scratch on MNIST (runs in ~5 minutes on CPU), visualise every step of the diffusion process, then run Stable Diffusion locally using `diffusers` + a turbo/LCM checkpoint. Step-by-step output viewable in the notebook. |

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

### Path E — "I want a fast CPU-only multimodal demo"

```
AudioGeneration (standalone quick win)
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
| Patch embeddings in ViT | Transformer architecture → [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| InfoNCE contrastive loss | Embedding training objectives → [RAGAndEmbeddings.md](.$103-ai/ch04_rag_and_embeddings/rag-and-embeddings.md) |
| CLIP text encoder inside Stable Diffusion | Tokenisation + transformer encoder → [LLMFundamentals.md](.$103-ai/ch01_llm_fundamentals/llm-fundamentals.md) |
| CFG conditioning via cross-attention | Attention mechanics → [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| VAE (encoder-decoder architecture) | Neural network layers + backprop → [ML Ch.4](../ml/03_neural_networks/ch02_neural_networks/README.md) + [ML Ch.5](../ml/03_neural_networks/ch03_backprop_optimisers/README.md) |
| Fine-tuning LLaVA on visual instructions | Fine-tuning concepts → [FineTuning.md](.$103-ai/ch10_fine_tuning/fine-tuning.md) |
| FID as a distribution-level metric | Evaluation concepts → [EvaluatingAISystems.md](.$103-ai/ch08_evaluating_ai_systems/evaluating-ai-systems.md) |

---

## The Build Plan

> This section tracks the chapter-by-chapter build of the Multimodal AI notes library. Each chapter lives under `notes/MultimodalAI/` in its own folder, containing a `.md` note and a Jupyter notebook. The running example (PixelSmith) progresses through each chapter.

Animation rollout tracker:

- [ANIMATION_PLAN.md](animation-plan.md) - data-flow animation coverage and chapter closeout status

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
| 1 | Multimodal Foundations | `ch01_multimodal_foundations/` | Complete |
| 2 | Vision Transformers | `ch02_vision_transformers/` | Complete |
| 3 | CLIP | `ch03_clip/` | Complete |
| 4 | Diffusion Models | `ch04_diffusion_models/` | Complete |
| 5 | Guidance & Conditioning | `ch07_guidance_conditioning/` | Complete |
| 6 | Schedulers | `ch05_schedulers/` | Complete |
| 7 | Latent Diffusion | `ch06_latent_diffusion/` | Complete |
| 8 | Text-to-Image | `ch08_text_to_image/` | Complete |
| 9 | Text-to-Video | `ch09_text_to_video/` | Complete |
| 10 | Multimodal LLMs | `ch10_multimodal_llms/` | Complete |
| 11 | Generative Evaluation | `ch12_generative_evaluation/` | Complete |
| 12 | Local Diffusion Lab (capstone) | `ch13_local_diffusion_lab/` | Complete |
| 13 | Audio Generation (CPU quick win) | `ch11_audio_generation/` | Complete |

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
| 13 (Audio Generation) | MMS TTS inference via `transformers` on CPU | 5–20 seconds / sample |

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
| What a transformer is and how attention works | [ML Ch.17 — From Sequences to Attention](../ml/03_neural_networks/ch09_sequences_to_attention/README.md) then [ML Ch.18 — Transformers](../ml/03_neural_networks/ch10_transformers/README.md) |
| What embeddings are and why cosine similarity matters | [RAGAndEmbeddings.md](.$103-ai/ch04_rag_and_embeddings/rag-and-embeddings.md) |
| Basic PyTorch tensor operations (`torch.Tensor`, `.view()`, matrix multiply) | [ML Ch.4 — Neural Networks](../ml/03_neural_networks/ch02_neural_networks/README.md) |
| What a convolutional layer does (for comparison with ViT) | [ML Ch.7 — CNNs](../ml/03_neural_networks/ch05_cnns/README.md) |

You do **not** need: prior experience with image generation, a GPU, or familiarity with `diffusers` / `huggingface_hub` before you start.
