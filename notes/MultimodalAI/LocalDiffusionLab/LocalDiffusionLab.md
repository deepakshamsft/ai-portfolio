# Local Diffusion Lab — Assembling the Full Pipeline

> **Track:** Multimodal AI 
> **Prerequisites:** All 11 preceding chapters

> **The story.** As recently as **2022**, running Stable Diffusion required a 16 GB-VRAM server GPU. **AUTOMATIC1111**'s WebUI (August 2022) was the first community tool to make local generation accessible. **diffusers** (Hugging Face, summer 2022) provided a clean Python API. **xFormers** memory-efficient attention (Meta, 2022) and **bitsandbytes** 8-bit (Tim Dettmers, 2022) collapsed VRAM requirements. **ComfyUI** (comfyanonymous, January 2023) let users wire pipelines as node graphs. **GGUF** quantisation (the llama.cpp lineage, 2023–24) and the **MLX** framework (Apple, December 2023) brought 4-bit diffusion to Mac M-series and 8 GB consumer GPUs. By 2025, the open generative-AI stack was something you could assemble on a laptop and run offline — a remarkable retreat of the cloud-only assumption that defined 2022.
>
> **Where you are in the curriculum.** You've built every conceptual piece across the previous 11 chapters. This is the assembly chapter — wiring CLIP + VAE + U-Net + scheduler + guidance + ControlNet into a working pipeline that runs entirely on a stock developer laptop. After this, the [PixelSmith](../README.md) studio is real, and you have the production pattern for any locally-hosted generative system.

---

## 1 · Core Idea

A **local diffusion lab** is a complete, offline-first AI image studio that runs on consumer hardware. No cloud API required. The twelve chapters of this track are its blueprint:

```
Input text / image
 ↓
[CLIP text encoder] ← Ch. 3 · CLIP
 ↓
[Classifier-free guidance] ← Ch. 5 · GuidanceConditioning
 ↓
[DDIM reverse diffusion] ← Ch. 6 · Schedulers
 ↓
[Latent space denoising] ← Ch. 7 · LatentDiffusion
 ↓ (optionally conditioned on edges/depth)
[ControlNet residuals] ← Ch. 8 · TextToImage
 ↓
[VAE decoder] ← Ch. 7 · LatentDiffusion
 ↓
Output image
 ↓
[Evaluation: FID / CLIP Score] ← Ch. 11 · GenerativeEvaluation
```

The video pipeline (Ch. 9) adds a temporal attention layer at the denoising loop. The MLLM (Ch. 10) wraps the whole thing so you can *chat* with your generated images.

---

## 2 · Running Example

**PixelSmith v0 → v6 — full retrospective**

| Chapter | PixelSmith version | New capability |
|---------|-------------------|----------------|
| 1 · MultimodalFoundations | v0 | Architecture overview; file ingestion (CLIP, DDPM, ViT) |
| 2 · VisionTransformers | v1 | Vision-Transformer image encoder replaces CNN |
| 3 · CLIP | v2 | Text–image alignment; zero-shot retrieval |
| 4 · DiffusionModels | v3 | Unconditional DDPM generation |
| 5 · GuidanceConditioning | v3.5 | Class-conditional CFG; guidance scale knob |
| 6 · Schedulers | v3.5+ | DDIM sampler → 10× speed-up; deterministic mode |
| 7 · LatentDiffusion | v4 | Latent-space diffusion; VAE compression |
| 8 · TextToImage | v5 | Edge-conditioned ControlNet; prompt/negative-prompt |
| 9 · TextToVideo | v5.5 | Temporal attention; frame-consistent video |
| 10 · MultimodalLLMs | v6 | LLaVA-style "describe this image" interface |
| 11 · GenerativeEvaluation | — | FID / CLIP Score automated quality gate |
| 12 · LocalDiffusionLab | — | Everything orchestrated together |

---

## 3 · The Math

No new mathematics in this chapter. The capstone assembles results from previous chapters:

| Component | Mathematical object | Chapter |
|-----------|---------------------|---------|
| Patch embedding | $z_i = W_p \cdot p_i + e_i^{\text{pos}}$ | 2 |
| Contrastive loss | $\mathcal{L} = -\log \frac{e^{\text{sim}(v_i,t_i)/\tau}}{\sum_j e^{\text{sim}(v_i,t_j)/\tau}}$ | 3 |
| Forward diffusion | $q(x_t\|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ | 4 |
| CFG score | $\tilde{\epsilon} = \epsilon_\theta(x_t,\varnothing) + w[\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)]$ | 5 |
| DDIM step | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sigma_t\epsilon + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta$ | 6 |
| VAE ELBO | $\mathcal{L}_{\text{VAE}} = \mathbb{E}[\log p(x\|z)] - \beta D_{\text{KL}}(q\|p)$ | 7 |
| FID | $\text{FID} = \|\mu_r-\mu_g\|^2 + \text{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ | 11 |

---

## 4 · How It Works — Step by Step

### What runs locally on CPU/consumer GPU

| Component | VRAM / RAM | Time per image | Recommended tool |
|-----------|-----------|----------------|-----------------|
| CLIP text encoding | < 1 GB | < 100 ms | `open_clip` |
| DDIM 20 steps (MNIST) | CPU | ~2 s | Your Ch.4/6 code |
| Latent DDIM 20 steps 512px | 4 GB | 5–10 s | `diffusers` + SDXL-Turbo |
| ControlNet | 6–8 GB | 8–15 s | `diffusers` |
| LLaVA 7B inference | 8 GB | 3–5 s | `ollama` |
| LLaVA 34B inference | 20 GB | 15–30 s | `ollama` |

### Building a Full Local Pipeline

1. **Text in → CLIP encode** → 512-dim text embedding `c`.
2. **Sample latent** $z_T \sim \mathcal{N}(0, I)$.
3. **DDIM reverse loop** (20 steps) with CFG: `z_{t-1} = ddim_step(ε_θ, z_t, c)`.
4. **VAE decode** `z_0` → pixel image $\hat{x}_0$.
5. **ControlNet** (optional): inject edge map as spatial condition at step 3.
6. **Evaluate**: compute `CLIP Score(ĉ_image, c_text)`, optionally FID over batch.
7. **LLaVA caption** (optional): feed $\hat{x}_0$ to MLLM → natural-language description.

---

## 5 · The Key Diagrams

```
PIXELSMITH v6 — FULL ARCHITECTURE
──────────────────────────────────

 Prompt: "a handwritten four"
 │
 ▼
 ┌─────────────┐ ┌──────────────────────────────────┐
 │ CLIP Text │ │ Latent Diffusion Loop (DDIM) │
 │ Encoder │ │ │
 │ c ∈ R^512 │─────▶│ z_T ~ N(0,I) │
 └─────────────┘ │ for t = T..1: │
 │ ε̃ = CFG(ε_θ(z_t,c), ε_θ(z_t,∅))│
 Optional: │ z_{t-1} = DDIM_step(z_t, ε̃) │
 ┌─────────────┐ │ z_0 obtained │
 │ Edge map │──────▶│ (ControlNet injects residuals) │
 └─────────────┘ └───────────────┬──────────────────┘
 │
 ▼
 ┌─────────────────┐
 │ VAE Decoder │
 │ z_0 → x̂_0 │
 └────────┬────────┘
 │
 ┌──────────────────┴───────────────────┐
 │ │
 ▼ ▼
 ┌──────────────┐ ┌──────────────────┐
 │ LLaVA MLLM │ │ Eval: FID / │
 │ "What digit │ │ CLIP Score / │
 │ is this?" │ │ Precision-Recall │
 └──────────────┘ └──────────────────┘
```

---

## 6 · What Changes at Scale

| Local lab | Production |
|-----------|------------|
| MNIST 28px, DDPM/DDIM | SDXL 1024px, DPM-Solver++ |
| Linear CFG | Advanced prompt weighting, compel library |
| Single ControlNet (Canny) | ControlNet stack (depth + pose + canny) |
| LLaVA 7B | GPT-4o / Claude 3 Vision |
| Manual FID at end of training | CI metric gate (auto-fail if FID regresses) |
| Single GPU | Multi-GPU distillation (LCM, Turbo) |

### Ecosystem Tools

| Tool | Purpose |
|------|---------|
| **Automatic1111 WebUI** | Most popular local T2I UI; supports 1000+ extensions |
| **ComfyUI** | Node-based workflow editor; great for ControlNet pipelines |
| **Invoke AI** | Professional-grade local studio |
| **diffusers (HuggingFace)** | Production Python API for any diffusion model |
| **ollama** | One-command local LLM / MLLM serving |
| **open_clip** | Open-source CLIP training and inference |

---

## 7 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "You need a cloud GPU to run Stable Diffusion" | SDXL-Turbo runs in ~10 s on a 4 GB VRAM GPU; SDXL Lite runs on CPU |
| "More DDIM steps always looks better" | Past 20–50 steps, gains are invisible; quality plateaus |
| "ControlNet only works with Canny edges" | ControlNet has depth, pose, normal, scribble, and segmap variants |
| "FID < 10 means the model is great" | FID measures *match to training data*, not human aesthetic preference |
| "Local models are behind GPT-4V in all tasks" | Open-source LLaVA-1.6 34B matches or exceeds GPT-4V on many benchmarks |
| "VAE is the bottleneck in Stable Diffusion" | The denoising U-Net/DiT dominates runtime; VAE is <5% of total |

---

## 8 · Interview Checklist

### Must Know
- The full T2I pipeline: CLIP encode → latent DDIM → VAE decode.
- Key speed-up levers: DDIM (fewer steps), latent space (smaller feature maps), LCM/Turbo distillation.
- CFG: two forward passes per step, guidance scale w, why w>1 improves quality but hurts diversity.
- FID as the standard quality metric; its N-sample bias.

### Likely Asked
- "Walk me through generating an image from a text prompt with Stable Diffusion."
- "What is the role of the VAE in latent diffusion?"
- "How does ControlNet inject spatial conditioning?"
- "How would you systematically evaluate a new generative model?"
- "What's the trade-off between guidance scale and diversity?"

### Traps to Avoid
- Saying "Stable Diffusion runs the denoiser in pixel space" — it runs in **latent** space.
- Confusing CLIP's training objective (contrastive) with the diffusion objective (score matching).
- Overlooking that DDIM's speed-up comes from **skipping time steps**, not faster forward passes.
- Conflating LoRA (parameter-efficient fine-tuning) with textual inversion (token-based fine-tuning).

---

## 9 · What's Next

You've completed the **12-chapter Multimodal AI** track. Suggested next steps:

| Path | Description |
|------|-------------|
| **Fine-tune Stable Diffusion** | DreamBooth or LoRA on your own images |
| **Deploy an API** | Wrap `diffusers` in FastAPI; stream tokens via Server-Sent Events |
| **RL from human feedback** | RLHF for image generation using reward models (HPSv2, ImageReward) |
| **Multimodal RAG** | Combine CLIP retrieval (Ch. 3) with LLaVA generation (Ch. 10) |
| **Video generation** | Fine-tune AnimateDiff or CogVideoX on domain-specific video |
| **Distillation** | LCM or SDXL-Turbo-style consistency distillation for 1-step generation |

> "The best way to understand diffusion is to implement it — which you just did."

## Illustrations

![Local diffusion lab - pipeline, VRAM budget, latency, quality/speed knobs](img/Local%20Diffusion%20Lab.png)
