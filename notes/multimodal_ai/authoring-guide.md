# Authoring Guide — Multimodal AI Grand Challenge

> **Purpose**: This guide defines the template for authoring Multimodal AI chapters using the **VisualForge Studio** Grand Challenge as the unifying thread.

---

## The Grand Challenge — VisualForge Studio

**VisualForge Studio** is a boutique creative agency replacing $600k/year freelancer costs with an in-house AI system that generates professional-grade marketing visuals, runs entirely on local hardware, and delivers <30 seconds per image for rapid client iterations.

**6 constraints to track**:

| # | Constraint | Target | Status After Ch.12 |
|---|------------|--------|-------------------|
| #1 | **QUALITY** | ≥4.0/5.0 professional quality score | ✅ 4.1/5.0 (HPSv2) |
| #2 | **SPEED** | <30 seconds per 512×512 image | ✅ 8 seconds (SDXL-Turbo) |
| #3 | **COST** | <$5,000 hardware + $0/month cloud | ✅ $2,500 laptop, no cloud |
| #4 | **CONTROL** | <5% unusable generations | ✅ 3% unusable (ControlNet) |
| #5 | **THROUGHPUT** | 100+ images/day capacity | ✅ 120 images/day |
| #6 | **VERSATILITY** | Text→Image + Video + Understanding | ✅ All 3 modalities |

**Final outcome**: $600k/year savings, 2.5-month payback, 40× faster turnaround, 8× throughput increase.

---

## Chapter Structure Template

Every chapter follows this template:

```markdown
# [Topic] — [Subtitle]

> **The story.** [Historical context of the technique]
>
> **Where you are in the curriculum.** [What you learned before, what this chapter adds]

---

## 0 · The VisualForge Studio Challenge

[COPY FROM § 0 TEMPLATE BELOW]

---

## 1 · Core Idea

[Technical concept]

---

## 2 · Running Example — PixelSmith vX

[Code example showing technique in action]

---

## 3 · The Math

[Equations, derivations]

---

## 4 · Visual Intuition

[Diagrams showing how technique works]

---

## 5 · Production Example — VisualForge in Action

[Show how VisualForge uses this technique to move closer to constraints]

---

## 6 · Common Failure Modes

[What goes wrong, how to debug]

---

## 7 · When to Use This vs Alternatives

[Decision framework]

---

## 8 · Connection to Prior Chapters

[How this builds on previous chapters]

---

## 9 · Interview Checklist

[Key questions employers ask]

---

## 10 · Further Reading

[Papers, repos, tutorials]

---

## 11 · Notebook

[Link to executable notebook]

---

## 11.5 · Progress Check — What Have We Unlocked?

[COPY FROM PROGRESS CHECK TEMPLATE BELOW]

---

## Bridge to Chapter [X]

[What's still blocking, teaser for next chapter]
```

---

## § 0 Template — The Challenge Section

Place this **after "The story"** block and **before "1 · Core Idea"**.

```markdown
## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge Studio needs to replace $600k/year freelancer costs with an in-house AI system running on local hardware (<$5k), delivering professional-grade marketing visuals (<30s per image, ≥4.0/5.0 quality), with <5% unusable generations and 100+ images/day throughput. The system must handle text→image, image→video, and image understanding for automated QA.

**Current blocker at Chapter [X]**: [Specific technical problem preventing progress]

**What this chapter unlocks**: [The technique introduced here and how it moves us forward]

---

### The 6 Constraints — Snapshot After This Chapter

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | [Symbol + Score] | [How measured] |
| #2 Speed | <30 seconds | [Symbol + Time] | [What hardware] |
| #3 Cost | <$5k hardware | [Symbol + $] | [What changed] |
| #4 Control | <5% unusable | [Symbol + %] | [What technique enabled this] |
| #5 Throughput | 100+ images/day | [Symbol + Count] | [What bottleneck removed] |
| #6 Versatility | 3 modalities | [Symbol + Which] | [What capability unlocked] |

**Symbols**:
- ❌ = Blocked (constraint not addressed yet)
- ⚡ = Foundation laid (partial progress, not at target yet)
- ✅ = Target hit (constraint fully satisfied)

---

### What's Still Blocking Us After This Chapter?

[Specific technical/business problem that next chapter will solve]

---
```

---

## Progress Check Template — § 11.5

Place this **after § 11 Notebook** and **before "Bridge to Chapter [X]"**.

```markdown
## 11.5 · Progress Check — What Have We Unlocked?

### Before This Chapter
- [Constraint #X]: [Previous state] (e.g., "5 minutes per image")
- [Constraint #Y]: [Previous state] (e.g., "No structural control")

### After This Chapter
- [Constraint #X]: ✅ [New state] (e.g., "30 seconds per image with DDIM")
- [Constraint #Y]: ⚡ [New state] (e.g., "Can condition on text prompts")

---

### Key Wins

1. **[Win 1]**: [Specific improvement] (e.g., "DDIM reduces steps from 1000 → 50")
2. **[Win 2]**: [Specific improvement] (e.g., "Latent diffusion runs on laptop CPU")
3. **[Win 3]**: [Specific improvement] (e.g., "ControlNet guarantees composition")

---

### What's Still Blocking Production?

[Specific remaining problems that next chapter addresses]

**Next unlock**: [Preview next chapter's solution]

---

### VisualForge Status — Full Constraint View

[Include diagram showing constraint progression across chapters]

| Constraint | Ch.1 | Ch.2 | Ch.3 | ... | This Ch. | Target |
|------------|------|------|------|-----|----------|--------|
| Quality | ❌ | ❌ | ⚡ | ... | ⚡ 3.8/5.0 | 4.0/5.0 |
| Speed | ❌ | ❌ | ❌ | ... | ✅ 8s | <30s |
| ... | ... | ... | ... | ... | ... | ... |

---
```

---

## Key Principles

### 1. Every Code Example Uses VisualForge Context

**❌ Bad** (generic example):
```python
# Generate an image
prompt = "a cat"
image = pipeline(prompt).images[0]
```

**✅ Good** (VisualForge context):
```python
# VisualForge: Generate client brief — "modern office interior with natural light"
client_brief = "modern office interior, natural light, minimalist, professional photography"
image = visualforge_pipeline(client_brief).images[0]
# QA check: verify against brief before sending to client
```

### 2. Every Math Equation Connects to the Constraint

**❌ Bad**: "Here is the InfoNCE loss: $\mathcal{L} = ...$"

**✅ Good**: "The InfoNCE loss maximizes similarity between paired (image, text) → enables **Constraint #4 (Control)**: we can now condition generation on text descriptions of the desired output."

### 3. Metrics Are Specific and Measurable

**❌ Bad**: "Generation is now faster."

**✅ Good**: "DDIM reduces generation time from **5 minutes** (1000 steps) to **30 seconds** (50 steps) → moving toward **Constraint #2** (<30s target)."

### 4. Diagrams Show Progress Toward Constraints

Every chapter should include:
- **Before/After diagram**: Visual proof of improvement (e.g., 5min → 30s timeline)
- **Architecture diagram**: How this component fits in full VisualForge pipeline
- **Bottleneck visualization**: What was blocking, what's unblocked now

---

## Constraint Progression Across Chapters

| Chapter | Quality | Speed | Cost | Control | Throughput | Versatility |
|---------|---------|-------|------|---------|------------|-------------|
| Ch.1 Foundations | ❌ | ❌ | ❌ | ❌ | ❌ | ⚡ Can load images |
| Ch.2 ViT | ❌ | ❌ | ❌ | ❌ | ❌ | ⚡ Image embeddings |
| Ch.3 CLIP | ❌ | ❌ | ❌ | ⚡ Text conditioning | ❌ | ⚡ Text-image search |
| Ch.4 Diffusion | ⚡ 3.0/5.0 | ❌ 5min | ❌ | ⚡ | ❌ | ⚡ Can generate |
| Ch.5 Schedulers | ⚡ 3.2/5.0 | ⚡ 30-60s | ❌ | ⚡ | ❌ | ⚡ |
| Ch.6 Latent Diff | ⚡ 3.5/5.0 | ✅ 20s | ✅ $2.5k laptop | ⚡ | ❌ | ⚡ Text→Image |
| Ch.7 Guidance | ⚡ 3.8/5.0 | ✅ 20s | ✅ | ⚡ <15% unusable | ❌ | ⚡ |
| Ch.8 TextToImage | ⚡ 3.8/5.0 | ✅ 18s | ✅ | ✅ 3% unusable | ⚡ 80/day | ⚡ |
| Ch.9 TextToVideo | ⚡ 3.8/5.0 | ✅ 18s | ✅ | ✅ | ⚡ 85/day | ⚡ Video enabled |
| Ch.10 MultimodalLLM | ⚡ 3.9/5.0 | ✅ 18s | ✅ | ✅ | ✅ 120/day | ✅ All 3 modalities |
| Ch.11 Evaluation | ✅ 4.1/5.0 | ✅ 18s | ✅ | ✅ | ✅ | ✅ |
| Ch.12 LocalLab | ✅ 4.1/5.0 | ✅ 8s | ✅ | ✅ | ✅ | ✅ |

**Legend**:
- ❌ = Not yet addressed
- ⚡ = Foundation laid / partial progress
- ✅ = Target hit

---

## Example — Ch.4 Diffusion Models § 0 Section

```markdown
## 0 · The VisualForge Studio Challenge

**Mission**: VisualForge Studio needs to replace $600k/year freelancer costs with an in-house AI system running on local hardware (<$5k), delivering professional-grade marketing visuals (<30s per image, ≥4.0/5.0 quality), with <5% unusable generations and 100+ images/day throughput.

**Current blocker at Chapter 4**: We can search existing images (Ch.3 CLIP) but **cannot generate new images**. Freelancers create custom visuals; we need generative capability to compete.

**What this chapter unlocks**: **Diffusion models** — learn to generate entirely new images by reversing a noise-injection process. We'll train a U-Net to denoise random Gaussian noise into coherent images.

---

### The 6 Constraints — Snapshot After Chapter 4

| Constraint | Target | Status | Evidence |
|------------|--------|--------|----------|
| #1 Quality | ≥4.0/5.0 | ⚡ **3.0/5.0** | DDPM generates coherent MNIST digits (proof-of-concept) |
| #2 Speed | <30 seconds | ❌ **~5 minutes** | 1000 denoising steps on laptop CPU |
| #3 Cost | <$5k hardware | ❌ Not validated | Haven't tested on target hardware yet |
| #4 Control | <5% unusable | ⚡ **~40% unusable** | Random sampling, no text conditioning |
| #5 Throughput | 100+ images/day | ❌ **~10 images/day** | Limited by 5-minute generation time |
| #6 Versatility | 3 modalities | ⚡ **Text→Image partial** | Can generate images, but not from text descriptions |

---

### What's Still Blocking Us After This Chapter?

**Speed**: 5 minutes per image is unusable for client calls. Need <30 seconds for real-time iteration.

**Next unlock (Ch.5)**: **Schedulers** (DDIM, DPM-Solver) reduce steps from 1000 → 50, achieving 30-60 second generation.

---
```

---

## Checklist Before Publishing a Chapter

- [ ] **§ 0 Challenge section** present (after story, before Core Idea)
- [ ] **§ 11.5 Progress Check** present (after Notebook, before Bridge)
- [ ] **All code examples** use VisualForge context (not generic examples)
- [ ] **Constraint table** shows measurable progress (specific numbers, not vague improvements)
- [ ] **Diagrams** included:
  - [ ] Before/After comparison (visual proof of improvement)
  - [ ] Architecture diagram (how component fits in pipeline)
  - [ ] Bottleneck visualization (what was blocking, what's unblocked)
- [ ] **Metrics** are specific (e.g., "5 min → 30s" not "faster")
- [ ] **"What's still blocking"** section clearly identifies next problem
- [ ] **Notebook** runs successfully and generates VisualForge-relevant output

---

## FAQ

**Q: Do all 12 chapters need the Grand Challenge?**
Yes. Every chapter should have § 0 (Challenge) and § 11.5 (Progress Check). The constraint progression is the narrative backbone.

**Q: What if my chapter doesn't directly impact a constraint?**
Frame it as "foundation laid" (⚡) — e.g., Ch.2 ViT enables image embeddings (needed for CLIP in Ch.3, which enables text conditioning for Constraint #4).

**Q: Can examples be educational (e.g., MNIST) or must they be VisualForge-specific?**
Educational examples are fine in § 2 (Running Example). But § 5 (Production Example) MUST show VisualForge usage.

**Q: How do I measure Quality before Ch.11 (Evaluation)?**
Use proxies: "DDPM generates coherent digits (proof-of-concept)" → "CFG improves prompt adherence (fewer retries)" → "HPSv2 score 4.1/5.0 (Ch.11)".

**Q: What if I'm unsure how my chapter fits the constraints?**
Check [GRAND_CHALLENGE_PROPOSAL.md](GRAND_CHALLENGE_PROPOSAL.md) for the full chapter-by-chapter breakdown.

---

## Reference — Full VisualForge Constraint Table

| # | Constraint | Target | Rationale |
|---|------------|--------|-----------|
| #1 | **QUALITY** | ≥4.0/5.0 professional quality score | Freelancer baseline: 4.2/5.0. Must match to maintain agency reputation. |
| #2 | **SPEED** | <30 seconds per 512×512 image | Clients expect real-time iteration during review calls. 5-min = unusable. |
| #3 | **COST** | <$5,000 hardware + $0/month cloud | Freelancers = $50k/mo. Local hardware amortizes over 24 months. Cloud GPU = $2k/mo min. |
| #4 | **CONTROL** | <5% unusable generations | Random outputs waste time. Need structural control to hit brief first try. |
| #5 | **THROUGHPUT** | 100+ images/day capacity | Current: 15 images/day (3 designers × 5). Need 7× capacity for growth. |
| #6 | **VERSATILITY** | Text→Image + Image→Video + Image Understanding | Clients need hero images, 15s video ads, and QA verification workflow. |

---

## Diagrams to Include in Each Chapter

### 1. Constraint Progression Chart (Radar/Bar Chart)

Show before/after state for each constraint:
- X-axis: 6 constraints
- Y-axis: % toward target
- Two bars: "Before Ch.X" vs "After Ch.X"

### 2. Architecture Diagram

Show where this component fits in full VisualForge pipeline:
```
Client Brief (text)
  ↓
[CLIP Text Encoder] ← Ch.3
  ↓
[U-Net Denoiser] ← Ch.4-7
  ↓
[VAE Decoder] ← Ch.6
  ↓
Generated Image (512×512)
  ↓
[VLM QA] ← Ch.10
  ↓
Auto-approved / Flagged for review
```

Highlight the current chapter's component.

### 3. Timeline Diagram (Speed Improvements)

Show generation time progression:
```
Ch.4 DDPM:        |████████████████████████████████| 5 minutes
Ch.5 DDIM:        |██████| 30 seconds
Ch.6 Latent Diff: |████| 20 seconds
Ch.12 SDXL-Turbo: |█| 8 seconds
                  ↑
                  Target: <30s
```

### 4. Quality Progression (Line Chart)

Show quality score improving across chapters:
```
  5.0 ┤                              ┌──✅ 4.1
      │                         ┌────┘
  4.0 ┤─────────────────────────┤ Target
      │                    ┌────┘
  3.0 ┤────────────────────┘
      │               ┌────┘
  2.0 ┤───────────────┘
      └───────────────────────────────
       Ch.4  Ch.6  Ch.8  Ch.10  Ch.11
```

---

## Final Reminder

The Grand Challenge is not a distraction from learning — **it is the motivation**. Readers want to know: "Why does this math matter? What can I build with it?" VisualForge Studio shows them: $600k/year savings, 40× faster turnaround, 2.5-month payback. That's the story they'll remember.

---

## Style Ground Truth — Multimodal AI Track

> **LLM instruction:** Before authoring or reviewing any chapter in this track, treat the universal [notes/authoring_guidelines.md](../authoring-guidelines.md) as the base reference and the additional track-specific rules below as overrides/extensions.

---

### Voice and Register

The reader is the **Lead ML Engineer at VisualForge Studio**. They are replacing $600k/year of freelancer costs with a local AI pipeline — every second of generation time and every unusable image matters to the bottom line. Every section should feel like: "Here's the generation failure, here's the diffusion mechanism that fixes it, here's how it moves our quality/speed metrics."

**Second person default:**
> *"You're the Lead ML Engineer at VisualForge Studio. The client wants 20 variations of a spring collection hero image by tomorrow morning. Your diffusion pipeline is generating blurry, prompt-agnostic noise."*  
> *"Guidance scale 1.0 — the model ignores your text prompt entirely. You can watch it happen, epoch by epoch."*  
> *"Your RTX 4090 is generating one image every 47 seconds. The client needs 100 variations. You have 90 minutes."*

**Dry humour register:** one instance per major concept.
> *"Unconditioned diffusion produces beautiful, meaningless noise. A human child could do the same. The client is not paying for noise."*

---

### Failure-First Pattern — VisualForge Edition

Every multimodal concept is introduced through a **specific VisualForge generation failure**:

| Chapter | The Failure | The Fix | What Breaks Next |
|---------|-------------|---------|-----------------|
| Ch.1 Multimodal Foundations | Raw pixel generation: mode collapse → all outputs similar | Understand embedding spaces + cross-modal alignment | Can't generate from text yet |
| Ch.2 CLIP | Text and image embeddings unaligned; "red car" retrieves blue trucks | CLIP contrastive pretraining: text↔image similarity space | Good retrieval but can't generate |
| Ch.3 Diffusion Models | Gaussian noise → random output; no structure | Forward (add noise) + reverse (denoise) process | No text guidance yet |
| Ch.4 Latent Diffusion | Full pixel-space diffusion: 47s/image on RTX 4090 | VAE latent space: 8× compression → 6s/image ✅ Speed target hit | Prompt adherence weak |
| Ch.5 Guidance & Conditioning | CFG scale 1.0 → model ignores text prompt | Classifier-free guidance: scale 7.5 → prompt-responsive | Hard to steer precisely |
| Ch.6 Schedulers | 1,000 DDPM steps → 2min/image; too slow for iteration | DDIM/DPM++ 20 steps → 8s/image ✅ Speed confirmed | Fine control lacking |
| Ch.7 Text-to-Image | Generated images lack fine details; brand inconsistency | SDXL + LoRA fine-tuning on VisualForge style guide | Can't handle reference images |
| Ch.8 Vision Transformers | CNN features miss global composition | ViT attention: global patch relationships → better composition | Video generation not yet possible |
| Ch.9 Multimodal LLMs | Image understanding without generation: "describe this product photo" | LLaVA / GPT-4V for image→text grounding | Combining image+text→image still hard |
| Ch.10 Local Diffusion Lab | Environment fragmented; reproducibility issues | Docker + ComfyUI workflow → reproducible pipeline ✅ | Evaluation is subjective |
| Ch.11 Generative Evaluation | "Looks good to me" not scalable at 120 images/day | FID / HPSv2 / CLIP score automated evaluation | Video needs dedicated treatment |
| Ch.12 Text-to-Video | Static images can't show product demos | AnimateDiff / temporal attention → 24fps clips | 🎉 All 6 constraints met |

---

### Running Example — VisualForge Campaign Types

Anchor every technical concept to one of these canonical VisualForge client requests:

| Campaign type | Brief | Why it's hard |
|--------------|-------|--------------|
| Spring collection hero | "Woman in floral dress, Parisian café, golden hour, editorial" | Photorealism + accurate clothing + consistent lighting |
| Product demo | "Our running shoe in 5 colorways on a clean white background" | Exact product geometry + colour fidelity × 5 variations |
| Brand pattern | "Abstract repeat pattern, coral + navy, Moroccan influence, seamless tile" | Seamless tiling + colour accuracy + cultural style |
| Before/after | "Kitchen renovation: same room, before (dated) and after (modern)" | Spatial consistency + realistic transformation |
| Product lifestyle | "Coffee machine in a cozy home office, natural light, aspirational" | Scene composition + product placement + brand feel |

**Every worked example must use one of these brief types.** Generic "a photo of a dog" examples are not acceptable — they don't test the constraints VisualForge actually faces.

---

### Mathematical Moments — Multimodal Track

| Concept | What to show | How to present it |
|---------|-------------|------------------|
| Forward diffusion | `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)` | Show numerically: x_0 → x_1 → ... → x_T with β schedule |
| Reverse denoising | `p_θ(x_{t-1} | x_t)` as learned Gaussian | Show: model predicts the noise, then we subtract it |
| CFG formula | `ε̂ = ε_uncond + s(ε_cond - ε_uncond)` | Walk through s=1 (no guidance), s=7.5 (balanced), s=15 (over-sharp) |
| CLIP similarity | `cos_sim(text_embed, img_embed)` | Show 3 prompts × 3 images in a 3×3 similarity table |
| Latent compression | `z = E(x)` → 512×512×3 → 64×64×4 (8× reduction) | ASCII diagram showing spatial dimensions shrinking |
| FID calculation | Fréchet distance between real and generated feature distributions | Numerical: FID=5 (good) vs FID=80 (bad), with visual examples |

**Scalar-first rule applies:** show the 1D noise schedule before the full 2D image tensor. Show CLIP similarity for two 4-dimensional toy embeddings before mentioning 768 dimensions.

---

### ASCII Diagram Style — Multimodal Track

Use ASCII diagrams for diffusion pipelines (replacing matrix diagrams from the ML track):

```
Forward process (adds noise):
x_0 ──(+ε_1)──→ x_1 ──(+ε_2)──→ x_2 ──···──→ x_T ≈ N(0, I)
[clean image]                                    [pure noise]

Reverse process (removes noise, guided by text):
x_T ──(−ε̂_T)──→ x_{T-1} ──(−ε̂_{T-1})──→ ··· ──→ x_0
[pure noise]  [text: "floral dress, café"]       [clean image]

UNet at each step:
         ┌──────────────────────────┐
x_t ──→  │  UNet(x_t, t, text_emb)  │ ──→ predicted noise ε̂
         └──────────────────────────┘
                    ↑
             text embedding
             from CLIP encoder
```

---

### Callout Box Conventions — MultimodalAI Track

| Symbol | Typical use in MultimodalAI track |
|--------|----------------------------------|
| `💡` | "CFG scale controls the prompt adherence / diversity tradeoff. Scale 7.5 is the empirically tested sweet spot for photorealistic images. Going above 15 introduces over-saturation and artifacts." |
| `⚠️` | "Never evaluate image quality by looking at 5 examples. At 120 images/day, cognitive fatigue makes human review unreliable after the first hour. Use automated metrics." |
| `⚡` | Constraint achievement: "8s/image → Constraint #2 SPEED ✅ ACHIEVED (vs 30s target)" |
| `📖` | DDPM mathematical derivation, ELBO derivation for diffusion training loss, FID calculation from Fréchet distance formula |
| `➡️` | "We're treating the CLIP text encoder as a black box here. Ch.2 opens it — the contrastive training objective is what makes 'floral dress' and [image of floral dress] end up in the same embedding neighbourhood." |

---

### Code Style — MultimodalAI Track

**Standard inference setup:**
```python
import torch
from diffusers import StableDiffusionXLPipeline

# Production: full pipeline setup
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
pipe.enable_model_cpu_offload()  # fits 12GB+ VRAM; remove if VRAM > 16GB
```

**Variable naming conventions:**
| Variable | Meaning |
|----------|---------|
| `prompt` | Positive text prompt string |
| `negative_prompt` | Negative prompt string ("blurry, low quality, ...") |
| `guidance_scale` | CFG scale (float, typically 5–12) |
| `num_inference_steps` | Denoising steps (int, 20–50 for quality) |
| `image` | `PIL.Image` output |
| `latent` | Latent tensor `(1, 4, H/8, W/8)` |
| `noise` | Gaussian noise tensor |
| `t` | Current diffusion timestep (int, 0–999) |
| `clip_score` | CLIP image-text alignment score (float, 0–1) |
| `fid_score` | Fréchet Inception Distance (float, lower is better) |
| `gen_time_s` | Generation time in seconds |

**Educational vs Production labels:**
```python
# Educational: manual denoising loop (shows the mechanism)
for t in scheduler.timesteps:
    noise_pred = unet(latent, t, text_embeddings).sample
    latent = scheduler.step(noise_pred, t, latent).prev_sample

# Production: pipeline call (handles all the above)
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
```

---

### GPU Supplement Notebook Conventions

Chapters with GPU-intensive code must include a `notebook_supplement.ipynb` with:

1. **GPU presence guard at cell 1** — always:
```python
import torch
if not torch.cuda.is_available():
    print("No GPU detected. This notebook requires a CUDA GPU. Exiting.")
    raise SystemExit("GPU required — run on Colab GPU runtime or local CUDA machine.")
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

2. **No long-running cells without a time estimate comment:**
```python
# ~45 seconds on RTX 4090 / ~8 min on Colab T4
image = pipe(prompt, num_inference_steps=50).images[0]
```

3. **Memory cleanup before heavy operations:**
```python
torch.cuda.empty_cache()  # clear KV/activation fragments before loading new model
```

---

### Image Conventions — MultimodalAI Track

| Image type | Purpose | Example filename |
|-----------|---------|-----------------|
| Generation comparison | CFG scale 1.0 vs 7.5 vs 15 on same prompt | `ch05-cfg-scale-comparison.png` |
| Diffusion timeline | x_0 → x_T forward + reverse visualization | `ch03-diffusion-timeline.png` |
| VisualForge output | Before/after: freelancer vs AI for same brief | `chNN-visualforge-before-after.png` |
| Architecture diagram | UNet, VAE encoder/decoder, CLIP text encoder | `ch04-ldm-architecture.png` |
| Metric chart | FID / HPSv2 improving across chapters | `ch11-quality-metric-progress.png` |
| Needle GIF | Which constraint moved this chapter | `chNN-[topic]-needle.gif` |

All images in `notes/MultimodalAI/{ChapterName}/img/`. Dark background `#1a1a2e` for generated charts. **Generated images themselves** (the diffusion outputs) do not need dark backgrounds — show them as-is.

---

### Red Lines — MultimodalAI Track

In addition to the universal red lines:

1. **No generation example with a generic prompt** — every prompt must be a real VisualForge brief type (spring collection, product demo, brand pattern, etc.)
2. **No quality claim without a metric** — "better quality" must be backed by HPSv2, CLIP score, or FID; visual impressionism is not evidence
3. **No GPU notebook cell without a time estimate** — always comment expected runtime on RTX 4090 and Colab T4
4. **No GPU supplement cell without the GPU presence guard** — chapter 1, cell 1, always
5. **No magic hyperparameter without the failure case** — if you recommend `guidance_scale=7.5`, show what 1.0 and 15.0 look like first
