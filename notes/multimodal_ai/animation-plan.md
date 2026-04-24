# MultimodalAI Animation Rollout Plan

_Date: 2026-04-21_

## Goal

Add one clear **data-flow animation** per MultimodalAI chapter so readers can see how tensors/tokens move between components.

Primary requirement for each chapter:

1. Show component-to-component flow (encoder -> projector -> decoder/sampler)
2. Animate at least 3 stages: input -> transform -> output
3. Export both `.png` (poster frame) and `.gif` (animated flow)
4. Embed the GIF in the chapter note near the opening sections

---

## Current Audit Snapshot

What exists today:

- 12 generator scripts already exist under chapter `img/` folders (`gen_*.py`)
- scripts currently generate static concept images (mostly PNG)
- scripts use hardcoded absolute output paths in several files
- no centralized animation status tracker for this track

What is missing for the target state:

- standardized chapter flow animation naming + output conventions
- consistent GIF embedding in chapter notes
- a chapter-by-chapter closeout checklist

---

## Standard Convention

### File layout

```text
notes/MultimodalAI/<Chapter>/
├── <Chapter>.md or README.md
├── notebook.ipynb
├── gen_scripts/
│   └── gen_<chapter>_flow.py
└── img/
    ├── <chapter>-flow.png
    └── <chapter>-flow.gif
```

### Naming

- script: `gen_<chapter>_flow.py`
- assets: `<chapter>-flow.png` and `<chapter>-flow.gif`

### Rendering defaults

- backend: `matplotlib` Agg
- gif writer: `pillow`
- fps: `5` to `8`
- deterministic seeds for reproducible visuals
- relative output paths only (no machine-specific absolute paths)

---

## Chapter Animation Map

| # | Chapter | Flow to Animate | Target Asset | Status |
|---|---------|------------------|--------------|--------|
| 1 | MultimodalFoundations | raw signal -> tensor -> tokens -> shared embedding | `multimodal-foundations-flow.gif` | complete |
| 2 | VisionTransformers | image -> patches -> patch embeddings -> MHSA blocks -> CLS head | `vision-transformers-flow.gif` | complete |
| 3 | CLIP | image/text encoders -> projection heads -> cosine similarity matrix | `clip-flow.gif` | complete |
| 4 | DiffusionModels | x0 -> forward noise chain -> reverse denoise -> sample x0 hat | `diffusion-models-flow.gif` | complete |
| 5 | GuidanceConditioning | text conditioning + unconditional branch -> CFG combine -> denoise step | `guidance-conditioning-flow.gif` | complete |
| 6 | Schedulers | same model, different sampler trajectories (DDPM/DDIM/DPM) | `schedulers-flow.gif` | complete |
| 7 | LatentDiffusion | pixel -> VAE latent -> denoise in latent space -> VAE decode | `latent-diffusion-flow.gif` | complete |
| 8 | TextToImage | prompt -> tokenizer/text encoder -> UNet denoise loop -> image decode | `text-to-image-flow.gif` | complete |
| 9 | TextToVideo | prompt + temporal blocks -> latent frames -> decoder -> video clip | `text-to-video-flow.gif` | complete |
| 10 | MultimodalLLMs | image encoder -> projector/Q-former -> LLM tokens -> answer | `multimodal-llms-flow.gif` | complete |
| 11 | GenerativeEvaluation | generated set + reference set -> metrics -> preference signal | `generative-evaluation-flow.gif` | complete |
| 12 | LocalDiffusionLab | train mini DDPM -> sample -> compare scheduler outputs | `local-diffusion-lab-flow.gif` | complete |
| 13 | AudioGeneration | text -> tokenizer -> acoustic model -> waveform -> playable audio | `audio-generation-flow.gif` | complete |

---

## Rollout Phases

-### Phase 1 - Infrastructure + Path Fixes

- move script entrypoints to `gen_scripts/`
- normalize all outputs to local `img/`
- remove hardcoded absolute save paths
- create a tiny shared helper for common flow animation boilerplate

### Phase 2 - Core Generation Chapters

- DiffusionModels
- GuidanceConditioning
- Schedulers
- LatentDiffusion
- TextToImage
- TextToVideo

Success criteria:

- each chapter has GIF + PNG
- chapter note embeds GIF
- visual clearly shows data movement between components

### Phase 3 - Foundation + Alignment + MLLM

- MultimodalFoundations
- VisionTransformers
- CLIP
- MultimodalLLMs
- AudioGeneration

### Phase 4 - Evaluation + Capstone

- GenerativeEvaluation
- LocalDiffusionLab
- final consistency pass across all 13 chapters

---

## Closeout Checklist (per chapter)

- flow animation script runs from repo root
- output written to chapter `img/`
- chapter note embeds `.gif`
- one-sentence caption explains the flow in plain language
- notebook references animation where relevant

---

## Completion Definition

This plan is complete when all 13 chapters satisfy:

- `MISSING_GIF = 0`
- `MISSING_PNG = 0`
- `MISSING_EMBED = 0`

and all scripts use relative paths that run on any contributor machine.
