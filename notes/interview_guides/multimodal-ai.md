# Multimodal AI · Interview Guide

This guide consolidates interview preparation material from all chapters in the Multimodal AI track, covering vision transformers, contrastive learning, diffusion models, and production text-to-image/video systems.

---

## Ch.1 — Multimodal Foundations

### Must Know
- What shape is a colour image tensor in PyTorch? `(N, C, H, W)` — batch, channels, height, width
- What is the modality gap and why does it require a solution beyond simple concatenation?
- What normalisation does ImageNet-pretrained models expect, and why does it matter?

### Likely Asked
- "How would you convert a 5-second 16 kHz audio clip to a tensor ready for a ViT-based model?"
  → resample → STFT → mel filterbank → log scale → treat as 2-D image
- "Why does the same pixel value mean something different in a diffusion model vs a classification model?"
  → different normalisation conventions (`[-1,1]` vs `[0,1]` vs ImageNet stats)

### Trap to Avoid
- Forgetting to convert channel order (HWC → CHW) when going from PIL/NumPy to PyTorch
- Using ImageNet normalisation for medical images or satellite imagery — domain mismatch will hurt
- Assuming float16 is always safe — some operations (loss computation, LR schedules) need float32 accumulation

---

## Ch.2 — Vision Transformers

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

## Ch.3 — CLIP

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

## Ch.4 — Diffusion Models

### Must Know
- What does the U-Net predict in DDPM — the image or the noise?
- Write the closed-form forward process equation $q(x_t | x_0)$
- Why is the DDPM loss just MSE on noise prediction?

### Likely Asked
- "Why does DDPM need $T = 1000$ steps? Why not just use $T = 10$?"
 → Fewer steps → each $\beta_t$ must be larger → the Gaussian approximation of the reverse step breaks down → poor generation quality. Fast samplers (DDIM) solve inference speed without retraining.
- "What is the signal-to-noise ratio at step $t$, and what does $\bar{\alpha}_t$ represent?"
 → $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$; $\bar{\alpha}_t$ = fraction of original signal remaining
- "Why are diffusion models more stable than GANs?"
 → No adversarial game; the loss is a simple MSE; no generator/discriminator equilibrium required

### Trap to Avoid
- Confusing $\beta_t$ (noise variance) with $\alpha_t = 1 - \beta_t$ (signal retention fraction)
- Saying diffusion generates in a single network forward pass — inference requires repeated U-Net calls
- Forgetting to add noise at every sampling step except the final one (otherwise you lose stochasticity)

---

## Ch.5 — Schedulers

### Must Know
- Why DDPM needs 1 000 steps at inference: each step is a Gaussian approximation that only holds for small β — larger strides violate the Markov assumption
- DDIM key insight: rewrite as ODE, enabling deterministic sub-sequence sampling
- The trade-off axes: steps ↓ speed ↑ quality ↓ diversity (generally true)

### Likely Asked
- *"What scheduler does Stable Diffusion use by default?"* — PNDM (pseudo-numerical) or DDIM; SDXL defaults to EulerDiscreteScheduler
- *"How would you halve inference time without quality loss?"* — Switch from 50-step DDIM to 15-step DPM-Solver++
- *"What is the relationship between DDIM and DDPM?"* — DDIM is a non-Markovian generalization that reduces to DDPM when σ=original noise level; at σ=0 it becomes fully deterministic

### Trap to Avoid
- Don't confuse the **training** noise schedule (always 1 000 steps, defines q(x_t|x_0)) with the **inference** step count (scheduler-specific). They are independent after training.
- **LCM / distillation:** Latent Consistency Models learn to map any noisy latent directly to the clean latent in 1–4 steps by enforcing self-consistency along the ODE trajectory; LCM-LoRA distils this as a lightweight adapter. SD-Turbo and SDXL-Turbo use adversarial diffusion distillation. Trap: "LCM images are the same quality as 50-step DDIM" — 1–4 step models sacrifice fine detail and diversity; 8+ steps are usually needed to match 50-step DDIM quality

---

## Ch.6 — Latent Diffusion

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

---

## Ch.7 — Guidance & Conditioning

### Must Know
- Write the CFG equation. What are the two model calls?
- What does guidance scale $w = 1$ vs $w = 7.5$ vs $w = 15$ produce?
- How is condition dropout used during CFG training?

### Likely Asked
- "How does a negative prompt work mechanically?"
 → Replace the unconditioned embedding with the negative prompt embedding; the CFG equation then steers the image away from the negative and toward the positive
- "Why is CFG inference twice as slow as unconditioned inference?"
 → Two separate U-Net forward passes per denoising step — one conditioned, one not
- "What is attention control / prompt-to-prompt?"
 → Manipulate the cross-attention maps directly (instead of changing the embedding) to achieve localised edits: change "a photo of a cat" to "a photo of a dog" while keeping the composition

### Trap to Avoid
- Confusing guidance scale with classifier temperature — they are different mechanisms
- Saying negative prompts "block" content — they steer, not filter
- Forgetting that $w=0$ ignores the prompt entirely (unconditioned), not $w=1$

---

## Ch.8 — Text-to-Image

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

## Ch.9 — Text-to-Video

### Must Know
- Why video adds temporal consistency as a fundamental new challenge (image-by-image generates flicker)
- Temporal attention: attend across frames at same spatial position
- AnimateDiff design: freeze spatial SD layers, train only temporal modules on video data

### Likely Asked
- *"How would you generate a 30-fps video cheaply?"* — Generate 8-frame keyframes with a T2V model, interpolate with a frame interpolation model (RIFE, EMA-VFI)
- *"What is a spacetime patch in Sora/DiT?"* — A 3D patch $(t_p, h_p, w_p)$ treated as a single transformer token; enables arbitrary-length, arbitrary-resolution video generation
- *"What's the difference between animating a static image vs. text-to-video?"* — Animating (img2video, Stable Video Diffusion) starts from a real image latent; T2V starts from pure noise; both use temporal attention

### Trap to Avoid
- Don't claim that temporal consistency is free from DDPM — the Markovian noise process is *independent per frame*; correlation across frames must be learned explicitly via temporal attention layers.

---

## Ch.10 — Multimodal LLMs

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

---

## Ch.11 — Generative Evaluation

### Must Know
- FID formula: Fréchet distance between Gaussians fitted to Inception features
- Why FID needs large N (bias, variance)
- CLIP Score: cosine similarity between CLIP text and image embeddings, scaled by 2.5
- Trade-off: no single metric captures fidelity *and* diversity *and* text alignment
- LPIPS vs. SSIM — learned vs. hand-crafted perceptual similarity

### Likely Asked
- "What's the difference between FID and IS?" (FID uses real images; IS does not)
- "How would you evaluate text-to-image generation?" (FID + CLIP Score + human eval)
- "Why does FID increase when you use fewer samples?" (Gaussian fit becomes noisier)
- "Name a metric for evaluating compositional text prompts" (GenEval, T2I-CompBench)
- "What is Precision/Recall in the context of generative models?"

### Traps to Avoid
- Confusing CLIP Score (semantic alignment) with FID (distributional realism).
- Reporting FID on < 5k samples without flagging the bias.
- Conflating LPIPS (reference-based perceptual similarity) with CLIP Score (reference-free text alignment).
- Forgetting that FID is scale-sensitive: spatial resolution must match between real and generated sets.
- **Video generation metrics:** FVD (Fréchet Video Distance) extends FID to video using an I3D 3D-CNN feature extractor; captures temporal coherence, not just per-frame quality. CLIPSIM averages CLIP Score across frames — measures text alignment but ignores temporal consistency. VBench is the current standardised suite (16 dimensions including subject consistency and motion smoothness). Trap: "high per-frame FID means good video" — per-frame FID ignores temporal coherence entirely; a strobing video can score well per-frame
- **Compositional text-to-image evaluation:** standard FID/CLIP Score miss attribute binding failures ("a red cube and blue sphere" where colours are swapped). GenEval and T2I-CompBench specifically test spatial relations, attribute-object binding, and counting. Trap: "CLIP Score captures compositional accuracy" — CLIP Score is a global semantic similarity; it cannot verify fine-grained binding and scores an image with swapped attributes almost identically to the correct one

---

## Ch.12 — Local Diffusion Lab

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

## Related Topics

- [Agentic AI Interview Guide](agentic-ai.md) — CoT, ReAct, RAG, embeddings fundamentals
- [Multi-Agent AI Interview Guide](multi-agent-ai.md) — agent protocols, MCP, A2A, event-driven systems
- [AI / LLM Fundamentals](../ai/llm_fundamentals) — transformer architecture, attention mechanism
- [AI / Fine-tuning](../ai/fine_tuning) — LoRA, full fine-tuning, adapter methods

---

**End of Multimodal AI Interview Guide**
