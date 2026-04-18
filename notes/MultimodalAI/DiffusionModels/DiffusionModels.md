# Diffusion Models — The Mathematics of Denoising

> **The story.** Generative modelling spent a decade trying to make GANs work. **Ian Goodfellow's GAN** (2014) was a brilliant idea — generator vs discriminator in adversarial equilibrium — but training was famously unstable, mode-collapse-prone, and an art form. The shift began with **Sohl-Dickstein et al.** (Stanford, **2015**), who proposed a different idea inspired by *non-equilibrium thermodynamics*: gradually destroy data with noise, then learn to reverse the noise. The recipe sat on the shelf for five years until **Jonathan Ho, Ajay Jain, and Pieter Abbeel** at Berkeley published **DDPM** — *Denoising Diffusion Probabilistic Models* — in **June 2020**. DDPM matched GANs on image quality with stable, deterministic training and a clean probabilistic foundation. **Song & Ermon's score-based models** (NeurIPS 2019, 2020) gave the same thing a continuous-time interpretation. By 2022 GANs were largely abandoned and diffusion was the default — a 2-year transition that rivals 2017's transformer takeover.
>
> **Where you are in the curriculum.** This is the math chapter of the multimodal track. You will derive the forward noising process, the reverse denoising process, the loss the U-Net actually minimises (predict the noise, not the image), and why diffusion is more stable to train than GANs. After this, [LatentDiffusion](../LatentDiffusion/), [Schedulers](../Schedulers/), and [GuidanceConditioning](../GuidanceConditioning/) are all engineering refinements of the same core idea.

---

## 1 · Core Idea

**Diffusion models** generate images by learning to reverse a noise-injection process. The key insight is the **forward process**: take a real image, add Gaussian noise at each of $T$ steps, and after $T$ steps you have pure Gaussian noise — indistinguishable from random. This process has a beautiful analytical form: you can jump to any noisy step in a single operation. The **reverse process** learns a neural network to undo this noise, one step at a time. Given pure noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, denoise it $T$ times and you reconstruct a plausible image.

The crucial distinction from earlier generative models: **diffusion models predict noise, not images**. The U-Net is trained to predict the noise that was added at each step, not to reconstruct the original image directly. This indirect objective produces more stable training than GANs (no adversarial game, no mode collapse) while achieving superior image quality.

---

## 2 · Running Example — PixelSmith v3

```
Goal: Generate new handwritten digit images from pure noise
Architecture: A U-Net trained on MNIST-style data
Training: ~5 minutes on CPU (MNIST is small)
Inference: Sample x_T ~ N(0, I) → denoise 1000 steps → x_0 ∈ [0, 1]²⁸ˣ²⁸
```

---

## 3 · The Math

### 3.1 The Forward (Noising) Process

Define a fixed Markov chain that gradually adds Gaussian noise over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

where $\beta_t \in (0,1)$ is the **noise schedule** — small at first (barely any noise) and large near $T$ (mostly noise).

**The key shortcut:** using reparameterisation, you can jump directly to any noisy step $t$ without iterating:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

where:

$$\alpha_t = 1 - \beta_t \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

**Signal-to-noise ratio:** at step $t$, the fraction of original signal is $\sqrt{\bar{\alpha}_t}$ and the fraction of noise is $\sqrt{1 - \bar{\alpha}_t}$. When $t = T$, $\bar{\alpha}_T \approx 0$ — pure noise.

### 3.2 The Reverse Process

The reverse process is what the model learns. It is also Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \boldsymbol{\mu}_\theta(x_t, t), \tilde{\beta}_t \mathbf{I})$$

The mean $\boldsymbol{\mu}_\theta$ is parameterised by a neural network (U-Net) $\boldsymbol{\epsilon}_\theta$:

$$\boldsymbol{\mu}_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right)$$

The model predicts the **noise** $\boldsymbol{\epsilon}_\theta$, not $x_0$ directly.

### 3.3 The Training Objective

The variational lower bound simplifies to a surprisingly clean loss:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\underbrace{\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}_{x_t}, t) \|^2 \right]$$

**In plain English:** sample a random image $x_0$ from the training set, sample a random noise $\boldsymbol{\epsilon}$, sample a random timestep $t$, compute $x_t$ using the closed-form forward jump, ask the U-Net to predict $\boldsymbol{\epsilon}$, compute MSE. That's it.

### 3.4 The DDPM Sampling Algorithm

At inference, start from $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iterate:

```
for t = T, T-1, ..., 1:
 ε̂ = ε_θ(x_t, t) # predict noise
 x̂₀ = (x_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t # estimate original image
 μ = (√ᾱ_{t-1} · β_t · x̂₀ + √α_t · (1-ᾱ_{t-1}) · x_t) / (1 - ᾱ_t)
 z ~ N(0, I) if t > 1 else z = 0
 x_{t-1} = μ + √β̃_t · z # add controlled noise (except last step)
```

### 3.5 Noise Schedules

| Schedule | $\beta_t$ formula | Properties |
|----------|-------------------|-----------|
| Linear | $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$ | Simple; DDPM default ($\beta_1=10^{-4}$, $\beta_T=0.02$) |
| Cosine | $\bar{\alpha}_t = \cos^2 \left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Smoother; avoids abrupt noise near $T$ |
| Sigmoid | Based on sigmoid function | Better signal-to-noise at boundaries |

The cosine schedule (improved DDPM, 2021) became the standard because the linear schedule destroys too much signal in the first steps at high resolution.

---

## 4 · How It Works — Step by Step

**Training:**
1. Sample $x_0$ from training set (a real image)
2. Sample noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ same shape as $x_0$
3. Sample timestep $t \sim \text{Uniform}(1, T)$
4. Compute noisy image: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$
5. Feed $(x_t, t)$ to U-Net; predict $\hat{\boldsymbol{\epsilon}}$
6. Compute loss: $\| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}} \|^2$
7. Backprop, update U-Net weights

**Inference:**
1. Sample $x_T \sim \mathcal{N}(0, \mathbf{I})$
2. For $t = T, T-1, \ldots, 1$:
 - Predict noise: $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(x_t, t)$
 - Compute posterior mean $\boldsymbol{\mu}$
 - Sample $x_{t-1} = \boldsymbol{\mu} + \sqrt{\tilde{\beta}_t} \cdot z$
3. Return $x_0$ — the generated image

---

## 5 · The Key Diagrams

### Forward and Reverse Processes

```
FORWARD PROCESS (fixed, no learnable parameters)

x₀ ─────q────▶ x₁ ─────q────▶ x₂ ─────q────▶ ... ─────q────▶ xₜ
[clean] [small noise] [more noise] [pure noise]

q(x_t | x_{t-1}) = N(√(1-βt)·x_{t-1}, βt·I)


REVERSE PROCESS (learned by U-Net)

xₜ ──pθ──▶ x_{t-1} ──pθ──▶ x_{t-2} ──pθ──▶ ... ──pθ──▶ x₀
[pure noise] [generated image]

pθ(x_{t-1}|x_t) = N(μθ(x_t,t), β̃t·I)
```

### U-Net Architecture

```
Input: (x_t, t) ← noisy image + timestep embedding

 ┌────────────────────────────────────────────────────┐
 │ Encoder (downsampling) │
 │ ResBlock → ResBlock → Downsample → ... │
 │ Spatial: 64×64 → 32×32 → 16×16 → 8×8 │
 │ Attention at 16×16 and 8×8 (for larger models) │
 └───────────────────────┬────────────────────────────┘
 │
 ┌────────────────────────▼────────────────────────────┐
 │ Bottleneck: ResBlock + Self-Attention + ResBlock │
 └───────────────────────┬────────────────────────────┘
 │
 ┌────────────────────────▼────────────────────────────┐
 │ Decoder (upsampling) │
 │ Upsample → ResBlock (+ skip from encoder) → ... │
 │ Skip connections preserve spatial detail │
 └───────────────────────┬────────────────────────────┘
 │
 Output: ε̂ ← predicted noise, same shape as input
```

---

## 6 · What Changes at Scale

| Aspect | MNIST DDPM (this chapter) | Production (Stable Diffusion) |
|--------|--------------------------|-------------------------------|
| Image size | 28×28 pixels | 512×512+ (in latent space: 64×64) |
| U-Net channels | 32–128 | 320–1280 |
| Attention | Sometimes omitted | At multiple resolutions; cross-attention for text |
| T steps | 1000 | 1000 (training); 20–50 (inference with fast samplers) |
| Training data | 60K MNIST images | Billions of image-text pairs |
| Training time | 5 minutes CPU | Weeks on 256+ A100s |
| Conditioning | Unconditional | Text via cross-attention (CLIP encoder) |

**Why 1000 steps?** The MSE loss over the full Step 5 trajectory gives a very smooth optimisation landscape. You can use fewer steps at inference with DDIM (Chapter 6) — but training still requires 1000 steps to learn a good noise predictor at every noise level.

---

## 7 · Common Misconceptions

**"The U-Net predicts the clean image $x_0$"**
It predicts the noise $\boldsymbol{\epsilon}$. You can reparameterise to predict $x_0$ (x-prediction parameterisation), but the standard DDPM paper and most implementations use noise prediction. The two are mathematically equivalent given $x_t$, but noise prediction tends to train more stably.

**"Diffusion models are slow because they need 1000 steps"**
Training requires 1000 steps. Inference with DDIM or DPM-Solver can generate images in 20–50 steps with nearly identical quality. Chapter 6 covers this in full.

**"More noise steps always means better quality"**
Beyond a certain threshold (typically T=1000), adding more steps gives diminishing returns. The quality is determined primarily by the U-Net capacity, training data, and loss weighting — not just T.

**"GANs are better because they generate in one step"**
GANs are faster at inference but harder to train (mode collapse, training instability), and at scale diffusion models produce significantly better image quality and diversity. GANs have been largely superseded for image generation tasks.

---

## 8 · Interview Checklist

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

## 9 · What's Next

→ **[GuidanceConditioning.md](../GuidanceConditioning/GuidanceConditioning.md)** — PixelSmith v3 generates unconditional MNIST images. PixelSmith v4 will generate images from text prompts. The bridge is **guidance**: conditioning the denoising process on an additional signal (a text embedding from CLIP) so that the final image reflects your prompt. This requires classifier-free guidance (CFG) — the mechanism that makes "guidance scale 7.5" produce sharper, more prompt-aligned images than "guidance scale 1.0".

## Illustrations

![Diffusion models - forward noising, reverse denoising, noise schedule, loss](img/Diffusion%20Models.png)
