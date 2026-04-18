# Generative Evaluation — Measuring What You Made

> **Track:** Multimodal AI 
> **Prerequisites:** [DiffusionModels.md](../DiffusionModels/DiffusionModels.md), [GuidanceConditioning.md](../GuidanceConditioning/GuidanceConditioning.md), [CLIP.md](../CLIP/CLIP.md)

> **The story.** Evaluating generative images is a 9-year-old subfield. **Inception Score (IS)** (Salimans et al., OpenAI, **2016**) was the first widely-used automatic metric — high IS meant images were both confident and diverse under an Inception classifier — but it was famously gameable. **FID** — *Fréchet Inception Distance* (**Heusel et al.**, **NIPS 2017**) — replaced IS by comparing the distribution of generated and real Inception features under a Gaussian assumption. FID became the field's default metric for half a decade despite its quirks (sample-size sensitivity, blindness to text alignment). **CLIPScore** (Hessel et al., 2021) added text-image alignment by reusing CLIP. **Human Preference Score (HPS)** (Wu et al., 2023) and **PickScore** (Kirstain et al., NeurIPS 2023) trained reward models on millions of human preference pairs from ChatGPT-style A/B comparisons — the first metrics that actually correlated well with what humans like. The 2026 evaluation stack pairs all of these with a multimodal LLM judge.
>
> **Where you are in the curriculum.** You can generate images. The honest question is: *are they any good?* This chapter gives the toolkit — FID, IS, CLIPScore, HPS, human preference — and explains why each one can mislead you and which combination to ship in production.

---

## 1 · Core Idea

Generative evaluation is the science of measuring the **quality, fidelity, diversity, and alignment** of images produced by a model — without requiring a human judge for every sample.

Three orthogonal axes to measure:

| Axis | Question | Representative metric |
|------|----------|-----------------------|
| **Fidelity** | Do generated images look real? | FID ↓ |
| **Diversity** | Does the model cover the full distribution? | FID ↓, Precision/Recall |
| **Alignment** | Does the image match its text prompt? | CLIP Score ↑ |

No single metric captures all three. Use at least two.

---

## 2 · Running Example

**PixelSmith evaluation suite.** At the end of Chapter 4 (DDPM) training, we want to know:
- Is our DDPM generating digits that look like MNIST? → FID
- Are all 10 digit classes represented? → class recall
- (If we add text conditioning) Does "handwritten seven" produce a 7? → CLIP Score

---

## 3 · The Math

### 3.1 Fréchet Inception Distance (FID)

Extract features $\mu_r, \Sigma_r$ from **real** images and $\mu_g, \Sigma_g$ from **generated** images using a pre-trained feature extractor (canonically Inception-v3 pool3 layer):

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr} \left(\Sigma_r + \Sigma_g - 2 \left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

- Lower = better.
- Measures distance between the *distributions*, not individual images.
- **Biased at small N** — needs ≥ 5,000 samples for stable estimates (often 50k).

### 3.2 Inception Score (IS)

Uses marginal $p(y)$ and conditional $p(y \mid x)$ from the Inception classifier:

$$
\text{IS} = \exp \left(\mathbb{E}_x\bigl[D_\text{KL}(p(y|x)\|p(y))\bigr]\right)
$$

- Higher = better (sharp images → high $p(y|x)$; diverse images → high entropy $p(y)$).
- **Does not compare to real images** — a model memorising training data can achieve high IS.
- Rarely used alone after FID became standard.

### 3.3 CLIP Score

Given generated image $x$ and its text prompt $t$:

$$
\text{CLIP Score} = w \cdot \max(0, \cos(\text{CLIP}_I(x), \text{CLIP}_T(t)))
$$

where $w = 2.5$ is a scaling constant (originates from CLIPScore paper, Hessel et al. 2021).

- Higher = better.
- Reference-free: no real image needed.
- The CLIP embedding space is **shared** across images and text, so cosine similarity measures semantic alignment.

### 3.4 LPIPS (Learned Perceptual Image Patch Similarity)

Compare a generated image $\hat{x}$ to a reference $x$:

$$
\text{LPIPS}(\hat{x}, x) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}^l_{hw} - y^l_{hw}) \|^2_2
$$

- $y^l$: VGG/AlexNet/SqueezeNet feature map at layer $l$, channel-normalised.
- $w_l$: learned channel weights.
- **Lower = more perceptually similar** to reference.
- Used for img2img tasks (e.g., inpainting quality).

### 3.5 Precision & Recall for Generative Models

Kynkäänniemi et al. 2019 formulation using $k$-NN manifold estimation:

- **Precision**: fraction of generated samples inside the real manifold (fidelity)
- **Recall**: fraction of real samples covered by the generated manifold (diversity)

$$
\text{Precision} = \frac{1}{|X_g|}\sum_i \mathbf{1}[x_{g,i} \in \text{manifold}(X_r)]
$$

---

## 4 · How It Works — Step by Step

### Computing FID

1. **Generate** $N$ images from your model ($N \geq 5000$, ideally 50k).
2. **Extract features**: pass each real and generated image through Inception-v3 up to the `mixed_7c` pooling layer → 2048-dim vector.
3. **Fit Gaussians**: compute $(\mu_r, \Sigma_r)$ on real features, $(\mu_g, \Sigma_g)$ on generated features.
4. **Compute matrix square root**: $(\Sigma_r \Sigma_g)^{1/2}$ via eigendecomposition.
5. **Plug into formula above** — result is FID.

### Computing CLIP Score

1. Encode the prompt with `CLIPTextEncoder` → $\mathbf{t} \in \mathbb{R}^{512}$.
2. Encode the generated image with `CLIPImageEncoder` → $\mathbf{v} \in \mathbb{R}^{512}$.
3. Normalise both to unit length.
4. Score = $2.5 \cdot \max(0, \mathbf{t} \cdot \mathbf{v})$.

---

## 5 · The Key Diagrams

```
 GENERATIVE EVALUATION LANDSCAPE
 ─────────────────────────────────

 Reference-free Reference-based
 (no real images needed) (compares to real distribution)

 ┌─────────────────────┐ ┌──────────────────────────────┐
 │ CLIP Score │ │ FID (distribution match) │
 │ text ↔ image align │ │ IS (fidelity + diversity) │
 │ HPSv2, ImageReward │ │ Precision / Recall │
 │ (human preference) │ │ LPIPS (pixel-level, per img)│
 └─────────────────────┘ └──────────────────────────────┘

 Sample-level Distribution-level
 (per image score) (needs thousands of images)

 ┌─────────────────────┐ ┌──────────────────────────────┐
 │ LPIPS │ │ FID, IS, Precision/Recall │
 │ SSIM, PSNR │ │ (stable only with N≥5k) │
 │ CLIP Score │ └──────────────────────────────┘
 └─────────────────────┘
```

```
FID BIAS VS SAMPLE COUNT
─────────────────────────
FID
 ↑
300│ × N=100
200│ × N=500
100│ × N=1k
 50│ × N=5k
 20│ × N=50k ← stabilises here
 └───────────────────────────→ N (log scale)

True FID attained only at large N; small N inflates FID.
```

---

## 6 · What Changes at Scale

| Scale | What matters |
|-------|-------------|
| **Research prototyping** | FID on 2k–10k samples, CLIP Score spot-check |
| **Production model eval** | FID on 50k, human preference study (HPSv2 / ELO) |
| **T2I leaderboards** | GenEval (compositional), T2I-CompBench, DrawBench |
| **Video generation** | FVD (Fréchet Video Distance) — temporal extension of FID |
| **Beyond images** | CLIPScore adapted to audio-text, video-text |

Human preference models (HPSv2, ImageReward, PickScore) train a reward model on human pairwise comparisons — better aligned with user perception than automated metrics.

---

## 7 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Lower FID is always better" | FID measures *match* to the training distribution; a model overfitting real images can get near-zero FID but zero diversity |
| "CLIP Score measures photorealism" | CLIP Score measures text-image alignment, not visual quality |
| "IS is equivalent to FID" | IS doesn't compare to real images at all — it only uses the generator's class distribution |
| "FID on 1,000 samples is reliable" | FID has O(1/√N) variance; ±10 FID spread is common at N=1k |
| "LPIPS = SSIM" | LPIPS uses deep network features (learned); SSIM is a hand-crafted pixel similarity |
| "CLIP embeddings are perceptually uniform" | CLIP can match text to semantically wrong images if colours/textures align spuriously |

---

## 8 · Interview Checklist

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

## 9 · What's Next

→ [LocalDiffusionLab.md](../LocalDiffusionLab/LocalDiffusionLab.md) — Capstone: combine everything you've built across all 12 chapters into a single local diffusion lab.

## Illustrations

![Generative evaluation - FID, CLIPScore, metric coverage, human eval pipeline](img/Generative%20Evaluation.png)
