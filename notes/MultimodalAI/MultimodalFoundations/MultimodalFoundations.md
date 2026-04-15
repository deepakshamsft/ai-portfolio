# Multimodal Foundations — How Raw Signals Become Tensors

> **After reading this note** you will understand how images, audio, and video are represented as numerical tensors; why modalities cannot be naively mixed in a single model; and why every subsequent chapter in this track begins with "project the signal into a shared embedding space."

---

## 1 · Core Idea

A neural network can only process numbers. Every raw signal — a JPEG, an MP3, a video clip — must be converted into a **tensor** (a multi-dimensional array of floating-point numbers) before a model can operate on it. The challenge is not the conversion itself; it is that the resulting tensors are wildly different in shape, density, and statistical distribution, and models trained on one modality cannot transfer directly to another. **Multimodal AI is the problem of bridging these representations** so that a single model — or a paired set of models — can reason jointly over text, images, audio, and video.

---

## 2 · Running Example — PixelSmith v0

```
Goal:       Load a JPEG → inspect its tensor representation → understand what
            the model actually "sees" before any processing.

Input:      Any photograph (e.g. a city skyline at night)
Output:     A (3, H, W) float32 tensor, pixel value statistics, and a
            visualisation of individual colour channels
```

By the end of this chapter you will have built the input stage of PixelSmith — the part that accepts raw files and produces tensors ready for downstream model layers.

---

## 3 · The Math

### 3.1 Images as Tensors

A colour image is a 3-D tensor:

$$I \in \mathbb{R}^{C \times H \times W}$$

where $C = 3$ (red, green, blue channels), $H$ is height in pixels, and $W$ is width in pixels. Each value is an integer in $[0, 255]$ (uint8) at load time, normalised to $[0, 1]$ or $[-1, 1]$ before model input.

**Normalisation** used by most vision models (ImageNet statistics):

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}$$

| Channel | $\mu_c$ | $\sigma_c$ |
|---------|---------|-----------|
| Red     | 0.485   | 0.229     |
| Green   | 0.456   | 0.224     |
| Blue    | 0.406   | 0.225     |

### 3.2 Audio as Tensors

A mono audio clip sampled at 16 kHz is a 1-D tensor:

$$a \in \mathbb{R}^{T}$$

where $T = \text{duration\_s} \times 16000$. A 5-second clip gives $T = 80{,}000$ samples.

For model input, raw waveforms are almost always converted to a **mel spectrogram** — a 2-D representation $M \in \mathbb{R}^{F \times T'}$ where $F$ is the number of mel frequency bins (typically 80 or 128) and $T'$ is the number of time frames. The mel spectrogram is computed via:

1. Short-Time Fourier Transform (STFT): window the signal → FFT → magnitude
2. Apply mel filterbank: project frequency bins onto perceptual mel scale
3. Apply log: $\log(M + \epsilon)$ to compress dynamic range

### 3.3 Video as Tensors

A video clip is a 4-D tensor:

$$V \in \mathbb{R}^{T \times C \times H \times W}$$

where $T$ is the number of sampled frames (not raw frame count — typically every $k$th frame). A 1-second clip at 8 fps, 224×224, gives shape $(8, 3, 224, 224)$ — roughly 1.2 million floats per second of video.

### 3.4 The Modality Gap

Even when tensors have similar dimensions, the **statistical distributions** differ enormously:

| Modality | Typical value range | Correlation structure |
|----------|--------------------|-----------------------|
| Image pixels | $[0, 1]$ | Smooth spatial gradients; local structure |
| Text token IDs | $[0, 50256]$ (integers) | Sequential; long-range dependencies |
| Audio mel-spec | $[-10, 2]$ (log scale) | Temporal; harmonic patterns |
| Video frames | $[0, 1]$ per frame | Spatial + temporal; high redundancy |

A model trained purely on images has no mechanism to process a sequence of integers: they live in incompatible spaces. **The modality gap is the core problem multimodal AI solves.**

---

## 4 · How It Works — Step by Step

### Step 1: Load the raw signal

```
JPEG file → PIL.Image.open() → mode "RGB" → numpy array shape (H, W, 3)
```

### Step 2: Reshape to channel-first

```
(H, W, 3)  →  (3, H, W)
```
PyTorch convention: channels first. PIL/NumPy convention: channels last. Always convert.

### Step 3: Normalise

```
uint8 [0, 255]  →  float32 [0.0, 1.0]   by dividing by 255
                →  float32 [-1.0, 1.0]  by (x - 0.5) / 0.5   (diffusion models)
                →  ImageNet-normalised  by channel-wise (x - μ) / σ  (ViT, CLIP, ResNet)
```

### Step 4: Batch

```
Single image (3, H, W)  →  batched (N, 3, H, W)  by unsqueeze(0) or torch.stack
```

### Step 5: Send to model

The model receives a `(N, 3, H, W)` tensor. It knows nothing else about the image. Everything the model "understands" about the photograph — objects, colours, scene, relationships — is derived exclusively from these numbers.

---

## 5 · The Key Diagrams

### Signal → Tensor Pipeline

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────────────────────┐
│  Raw File     │    │   Load & Decode  │    │  Normalise + Reshape           │
│               │    │                  │    │                                │
│  image.jpg    │───▶│  (H, W, 3)       │───▶│  (3, H, W)   float32 [-1,1]   │
│  audio.wav    │    │  (T,)             │    │  (F, T')     log mel-spec      │
│  video.mp4    │    │  (T, H, W, 3)    │    │  (T, 3, H, W) float32 [0,1]   │
└──────────────┘    └─────────────────┘    └────────────────────────────────┘
```

### The Modality Gap — Why Different Projections are Needed

```
Raw Space                   Shared Embedding Space

  Text tokens ─────────────▶ ┌──────────────────────┐
  (discrete integers)    ┌──▶│                      │
                         │   │   "a cat sitting     │
  Image pixels ──────────┘   │    on a mat"         │
  (spatial float tensor)     │                      │◀── Contrastive loss
                         ┌──▶│   [photo of cat]     │    pulls matching
  Audio mel-spec ────────┘   │                      │    pairs together
  (spectral float tensor)    └──────────────────────┘

Each modality needs its own encoder to project into the shared space.
CLIP (Ch.3) is the canonical example of this alignment.
```

---

## 6 · What Changes at Scale

| Concern | Small scale (this laptop) | Production scale |
|---------|--------------------------|-----------------|
| Image resolution | 224×224 (ViT-B standard) | 512–4096+ for generation |
| Video frames | 8–16 per clip | 100s of frames; hierarchical sampling |
| Data type | float32 | bfloat16 or float16 (4× memory saving) |
| Normalisation | ImageNet stats | Dataset-specific or per-batch |
| Batching | Single image | Batches of 1024+; dynamic padding |
| Storage | Local files | Streaming from object store (S3/GCS) |

**The most important production change:** diffusion models almost always work in **float16** or **bfloat16** to fit on consumer GPUs. This is why `torch_dtype=torch.float16` appears in almost every `diffusers` example.

---

## 7 · Common Misconceptions

**"Images are just passed directly to the model"**
They are not. The pixel values are normalised, resized, and batched before the model sees them. The wrong normalisation will silently degrade performance — the model still runs, it just produces worse outputs.

**"The model sees the whole image at once"**
For ViT and diffusion U-Nets, the image is split into patches first. The model processes a *sequence of patch tensors*, not the image as a 2-D grid. (Chapter 2 covers this.)

**"Higher bit depth = always better"**
float16 models often match float32 models in perceptual quality, and the 2× memory reduction enables running Stable Diffusion on a 6–8 GB consumer GPU that would OOM in float32.

**"Audio and images are completely different — you can't use the same model"**
Mel spectrograms resemble images: 2-D, spatial patterns, smooth gradients. Many audio models (Whisper, AudioSpectrogram Transformer) apply an image-based ViT architecture directly to the spectrogram. The key insight is that the *representation* matters more than the *modality*.

---

## 8 · Interview Checklist

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

## 9 · What's Next

→ **[VisionTransformers.md](../VisionTransformers/VisionTransformers.md)** — now that you have a `(3, H, W)` image tensor, the next question is: how does the model process it? A ViT splits the image into patches and applies self-attention over the patch sequence. This is the foundational architecture behind CLIP's image encoder, Stable Diffusion's U-Net, and every multimodal LLM's visual frontend.
