# Multimodal AI / Advanced DL / Math / Interview Guides — Authoring Guide Update Plans

## Multimodal AI Track

**Target:** `notes/05-multimodal_ai/authoring-guide.md`  
**Effort:** 3-4 hours  
**LLM Calls:** 2-3

### Grand Challenge (Preserved)
Cross-modal synthesis: Given text prompt, generate image; given audio, generate captions; given video, detect events. Unified representation learning across modalities.

### Workflow-Based Chapters Identified

#### High Priority Workflow Chapters

**Ch.2 Audio Processing Pipeline**
- **Phases:** Load & Validate → Preprocess (resample, normalize) → Extract Features (spectrograms, MFCCs) → Model Selection
- **Decision Checkpoints:**
  - Audio quality validation (sample rate, bit depth, clipping detection)
  - Feature extraction strategy (task-dependent: speech vs music vs environmental sounds)
  - Preprocessing pipeline selection (mono vs stereo, resampling strategy)
- **Industry Tools:** `librosa`, `torchaudio`, `soundfile`, `pydub`

**Ch.5 Image Preprocessing Workflow**
- **Phases:** Format Validation → Resize Strategy → Color Space Transform → Augmentation Selection
- **Decision Checkpoints:**
  - Aspect ratio preservation vs center crop
  - Normalization strategy (ImageNet stats vs dataset-specific)
  - Augmentation intensity based on dataset size
- **Industry Tools:** `torchvision.transforms`, `albumentations`, `opencv-python`, `Pillow`

**Ch.8 Video Pipeline (Temporal Modeling)**
- **Phases:** Frame Extraction → Temporal Sampling → Optical Flow → Action Detection
- **Decision Checkpoints:**
  - Frame sampling rate (uniform vs non-uniform)
  - Flow algorithm selection (Farneback vs RAFT)
  - Temporal window size based on action duration
- **Industry Tools:** `opencv-python` (VideoCapture), `av`, `decord`, `torchvision.io`

**Ch.11 Fine-Tuning Strategy Selector**
- **Phases:** Architecture Selection → Layer Freezing → Hyperparameter Grid → Validation Strategy
- **Decision Checkpoints:**
  - Full fine-tuning vs LoRA vs prompt tuning
  - Which layers to freeze (task similarity vs data size)
  - Learning rate schedule (warmup duration, decay strategy)
- **Industry Tools:** `transformers` (Trainer), `peft` (LoRA), `accelerate`, `deepspeed`

#### Medium Priority Workflow Chapters

**Ch.15 Multimodal Fusion Strategies**
- **Phases:** Early Fusion vs Late Fusion → Attention Mechanism Selection → Calibration
- **Decision Checkpoints:** Modality alignment, fusion layer placement, temperature scaling
- **Industry Tools:** `torch.nn.MultiheadAttention`, custom fusion layers

### Code Snippet Guidelines (Multimodal-Adapted)

**Rule 1: Each phase shows full preprocessing pipeline with file I/O**

```python
# ✅ Good: Audio preprocessing phase showing inspection
import librosa
import numpy as np

audio_path = "sample.wav"
y, sr = librosa.load(audio_path, sr=None)  # Keep original sample rate

# PHASE 1: Validate quality
print(f"Duration: {len(y)/sr:.2f}s")
print(f"Sample rate: {sr} Hz")
clipping_ratio = np.mean(np.abs(y) > 0.99)

# DECISION LOGIC
if sr < 16000:
    print(f"❌ SEVERE: Sample rate {sr}Hz too low for speech → Resample to 16kHz")
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr = 16000
elif clipping_ratio > 0.01:
    print(f"⚠️ HIGH: {clipping_ratio*100:.1f}% clipped samples → Apply normalization")
    y = librosa.util.normalize(y)
else:
    print(f"✅ SAFE: Audio quality acceptable")
```

**Rule 2: Show modality-specific decision thresholds**

```python
# ✅ Good: Image resize strategy
from PIL import Image

img = Image.open("input.jpg")
width, height = img.size
aspect_ratio = width / height

# DECISION LOGIC for resize
if 0.9 < aspect_ratio < 1.1:  # Nearly square
    strategy = "Center crop to 224×224"
    img_processed = img.resize((224, 224))
elif aspect_ratio > 2.0:  # Panoramic
    strategy = "Preserve aspect, pad to 224×224"
    img_processed = ImageOps.pad(img, (224, 224))
else:
    strategy = "Resize shortest side to 224, crop longest"
    img_processed = ImageOps.fit(img, (224, 224))

print(f"Aspect {aspect_ratio:.2f} → {strategy}")
```

**Rule 3: Show both manual and library approach for preprocessing**

```python
# Manual (learning): MFCC from scratch
def compute_mfcc_manual(audio, sr, n_mfcc=13):
    # Step-by-step: STFT → Mel filterbank → DCT
    stft = librosa.stft(audio)
    mel = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr)
    mfcc = scipy.fftpack.dct(librosa.power_to_db(mel), axis=0)[:n_mfcc]
    return mfcc

# Industry standard (production)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # One line!
```

**Rule 4: Include real multimodal datasets as running examples**

- Audio: LibriSpeech sample, ESC-50 environmental sounds
- Image: COCO subset, ImageNet class examples
- Video: Kinetics-400 clips (10-second samples)
- Text-Image: MS-COCO captions

### Industry Tools Decision Framework

#### When to Show From-Scratch Implementation
- Core concepts (attention mechanism, diffusion forward/reverse process)
- Mathematical foundations (VAE loss derivation, flow field computation)
- Educational notebooks (build intuition for cross-modal alignment)

#### When to Show Library-Based Implementation
- Production pipelines (always use `diffusers` for diffusion models)
- Inference optimization (xFormers, fp16, flash attention)
- Pre-trained models (CLIP, Whisper, Stable Diffusion, SAM)

#### Required Library Coverage by Chapter

| Chapter | Manual Implementation | Industry Standard |
|---------|----------------------|-------------------|
| Ch.2 Audio | Spectrogram from STFT | `librosa.feature.melspectrogram()` |
| Ch.4 Diffusion | DDPM forward/reverse | `diffusers.StableDiffusionPipeline` |
| Ch.6 Vision Transformers | Patch embedding, attention | `transformers.ViTModel.from_pretrained()` |
| Ch.8 Video | Frame extraction loop | `torchvision.io.read_video()` |
| Ch.10 CLIP | Contrastive loss | `transformers.CLIPModel.from_pretrained()` |
| Ch.13 ControlNet | Canny edge detection | `diffusers.StableDiffusionControlNetPipeline` |

#### Core Libraries with Version Pins
```python
# Required for production multimodal pipelines
diffusers==0.25.0           # Stable Diffusion, DDPM, schedulers
transformers==4.36.0        # CLIP, ViT, Whisper, multimodal encoders
torch==2.1.0                # Core tensor operations
torchaudio==2.1.0           # Audio I/O, transforms
torchvision==0.16.0         # Image/video I/O, transforms
librosa==0.10.1             # Audio feature extraction
opencv-python==4.8.1        # Image preprocessing, Canny edges
soundfile==0.12.1           # Audio file I/O
Pillow==10.1.0              # Image loading
accelerate==0.25.0          # Multi-GPU training
xformers==0.0.23            # Memory-efficient attention
```

### Notebook Exercise Pattern (Multimodal-Adapted)

#### Industry Standard Callout Pattern

```markdown
> 💡 **Industry Standard:** After implementing MFCC manually, use:
> ```python
> import librosa
> mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
> delta_mfcc = librosa.feature.delta(mfcc)  # Velocity features
> ```
> **When to use:** Always in production. Manual implementation shown for understanding DSP fundamentals.
> **Common alternatives:** `torchaudio.transforms.MFCC` (PyTorch-native), `python_speech_features.mfcc` (legacy)
> **Performance:** librosa ~50× faster than manual NumPy loops for batch processing.
```

**Callout frequency:** 4-6 per notebook (audio, image, video, fusion, inference)

#### Decision Logic Templates for Multimodal Workflows

**Template 1: Audio Quality Validation**
```markdown
**Decision Logic Template: Audio Preprocessing**

\```python
for audio_file in dataset:
    y, sr = librosa.load(audio_file, sr=None)
    
    # DECISION LOGIC
    snr = compute_snr(y)  # Signal-to-noise ratio
    
    if snr < 10:
        action = "❌ REJECT - Noise too high, exclude from training"
    elif snr < 20:
        action = "⚠️ DENOISE - Apply spectral gating (noisereduce library)"
    else:
        action = "✅ ACCEPT - Clean audio, proceed"
    
    print(f"{audio_file:20s}  SNR={snr:.1f}dB  {action}")
\```

**Thresholds:**
- SNR < 10 dB → Reject (speech unintelligible)
- SNR 10-20 dB → Denoise (background noise audible)
- SNR > 20 dB → Accept (clean recording)
```

**Template 2: Image Preprocessing Strategy**
```markdown
**Decision Logic Template: Resize Strategy Selection**

\```python
for img_path in image_dataset:
    img = Image.open(img_path)
    w, h = img.size
    aspect = w / h
    
    # DECISION LOGIC based on aspect ratio
    if 0.75 <= aspect <= 1.33:  # Near-square (e.g., 4:3, 1:1, 3:4)
        strategy = "resize"
        img_out = img.resize((224, 224))
    elif aspect > 2.0 or aspect < 0.5:  # Extreme (panorama, portrait)
        strategy = "pad_to_square"
        img_out = ImageOps.pad(img, (224, 224), color="black")
    else:
        strategy = "crop_center"
        img_out = ImageOps.fit(img, (224, 224))
    
    print(f"{img_path:25s}  {w}×{h} (AR={aspect:.2f})  → {strategy}")
\```
```

**Template 3: Fine-Tuning Strategy**
```markdown
**Decision Logic Template: Layer Freezing Decision**

\```python
# Given: pretrained_model, task_similarity_score (0-1), dataset_size

if dataset_size < 1000:
    if task_similarity_score > 0.8:
        strategy = "Freeze all, train classifier only"
        frozen_layers = list(range(len(model.layers) - 1))
    else:
        strategy = "❌ INSUFFICIENT DATA - Use few-shot learning instead"
elif dataset_size < 10000:
    strategy = "Freeze encoder, fine-tune last 2 blocks + classifier"
    frozen_layers = list(range(len(model.layers) - 3))
else:
    strategy = "Full fine-tuning with low LR"
    frozen_layers = []

print(f"Dataset: {dataset_size} samples, Similarity: {task_similarity_score:.2f}")
print(f"Strategy: {strategy}")
\```

**Thresholds:**
- < 1k samples + high similarity → Freeze all but head
- 1k-10k samples → Freeze early layers, fine-tune late layers
- > 10k samples → Full fine-tuning with careful LR
```

### Decision Checkpoint Pattern (Multimodal-Specific)

**Example: After Audio Preprocessing Phase**

```markdown
### 2.3 DECISION CHECKPOINT — Audio Validated & Features Extracted

**What you just saw:**
- Loaded 50 audio files: 45 at 16kHz, 5 at 8kHz (upsampled)
- SNR distribution: 12 files < 20dB (denoised), 3 files < 10dB (rejected)
- MFCC shape: (13, T) where T varies by duration (1.5-5.0 seconds)

**What it means:**
- 94% of dataset (47/50) is production-ready after automated QA
- Denoising improved SNR by average 8dB (spectral gating removed stationary noise)
- Variable sequence lengths require padding or dynamic batching for training

**What to do next:**
→ **For training:** Pad MFCCs to fixed length (max_T=500 frames) with zero-padding
→ **For inference:** Use dynamic batching (group similar lengths to minimize padding waste)
→ **Data augmentation:** Apply time stretch (0.9-1.1×), pitch shift (±2 semitones), background noise injection
→ **Next phase:** Move to §3 Temporal Modeling (LSTM/Transformer encoder)
```

### Files Modified Summary

**Primary authoring guide:** `notes/05-multimodal_ai/authoring-guide.md`
- Add Workflow-Based Chapter Pattern section (4 workflow chapters identified)
- Add Code Snippet Guidelines (4 rules, multimodal-adapted)
- Add Industry Tools Decision Framework (manual vs library criteria)
- Add Notebook Exercise Pattern (3 decision logic templates)
- Add Decision Checkpoint examples

**Chapters requiring workflow restructure:** (Future work)
- Ch.2 Audio Processing Pipeline
- Ch.5 Image Preprocessing Workflow
- Ch.8 Video Pipeline
- Ch.11 Fine-Tuning Strategy Selector

**Estimated LOC:** +450 lines to authoring guide

---

## Advanced Deep Learning Track

**Target:** `notes/02-advanced_deep_learning/authoring-guide.md`  
**Effort:** 3-4 hours  
**LLM Calls:** 2-3

### Grand Challenge (Preserved)
Architecture mastery: Design, implement, and optimize state-of-the-art deep learning architectures (ResNets, Transformers, GANs, VAEs). Understand training dynamics, convergence, and scaling to production.

### Workflow-Based Chapters Identified

#### High Priority Workflow Chapters

**Ch.0 Hyperparameter Search Strategy**
- **Phases:** Search Space Definition → Strategy Selection (Grid/Random/Bayesian) → Resource Allocation → Convergence Analysis
- **Decision Checkpoints:**
  - Search space size estimation (combinatorial explosion check)
  - Budget allocation (trials vs compute per trial)
  - Early stopping criteria (convergence detection)
- **Industry Tools:** `Optuna`, `Ray Tune`, `Weights & Biases Sweeps`, `Hydra`

**Ch.X Architecture Search Workflow (NAS)**
- **Phases:** Search Space Design → Search Strategy → Performance Estimation → Retraining
- **Decision Checkpoints:**
  - Cell-based vs global search
  - One-shot vs evolutionary vs RL-based
  - Proxy task accuracy correlation
- **Industry Tools:** `NAS-Bench-201`, `AutoGluon`, `Google Vertex AI AutoML`

**Ch.Y Distributed Training Setup**
- **Phases:** Parallelism Strategy → Communication Backend → Gradient Sync → Fault Tolerance
- **Decision Checkpoints:**
  - Data parallel vs model parallel vs pipeline parallel
  - Batch size scaling (linear scaling rule)
  - Gradient accumulation steps
- **Industry Tools:** `torch.distributed` (DDP), `DeepSpeed` (ZeRO), `FSDP`, `Horovod`

**Ch.Z Training Diagnostics & Debugging**
- **Phases:** Loss Inspection → Gradient Flow Analysis → Activation Distribution → Bottleneck Profiling
- **Decision Checkpoints:**
  - Loss plateau detection (convergence vs stuck)
  - Vanishing/exploding gradients (norm monitoring)
  - Overfitting vs underfitting (validation gap analysis)
- **Industry Tools:** `TensorBoard`, `WandB`, `torch.profiler`, `NVIDIA Nsight`

#### Medium Priority Workflow Chapters

**Ch.N Model Compression Pipeline**
- **Phases:** Pruning → Quantization → Distillation → Deployment
- **Decision Checkpoints:** Accuracy-size tradeoff, hardware target (edge vs cloud)
- **Industry Tools:** `torch.quantization`, `ONNX Runtime`, `TensorRT`, `Distiller`

### Code Snippet Guidelines (Advanced DL-Adapted)

**Rule 1: Show architecture implementation phases with progressive building**

```python
# ✅ Good: ResNet block implementation with decision points

import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with optional downsampling"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # PHASE 1: Main path (conv → BN → ReLU → conv → BN)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # PHASE 2: Skip connection (identity or projection)
        self.downsample = downsample  # None or Conv+BN for dimension matching
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # DECISION LOGIC: Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Match dimensions
        
        out += identity  # Residual addition
        out = self.relu(out)
        return out

# Usage showing decision point
def make_layer(in_channels, out_channels, blocks, stride=1):
    # DECISION: First block needs downsampling if stride > 1 or channels change
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(ResidualBlock(out_channels, out_channels))
    
    return nn.Sequential(*layers)
```

**Rule 2: Hyperparameter search decision trees in code**

```python
# ✅ Good: Bayesian optimization with Optuna

import optuna

def objective(trial):
    # PHASE 1: Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
    
    # DECISION LOGIC: Adjust weight decay based on optimizer
    if optimizer_name == 'AdamW':
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    else:
        weight_decay = 0.0  # Not used for Adam/SGD in this example
    
    # PHASE 2: Train model with suggested hyperparameters
    model = build_model()
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    val_acc = train_and_validate(model, optimizer, batch_size, epochs=10)
    
    # DECISION CHECKPOINT: Early stopping if clearly bad
    if val_acc < 0.6:  # Below baseline
        print(f"⚠️ Trial {trial.number}: Val acc {val_acc:.3f} < 0.6 → Pruning")
        raise optuna.TrialPruned()
    
    return val_acc

# Run study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)

print(f"✅ Best trial: {study.best_trial.params}")
print(f"   Best val acc: {study.best_value:.3f}")
```

**Rule 3: Show manual implementation, then production library**

```python
# Manual (learning): Custom gradient clipping
def clip_gradients_manual(model, max_norm=1.0):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

# Industry standard (production)
import torch.nn.utils as nn_utils
total_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Rule 4: Include real training scenarios with decision branches**

```python
# ✅ Good: Training loop with convergence detection

best_val_loss = float('inf')
patience_counter = 0
patience = 5

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # DECISION LOGIC: Learning rate scheduling
    if epoch > 0 and val_loss > prev_val_loss:
        patience_counter += 1
        print(f"⚠️ Val loss increased: {prev_val_loss:.4f} → {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"❌ EARLY STOP: No improvement for {patience} epochs")
            break
    else:
        patience_counter = 0  # Reset
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New best: {val_loss:.4f}")
    
    prev_val_loss = val_loss
```

### Industry Tools Decision Framework

#### When to Show From-Scratch Implementation
- Core architecture components (attention, residual connections, normalization layers)
- Training algorithms (SGD, Adam, learning rate schedules)
- Mathematical foundations (backprop, loss functions)
- Educational notebooks (build intuition for gradient flow, convergence)

#### When to Show Library-Based Implementation
- Production training pipelines (always use `pytorch-lightning` or `transformers.Trainer`)
- Pre-trained models (use `torchvision.models`, `timm`, `transformers.AutoModel`)
- Distributed training (use `torch.distributed.DDP`, `DeepSpeed`)
- Hyperparameter optimization (use `Optuna`, `Ray Tune`)

#### Required Library Coverage by Chapter

| Chapter | Manual Implementation | Industry Standard |
|---------|----------------------|-------------------|
| Ch.1 CNNs | Conv2d, pooling layers | `torchvision.models.resnet50()` |
| Ch.2 ResNets | Residual block, skip connections | `timm.create_model('resnet50')` |
| Ch.3 Inception | Inception module | `torchvision.models.inception_v3()` |
| Ch.4 MobileNets | Depthwise separable conv | `timm.create_model('mobilenetv3')` |
| Ch.5 GANs | Generator, Discriminator, training loop | `torch-gan`, `StyleGAN2-ADA` |
| Ch.6 VAEs | Encoder, Decoder, reparameterization | `pytorch-lightning` (structured training) |
| Ch.7 Transformers | Multi-head attention, positional encoding | `transformers.AutoModel.from_pretrained()` |
| Ch.8 ViT | Patch embedding, transformer encoder | `timm.create_model('vit_base_patch16_224')` |
| Ch.9 BERT | Masked LM, tokenization | `transformers.BertModel.from_pretrained()` |
| Ch.10 GPT | Autoregressive decoder | `transformers.GPT2Model.from_pretrained()` |

#### Core Libraries with Version Pins
```python
# Required for advanced DL production workflows
torch==2.1.0                # Core framework
torchvision==0.16.0         # Pretrained vision models
timm==0.9.12                # PyTorch Image Models (hundreds of architectures)
transformers==4.36.0        # Pretrained NLP/multimodal models
pytorch-lightning==2.1.3    # Structured training, distributed, callbacks
optuna==3.5.0               # Hyperparameter optimization
ray[tune]==2.9.0            # Distributed HPO, model serving
wandb==0.16.2               # Experiment tracking, sweeps
tensorboard==2.15.1         # Visualization
deepspeed==0.12.6           # ZeRO optimizer, pipeline parallelism
accelerate==0.25.0          # Multi-GPU/TPU training utilities
onnx==1.15.0                # Model export for deployment
tensorrt==8.6.1             # NVIDIA inference optimization
```

### Notebook Exercise Pattern (Advanced DL-Adapted)

#### Industry Standard Callout Pattern

```markdown
> 💡 **Industry Standard:** After implementing ResNet block manually, use:
> ```python
> import torchvision.models as models
> model = models.resnet50(pretrained=True)  # Load pretrained weights
> model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
> ```
> **When to use:** Always in production. Manual implementation shown to understand skip connections and gradient flow.
> **Common alternatives:** `timm.create_model('resnet50', pretrained=True, num_classes=10)` (more architectures)
> **Performance:** Pretrained ImageNet weights → 10-20% better accuracy on small datasets (transfer learning)
> **See also:** [timm docs](https://timm.fast.ai/) for 700+ architectures
```

**Callout frequency:** 5-7 per notebook (architecture loading, training loop, optimization, distributed, monitoring)

#### Decision Logic Templates for Training Workflows

**Template 1: Learning Rate Scheduling**
```markdown
**Decision Logic Template: LR Schedule Selection**

\```python
# Given: initial_lr, epochs, dataset_size, batch_size

steps_per_epoch = dataset_size // batch_size
total_steps = steps_per_epoch * epochs

# DECISION LOGIC based on training regime
if epochs < 10:  # Short training
    strategy = "Constant LR with decay at 50%, 75%"
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs//2, int(0.75*epochs)], gamma=0.1
    )
elif total_steps < 10000:  # Medium-scale
    strategy = "Cosine annealing (smooth decay)"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
else:  # Large-scale (Transformers)
    warmup_steps = min(10000, total_steps // 10)
    strategy = f"Linear warmup ({warmup_steps} steps) + cosine decay"
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

print(f"Training regime: {epochs} epochs, {total_steps} steps")
print(f"Strategy: {strategy}")
\```

**Guidelines:**
- Short training (< 10 epochs): Step decay at milestones
- Medium (10-100 epochs): Cosine annealing
- Long (Transformers): Warmup + cosine (prevents early divergence)
```

**Template 2: Distributed Training Strategy**
```markdown
**Decision Logic Template: Parallelism Selection**

\```python
import torch
import psutil

num_gpus = torch.cuda.device_count()
model_params = sum(p.numel() for p in model.parameters())
model_size_gb = model_params * 4 / (1024**3)  # FP32
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

# DECISION LOGIC
if num_gpus == 1:
    strategy = "Single GPU (no parallelism)"
    distributed_model = model.cuda()
    
elif model_size_gb < gpu_memory_gb * 0.7:  # Model fits on single GPU
    strategy = "Data Parallel (DDP)"
    model = model.cuda()
    distributed_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )
    print(f"✅ Batch size scales by {num_gpus}× (effective batch = {batch_size * num_gpus})")
    
else:  # Model too large for single GPU
    strategy = "Fully Sharded Data Parallel (FSDP)"
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    distributed_model = FSDP(model)
    print(f"⚠️ Model sharded across {num_gpus} GPUs (memory efficient)")

print(f"Model: {model_size_gb:.2f} GB, GPU: {gpu_memory_gb:.2f} GB")
print(f"Strategy: {strategy}")
\```

**Thresholds:**
- 1 GPU → No parallelism
- Model < 70% GPU memory → DDP (fastest, data parallel)
- Model > 70% GPU memory → FSDP or DeepSpeed ZeRO (model sharding)
```

**Template 3: Convergence Diagnosis**
```markdown
**Decision Logic Template: Training Diagnostic**

\```python
# Collect metrics over recent epochs
recent_train_losses = train_losses[-5:]
recent_val_losses = val_losses[-5:]

train_loss_slope = (recent_train_losses[-1] - recent_train_losses[0]) / 5
val_loss_slope = (recent_val_losses[-1] - recent_val_losses[0]) / 5
val_gap = recent_val_losses[-1] - recent_train_losses[-1]

# DECISION LOGIC
if train_loss_slope > 0.01:  # Increasing train loss
    diagnosis = "❌ DIVERGENCE - Reduce LR or check gradients"
    action = "Reduce LR by 10×, check for NaN gradients"
    
elif val_loss_slope > 0.01 and val_gap > 0.5:  # Val increasing, large gap
    diagnosis = "❌ SEVERE OVERFITTING"
    action = "Add regularization (dropout, weight decay), reduce model capacity"
    
elif val_loss_slope > 0.005 and val_gap > 0.2:  # Mild overfitting
    diagnosis = "⚠️ OVERFITTING - Early stopping recommended"
    action = "Stop training, restore best checkpoint"
    
elif abs(train_loss_slope) < 0.001 and abs(val_loss_slope) < 0.001:
    diagnosis = "✅ CONVERGED"
    action = "Training complete, evaluate on test set"
    
else:
    diagnosis = "🔄 TRAINING - Continue"
    action = "No action needed"

print(f"Train Δ: {train_loss_slope:+.4f}, Val Δ: {val_loss_slope:+.4f}, Gap: {val_gap:.3f}")
print(f"Diagnosis: {diagnosis}")
print(f"Action: {action}")
\```
```

### Decision Checkpoint Pattern (Training-Specific)

**Example: After Hyperparameter Search Phase**

```markdown
### X.3 DECISION CHECKPOINT — Hyperparameter Search Complete

**What you just saw:**
- Ran 50 Optuna trials over 2 hours (Bayesian TPE sampler)
- Best trial: LR=0.0012, batch_size=64, optimizer=AdamW, weight_decay=0.008
- Val accuracy improved from baseline 78.2% → 83.5% (+5.3pp)
- 12 trials pruned early (val_acc < 60% after 3 epochs)

**What it means:**
- Search space was well-defined (no wasted trials on extreme values)
- AdamW consistently outperformed SGD (42 of top 50 trials used AdamW)
- Optimal LR is ~10× lower than default (0.001 vs 0.01) for this dataset
- Batch size 64 balances speed and generalization (larger batches → worse val acc)

**What to do next:**
→ **Retrain with best hyperparams:** Use full training set (was 80% during search)
→ **Extend training:** 50 epochs instead of 10 (search used short runs)
→ **Add augmentation:** Now that base config works, add MixUp or CutMix
→ **Monitor for overfitting:** Val gap was 2.1% in best trial, watch for increase
→ **Next phase:** Move to §4 Distributed Training (scale to multi-GPU)
```

### Files Modified Summary

**Primary authoring guide:** `notes/02-advanced_deep_learning/authoring-guide.md`
- Add Workflow-Based Chapter Pattern section (4-5 workflow chapters identified)
- Add Code Snippet Guidelines (4 rules, training-focused)
- Add Industry Tools Decision Framework (manual vs library for architectures, training, optimization)
- Add Notebook Exercise Pattern (3 decision logic templates: LR, distributed, diagnostics)
- Add Decision Checkpoint examples (hyperparameter search, training convergence)

**Chapters requiring workflow restructure:** (Future work)
- Ch.0 Hyperparameter Search Strategy
- Ch.X Architecture Search Workflow
- Ch.Y Distributed Training Setup
- Ch.Z Training Diagnostics & Debugging

**Estimated LOC:** +500 lines to authoring guide

---

## Math Under the Hood Track

**Target:** `notes/00-math_under_the_hood/authoring-guide.md`  
**Effort:** 2-3 hours  
**LLM Calls:** 2

### Grand Challenge (Preserved)
Mathematical foundations: Derive loss functions, optimization algorithms, backpropagation, regularization techniques. Understand why ML works mathematically, not just how to code it.

### Workflow-Based Chapters Identified

**Important note:** Math track is predominantly **concept-based** (proofs, derivations, theorems). Only a few chapters benefit from workflow structure.

#### Limited Workflow Chapters (2-3 total)

**Ch.X Numerical Optimization Workflow**
- **Phases:** Problem Formulation → Gradient Computation → Step Size Selection → Convergence Check
- **Decision Checkpoints:**
  - Exact gradient (analytic) vs approximate (finite differences, autodiff)
  - Line search vs fixed step size vs adaptive (Adam)
  - Convergence criteria (gradient norm, function value change, iteration limit)
- **Industry Tools:** `scipy.optimize`, `torch.optim`, `JAX` (autodiff)

**Ch.Y Matrix Factorization Selection**
- **Phases:** Problem Characterization → Factorization Choice → Numerical Stability → Application
- **Decision Checkpoints:**
  - Dense vs sparse matrix (storage, algorithm choice)
  - Eigendecomposition vs SVD vs Cholesky vs QR
  - Conditioning check (condition number for stability)
- **Industry Tools:** `numpy.linalg`, `scipy.linalg`, `scipy.sparse.linalg`

**Most chapters remain concept-based:**
- Linear algebra fundamentals (vectors, matrices, eigenvalues)
- Calculus foundations (derivatives, chain rule, gradients)
- Probability theory (distributions, expectations, MLE)
- Loss function derivations (cross-entropy, MSE, hinge loss)
- Backpropagation derivation (computational graph, chain rule application)

### Code Snippet Guidelines (Math-Adapted)

**Core principle:** Show manual derivation/computation first (build mathematical intuition), then efficient library implementation.

**Rule 1: Manual implementation shows the mathematical steps explicitly**

```python
# ✅ Good: Gradient descent from first principles

import numpy as np

def gradient_descent_manual(f, grad_f, x0, lr=0.01, max_iters=100, tol=1e-6):
    """
    Minimize f(x) using gradient descent.
    
    Args:
        f: Objective function
        grad_f: Gradient of f (analytic)
        x0: Initial point
        lr: Learning rate (step size)
        max_iters: Maximum iterations
        tol: Convergence tolerance (gradient norm)
    """
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iters):
        # PHASE 1: Compute gradient
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        # DECISION CHECKPOINT: Check convergence
        if grad_norm < tol:
            print(f"✅ CONVERGED at iteration {i}: ||∇f|| = {grad_norm:.2e}")
            break
        
        # PHASE 2: Update step
        x_new = x - lr * grad
        
        # PHASE 3: Validate improvement
        f_old, f_new = f(x), f(x_new)
        
        if f_new > f_old:  # Function increased
            print(f"⚠️ Iter {i}: f increased {f_old:.4f} → {f_new:.4f} (LR too large?)")
        
        x = x_new
        history.append(x.copy())
        
        if i % 10 == 0:
            print(f"Iter {i:3d}: f(x) = {f_new:.6f}, ||∇f|| = {grad_norm:.2e}")
    
    return x, np.array(history)

# Example: Minimize f(x) = x^2 + 3x + 2
f = lambda x: x**2 + 3*x + 2
grad_f = lambda x: 2*x + 3  # Analytic gradient

x_opt, history = gradient_descent_manual(f, grad_f, x0=np.array([5.0]))
# ✅ CONVERGED at iteration 23: ||∇f|| = 9.54e-07
# Optimal: x = -1.5, f(x) = -0.25
```

**Rule 2: Show numerical stability considerations**

```python
# ✅ Good: Matrix inversion with condition number check

A = np.array([[1.0, 0.99], [0.99, 0.98]])  # Ill-conditioned

# PHASE 1: Check conditioning
cond = np.linalg.cond(A)

# DECISION LOGIC
if cond > 1e10:
    print(f"❌ SEVERE: Condition number {cond:.2e} → Matrix nearly singular")
    print("   Solution: Use regularization (ridge) or pseudoinverse")
    A_reg = A + 1e-6 * np.eye(len(A))  # Tikhonov regularization
    x = np.linalg.solve(A_reg, b)
elif cond > 1e5:
    print(f"⚠️ HIGH: Condition number {cond:.2e} → Numerical instability likely")
    print("   Solution: Use stable method (SVD-based solver)")
    x = np.linalg.lstsq(A, b, rcond=None)[0]  # SVD-based
else:
    print(f"✅ SAFE: Condition number {cond:.2e}")
    x = np.linalg.solve(A, b)  # Standard solver

print(f"Condition number: {cond:.2e}")
```

**Rule 3: Manual → library pattern for common operations**

```python
# Manual (learning): Matrix multiplication from definition
def matmul_manual(A, B):
    """A: (m, n), B: (n, p) → C: (m, p)"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Incompatible dimensions"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]  # Dot product
    return C

# NumPy (production): Vectorized, 100-1000× faster
C = np.dot(A, B)  # Or A @ B

# For large matrices: Use BLAS-optimized routines
C = np.matmul(A, B)  # Calls optimized BLAS (Intel MKL, OpenBLAS)
```

**Rule 4: Include mathematical derivation in comments**

```python
# ✅ Good: Cross-entropy loss with derivation

def cross_entropy_loss_manual(y_true, y_pred):
    """
    Cross-entropy: L = -Σ y_true[i] * log(y_pred[i])
    
    Derivation:
    1. Binary classification: L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    2. Multiclass (K classes): L = -Σ_{k=1}^K y_k * log(ŷ_k)
    3. Expected form: ŷ from softmax, y one-hot encoded
    
    Gradient: ∂L/∂logits = ŷ - y (remarkably simple!)
    """
    # Numerical stability: Add epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # For one-hot encoded y_true, simplifies to -log(y_pred[correct_class])
    loss = -np.sum(y_true * np.log(y_pred_clipped))
    
    return loss

# Industry standard (PyTorch)
import torch.nn.functional as F
loss = F.cross_entropy(logits, targets)  # Combines log_softmax + NLL
```

### Industry Tools Decision Framework

#### When to Show Manual Implementation (Majority of Math Track)
- **Always show manual first** for core mathematical concepts
- Derivations requiring step-by-step algebra (backprop, MLE, eigendecomposition)
- Proof-of-concept for theorems (convergence proofs, optimality conditions)
- Educational understanding (how optimization works, why regularization helps)

#### When to Show Library Implementation
- After manual implementation shown at least once in the chapter
- Production-scale computations (large matrices, high-dimensional optimization)
- Numerical stability required (library implementations handle edge cases)
- Performance-critical operations (matrix multiply, SVD, FFT)

#### Required Library Coverage (Minimal - Theory-Focused Track)

| Chapter Type | Manual Implementation | Industry Standard |
|--------------|----------------------|-------------------|
| Linear Algebra | Matrix operations (loops) | `numpy.linalg`, `scipy.linalg` |
| Calculus | Finite differences | `scipy.optimize`, `torch.autograd`, `JAX` |
| Optimization | Gradient descent from scratch | `scipy.optimize.minimize`, `torch.optim` |
| Probability | PDF/CDF from definition | `scipy.stats`, `numpy.random` |
| Statistics | Sample mean, variance | `numpy.mean`, `numpy.cov`, `scipy.stats` |

#### Core Libraries (Theory-Oriented)
```python
# Math track focuses on theory, minimal dependencies
numpy==1.24.0               # Core array operations, linear algebra
scipy==1.11.0               # Optimization, statistics, sparse matrices
sympy==1.12                 # Symbolic mathematics (derivatives, integrals)
matplotlib==3.7.0           # Visualization (loss curves, gradients)

# Optional (for autodiff examples)
torch==2.1.0                # Automatic differentiation (torch.autograd)
jax==0.4.20                 # Functional autodiff (grad, jit)
```

**Library usage philosophy:**
- NumPy/SciPy: After showing manual derivation
- SymPy: For symbolic math (derivatives, simplification) when pedagogy benefits
- Torch/JAX: Only in autodiff chapters, showing gradient computation

### Notebook Exercise Pattern (Math-Adapted)

#### Industry Standard Callout Pattern

```markdown
> 💡 **Industry Standard:** After implementing gradient descent manually, use:
> ```python
> from scipy.optimize import minimize
> result = minimize(f, x0, method='BFGS', jac=grad_f)  # Quasi-Newton method
> x_opt = result.x  # Optimal point
> ```
> **When to use:** Production optimization (faster convergence, handles constraints).
> **Common alternatives:** 
>   - `method='L-BFGS-B'` (bounded optimization, memory-efficient)
>   - `method='trust-constr'` (constrained nonlinear programming)
>   - `torch.optim.Adam` (neural networks, stochastic optimization)
> **Performance:** BFGS typically converges in 10-20 iterations vs 100+ for vanilla GD.
> **See also:** [SciPy optimization guide](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
```

**Callout frequency:** 2-3 per notebook (lower than ML track, since focus is theory)

#### Decision Logic Templates for Math Workflows

**Template 1: Optimization Method Selection**
```markdown
**Decision Logic Template: Choosing Optimization Algorithm**

\```python
# Given: objective function f, gradient grad_f, Hessian hess_f (optional)

problem_size = len(x0)
gradient_available = grad_f is not None
hessian_available = hess_f is not None

# DECISION LOGIC
if problem_size < 100 and hessian_available:
    method = "Newton's method (uses Hessian)"
    # x_new = x - H^(-1) * g  (quadratic convergence)
    
elif problem_size < 1000 and gradient_available:
    method = "BFGS (quasi-Newton, approximate Hessian)"
    # Builds Hessian approximation from gradients
    result = scipy.optimize.minimize(f, x0, method='BFGS', jac=grad_f)
    
elif problem_size < 10000:
    method = "L-BFGS-B (limited-memory BFGS)"
    # Memory-efficient for medium-scale problems
    result = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=grad_f)
    
else:  # Large-scale (> 10k variables)
    method = "Stochastic Gradient Descent"
    # Use for high-dimensional problems (neural networks)
    # Requires minibatch sampling, not full gradient
    
print(f"Problem: {problem_size} variables, Grad: {gradient_available}, Hess: {hessian_available}")
print(f"Method: {method}")
\```

**Guidelines:**
- Small problems (< 100 vars) + Hessian → Newton
- Medium (100-10k) + gradient → (L-)BFGS
- Large (> 10k) → SGD, Adam (stochastic methods)
```

**Template 2: Matrix Factorization Selection**
```markdown
**Decision Logic Template: Factorization Choice**

\```python
import numpy as np

A = get_matrix()  # Your matrix
m, n = A.shape

# PHASE 1: Characterize matrix
is_square = (m == n)
is_symmetric = is_square and np.allclose(A, A.T)
is_positive_definite = is_symmetric and np.all(np.linalg.eigvals(A) > 0)
sparsity = 1 - np.count_nonzero(A) / (m * n)

# DECISION LOGIC
if is_positive_definite:
    method = "Cholesky (A = LL^T)"
    # Fastest for positive definite, numerically stable
    L = np.linalg.cholesky(A)
    print(f"✅ Positive definite → Cholesky (O(n³/3), stable)")
    
elif is_symmetric:
    method = "Eigendecomposition (A = QΛQ^T)"
    # For symmetric, reveals eigenvalues/eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A)  # Symmetric variant (faster)
    print(f"✅ Symmetric → Eigendecomposition")
    
elif is_square:
    method = "LU factorization (A = LU)"
    # For solving Ax = b, determinant
    from scipy.linalg import lu
    P, L, U = lu(A)
    print(f"✅ Square → LU factorization")
    
else:  # Rectangular
    method = "SVD (A = UΣV^T)"
    # Most general, works for any m×n, reveals rank
    U, S, Vt = np.linalg.svd(A)
    print(f"✅ Rectangular ({m}×{n}) → SVD")
    
if sparsity > 0.9:
    print(f"⚠️ Matrix is {sparsity*100:.1f}% sparse → Use scipy.sparse for efficiency")

print(f"Shape: {A.shape}, Symmetric: {is_symmetric}, PD: {is_positive_definite}")
print(f"Method: {method}")
\```

**Decision tree:**
- Positive definite → Cholesky (fastest, stable)
- Symmetric → Eigendecomposition
- Square → LU factorization
- Rectangular → SVD (most general)
```

**Template 3: Numerical Stability Checks**
```markdown
**Decision Logic Template: Numerical Stability Validation**

\```python
# When solving Ax = b, check conditioning

A = get_matrix()
b = get_vector()

# PHASE 1: Compute condition number
cond_number = np.linalg.cond(A)

# DECISION LOGIC
if cond_number > 1e12:
    diagnosis = "❌ SEVERE: Nearly singular matrix"
    action = "Regularize: A_reg = A + λI (Tikhonov), or use pseudoinverse"
    # Tikhonov regularization
    lambda_reg = 1e-6
    A_reg = A + lambda_reg * np.eye(len(A))
    x = np.linalg.solve(A_reg, b)
    
elif cond_number > 1e8:
    diagnosis = "⚠️ HIGH: Ill-conditioned system"
    action = "Use SVD-based solver (np.linalg.lstsq) for stability"
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
elif cond_number > 1e4:
    diagnosis = "⚡ MODERATE: Some numerical error expected"
    action = "Standard solver acceptable, monitor residual ||Ax - b||"
    x = np.linalg.solve(A, b)
    residual = np.linalg.norm(A @ x - b)
    print(f"Residual: {residual:.2e}")
    
else:
    diagnosis = "✅ SAFE: Well-conditioned"
    action = "Standard solver (np.linalg.solve) is fine"
    x = np.linalg.solve(A, b)

print(f"Condition number: {cond_number:.2e}")
print(f"Diagnosis: {diagnosis}")
print(f"Action: {action}")
\```

**Thresholds:**
- κ(A) < 10⁴ → Safe
- 10⁴ < κ(A) < 10⁸ → Monitor residual
- 10⁸ < κ(A) < 10¹² → Use stable solver (SVD)
- κ(A) > 10¹² → Regularize or pseudoinverse
```

### Decision Checkpoint Pattern (Math-Specific)

**Example: After Optimization Convergence**

```markdown
### X.4 DECISION CHECKPOINT — Optimization Converged

**What you just saw:**
- Gradient descent converged in 47 iterations (tolerance ||∇f|| < 10⁻⁶)
- Initial function value: f(x₀) = 12.456 → Final: f(x*) = -0.250
- Learning rate: 0.01 (fixed throughout)
- Gradient norm decreased monotonically: 9.2 → 0.045 → 9.54×10⁻⁷

**What it means:**
- Smooth convergence indicates learning rate was appropriate (not too large/small)
- Monotonic decrease confirms convex problem (no local minima encountered)
- 47 iterations is typical for gradient descent on smooth quadratic functions
- Achieved machine precision (10⁻⁶) - further iterations yield negligible improvement

**What to do next:**
→ **For production:** Use BFGS (quasi-Newton) - converges in ~10 iterations instead of 47
→ **For constrained problems:** Add barrier/penalty methods or switch to `scipy.optimize.minimize`
→ **For stochastic setting:** Extend to SGD with minibatches (Ch.Y Stochastic Optimization)
→ **For non-convex:** Study convergence to saddle points vs local minima (Ch.Z Landscape Analysis)
→ **Next phase:** Move to §5 Newton's Method (quadratic convergence using Hessian)
```

### Files Modified Summary

**Primary authoring guide:** `notes/00-math_under_the_hood/authoring-guide.md`
- Add "Workflow vs Concept-Based Chapters" section (clarify that most chapters are concept-based)
- Add limited Workflow-Based Chapter Pattern (2-3 chapters only: optimization, matrix factorization)
- Add Code Snippet Guidelines (4 rules: manual derivation first, stability checks, manual→library, derivation in comments)
- Add Industry Tools Decision Framework (emphasize "manual first" philosophy)
- Add Notebook Exercise Pattern (3 decision templates: optimization method, factorization, stability)
- Add Decision Checkpoint examples (optimization convergence)

**Chapters requiring workflow restructure:** (Very limited - only 2-3)
- Ch.X Numerical Optimization Workflow
- Ch.Y Matrix Factorization Selection
- (Most chapters remain concept-based: proofs, derivations, theorems)

**Estimated LOC:** +350 lines to authoring guide (less than ML/MultimodalAI since fewer workflow chapters)

---

## Interview Guides Track

**Target:** `notes/interview_guides/authoring-guide.md`  
**Effort:** 2-3 hours  
**LLM Calls:** 2

### Grand Challenge (Preserved)
Interview mastery: Prepare for ML/AI technical interviews through structured problem-solving frameworks, coding patterns, system design blueprints, and behavioral question strategies. Build confidence through repetition and pattern recognition.

### Workflow-Based Chapters (HIGHLY APPLICABLE)

**Key insight:** Interview Guides is the **most workflow-driven track** - nearly every chapter is a decision tree or problem-solving framework.

#### High Priority Workflow Chapters (Majority of Track)

**Ch.1 Problem Type Recognition Workflow**
- **Phases:** Read Problem → Extract Signals → Map to Pattern → Select Algorithm
- **Decision Checkpoints:**
  - Input characteristics (sorted? duplicates? constraints?)
  - Output requirements (optimal? all solutions? count?)
  - Complexity constraints (must be O(n)? space-limited?)
- **Industry Tools:** LeetCode patterns, NeetCode roadmap, Algorithm design templates

**Ch.2 Coding Interview Framework (UMPIRE)**
- **Phases:** Understand → Match → Plan → Implement → Review → Evaluate
- **Decision Checkpoints:**
  - Clarifying questions (edge cases, input bounds)
  - Brute force → Optimized transition
  - Test case selection (edge cases, performance validation)
- **Industry Tools:** Pramp, LeetCode timer, Code submission platforms

**Ch.3 Data Structure Selection Workflow**
- **Phases:** Identify Operations → Frequency Analysis → Complexity Requirements → Structure Selection
- **Decision Checkpoints:**
  - Read-heavy vs write-heavy (array vs linked list)
  - Search-heavy (hash table vs BST)
  - Order matters (heap, priority queue)
- **Industry Tools:** Python `collections`, `heapq`, `bisect`; Time complexity cheat sheets

**Ch.4 Algorithm Pattern Recognition**
- **Phases:** Problem → Pattern → Template → Adaptation
- **Decision Checkpoints:**
  - Two pointers vs sliding window
  - DFS vs BFS (tree/graph traversal)
  - Dynamic programming vs greedy
- **Industry Tools:** LeetCode pattern lists, AlgoExpert categories

**Ch.5 System Design Workflow (RESHADED)**
- **Phases:** Requirements → Estimation → Storage → High-level → API → Detailed → Evaluation
- **Decision Checkpoints:**
  - Read-heavy vs write-heavy (caching strategy)
  - Consistency vs availability (CAP theorem)
  - SQL vs NoSQL (data model, query patterns)
- **Industry Tools:** Back-of-envelope calculator, AWS/GCP architecture patterns, CAP theorem diagram

**Ch.6 Complexity Analysis Framework**
- **Phases:** Identify Loops → Multiply Nested → Recurrence Relations → Simplify
- **Decision Checkpoints:**
  - Time vs space tradeoff decision
  - Amortized analysis (when relevant)
  - Best/average/worst case distinction
- **Industry Tools:** Master theorem, Big-O cheat sheet, Recurrence solver

**Ch.7 Behavioral Interview Framework (STAR)**
- **Phases:** Situation → Task → Action → Result
- **Decision Checkpoints:**
  - Leadership vs IC (individual contributor) stories
  - Conflict resolution strategies
  - Failure stories (what you learned)
- **Industry Tools:** Story bank template, Mock interview platforms

#### Medium Priority Workflow Chapters

**Ch.8 Debugging Strategy**
- **Phases:** Reproduce → Isolate → Hypothesize → Test → Fix
- **Decision Checkpoints:** Print debugging vs IDE breakpoints, binary search bug location

**Ch.9 ML System Design Workflow**
- **Phases:** Problem Formulation → Data Pipeline → Model Selection → Training → Serving → Monitoring
- **Decision Checkpoints:** Batch vs online learning, model complexity vs interpretability

### Code Snippet Guidelines (Interview-Adapted)

**Rule 1: Every pattern includes template with decision comments**

```python
# ✅ Good: Two Pointers pattern template with decision logic

def two_pointers_template(arr):
    """
    Pattern: Two pointers converging from ends
    Use when: Array is sorted, looking for pair/triplet
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        # DECISION LOGIC: Compare current sum to target
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]  # Found!
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return None  # No solution

# Variations:
# - Three sum: Fix one element, two pointers on rest
# - Four sum: Fix two elements, two pointers on rest
# - Container with most water: Max area instead of sum
```

**Rule 2: Show problem → pattern mapping explicitly**

```python
# ✅ Good: Pattern recognition decision tree

def recognize_pattern(problem_description):
    """
    Problem: "Find longest substring without repeating characters"
    
    PATTERN RECOGNITION:
    1. "Substring" → Contiguous elements → Consider sliding window
    2. "Without repeating" → Need to track seen elements → Use set/dict
    3. "Longest" → Optimization problem → Expand window when valid
    
    DECISION: Sliding window + hash set
    Time: O(n), Space: O(k) where k = charset size
    """
    seen = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window
        while s[right] in seen:
            # Contract window until no duplicates
            seen.remove(s[left])
            left += 1
        
        seen.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Key signals for this pattern:
# ✓ "Substring" or "subarray" (contiguous)
# ✓ Optimization (longest, shortest, maximum)
# ✓ Constraint on window contents
```

**Rule 3: Include complexity analysis with trade-off decisions**

```python
# ✅ Good: Show brute force → optimized with complexity reasoning

def two_sum_bruteforce(nums, target):
    """
    Approach 1: Brute force (check all pairs)
    Time: O(n²), Space: O(1)
    
    DECISION: Too slow for n > 10,000
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None

def two_sum_optimized(nums, target):
    """
    Approach 2: Hash table (trade space for time)
    Time: O(n), Space: O(n)
    
    DECISION LOGIC:
    - For each num, check if (target - num) seen before
    - Store seen numbers in hash table for O(1) lookup
    
    Tradeoff: Use O(n) space to achieve O(n) time
    Acceptable? YES - n ≤ 10⁴ → space is cheap
    """
    seen = {}  # {value: index}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return None

# Interview discussion:
# Q: "Can we do better than O(n) time?"
# A: "No - must examine each element at least once (lower bound: Ω(n))"
# Q: "What if we can't use extra space?"
# A: "Then O(n²) brute force, or sort first O(n log n) + two pointers O(n)"
```

**Rule 4: Real interview problem walkthroughs (LeetCode-style)**

```python
# ✅ Good: Full problem walkthrough with interviewer dialogue

"""
Problem: "Merge k Sorted Lists" (LeetCode 23, Hard)

Input: lists = [[1,4,5], [1,3,4], [2,6]]
Output: [1,1,2,3,4,4,5,6]

PHASE 1: UNDERSTAND (ask clarifying questions)
Q: "Can lists be empty?"
A: "Yes, handle edge case"
Q: "Are values unique?"
A: "No, duplicates allowed"
Q: "Memory constraints?"
A: "Prefer in-place, but O(k) extra space acceptable"

PHASE 2: MATCH PATTERN
Signal: "k sorted lists" → Many sorted sequences
Pattern: Min heap (priority queue)
- Extract minimum across k lists efficiently
- Heap maintains next candidate from each list

PHASE 3: PLAN
1. Build min heap with first element from each list
2. Extract min, add to result
3. Add next element from same list to heap
4. Repeat until heap empty

Complexity: O(N log k) where N = total elements, k = num lists
"""

import heapq

def mergeKLists(lists):
    # Edge case
    if not lists or all(not lst for lst in lists):
        return []
    
    # PHASE 1: Initialize heap with first element from each list
    min_heap = []
    for i, lst in enumerate(lists):
        if lst:  # Non-empty list
            heapq.heappush(min_heap, (lst[0], i, 0))  # (value, list_idx, elem_idx)
    
    result = []
    
    # PHASE 2: Extract min, add next element
    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)
        
        # DECISION: Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))
    
    return result

# Follow-up questions (prepare for these):
# Q: "What if k is very large (millions)?"
# A: "Still O(N log k), but consider external merge sort for disk-based data"
# Q: "Can you do it without extra space?"
# A: "Not with heap - would need to merge pairwise (divide & conquer) for O(1) space"
```

### Industry Tools Decision Framework

#### When to Show From-Scratch Implementation
- Core algorithm understanding (binary search, DFS/BFS, DP recurrence)
- Interview white-boarding (must be able to code without IDE)
- Pattern templates (demonstrate pattern, not rely on library)

#### When to Show Library/Built-in
- After manual implementation shown
- Python interview tips (`collections.Counter`, `heapq`, `bisect`)
- Production code (always prefer standard library)

#### Required Tool Coverage by Interview Type

| Interview Type | Core Patterns | Python Built-ins | Complexity |
|----------------|---------------|------------------|------------|
| Arrays/Strings | Two pointers, sliding window | `collections.Counter`, `set` | O(n), O(1) space preferred |
| Trees/Graphs | DFS, BFS, backtracking | `collections.deque` | O(V+E), O(h) recursion |
| Dynamic Programming | Top-down, bottom-up | `functools.lru_cache` | O(n²) or O(n×m) |
| Heaps | Min/max heap | `heapq` | O(log n) operations |
| Sorting/Searching | Binary search, merge sort | `bisect`, `sorted()` | O(n log n) |
| Hash Tables | HashMap patterns | `dict`, `collections.defaultdict` | O(1) average lookup |
| System Design | Load balancing, caching | Redis, Kafka, PostgreSQL | Horizontal scaling |

#### Core "Libraries" for Interviews (Language-Specific)

**Python:**
```python
# Essential built-ins for coding interviews
from collections import (
    Counter,        # Frequency counting
    defaultdict,    # Auto-initialization
    deque,          # Double-ended queue (BFS)
    OrderedDict     # Insertion-order dict
)
import heapq        # Min heap operations
import bisect       # Binary search on sorted list
from functools import lru_cache  # Memoization decorator

# Common patterns
counter = Counter(arr)                    # O(n) frequency map
graph = defaultdict(list)                 # Adjacency list
queue = deque([root])                     # BFS queue
heapq.heappush(heap, (priority, item))    # Heap operations
idx = bisect.bisect_left(sorted_arr, x)   # Binary search
```

**System Design:**
```
# Industry standard tools to mention
- Databases: PostgreSQL (relational), MongoDB (document), Redis (cache)
- Message queues: Kafka, RabbitMQ, AWS SQS
- Load balancing: NGINX, HAProxy, AWS ELB
- Monitoring: Prometheus, Grafana, DataDog
- Containerization: Docker, Kubernetes
- CI/CD: Jenkins, GitHub Actions, CircleCI
```

### Notebook Exercise Pattern (Interview-Adapted)

#### Industry Pattern Callout

```markdown
> 💡 **Interview Tip:** After explaining two-pointer pattern manually, mention Python built-ins:
> ```python
> # Manual two pointers (show in interview)
> left, right = 0, len(arr) - 1
> while left < right:
>     # ...decision logic
> 
> # Production shortcut (mention awareness)
> from itertools import combinations
> for i, j in combinations(range(len(arr)), 2):  # All pairs
>     # Not optimal (O(n²)), but shows Python knowledge
> ```
> **When to mention:** After demonstrating optimal solution
> **Why:** Shows awareness of language features, but algorithmic thinking comes first
> **Never:** Lead with library solution - interviewer wants to see problem-solving
```

**Callout frequency:** 3-5 per problem set (data structures, patterns, complexity analysis)

#### Decision Logic Templates for Interviews

**Template 1: Algorithm Selection Framework**
```markdown
**Decision Logic Template: Choosing Algorithm**

\```python
# Problem signals → Algorithm decision tree

def select_algorithm(problem_signals):
    """
    Given problem characteristics, choose optimal approach
    """
    
    # DECISION TREE
    if "sorted array" in signals and "search" in signals:
        return "Binary Search - O(log n)"
    
    elif "substring" in signals or "subarray" in signals:
        if "consecutive" in signals:
            return "Sliding Window - O(n)"
        else:
            return "Two Pointers - O(n)"
    
    elif "tree" in signals or "graph" in signals:
        if "shortest path" in signals:
            return "BFS (unweighted) or Dijkstra (weighted) - O(V+E)"
        elif "all paths" in signals:
            return "DFS + Backtracking - O(2^n)"
        else:
            return "DFS or BFS - O(V+E)"
    
    elif "optimization" in signals and "overlapping subproblems" in signals:
        return "Dynamic Programming - O(n²) typical"
    
    elif "top k" in signals or "kth largest" in signals:
        return "Min/Max Heap - O(n log k)"
    
    elif "permutations" in signals or "combinations" in signals:
        return "Backtracking - O(n!)"
    
    else:
        return "START WITH BRUTE FORCE, then optimize"

# Example usage in interview:
# Problem: "Find kth largest element in array"
# → Signals: "kth largest", "array"
# → Algorithm: Min heap of size k - O(n log k)
\```

**Key signals cheat sheet:**
- Sorted → Binary search
- Substring/subarray + consecutive → Sliding window
- Tree/graph + shortest → BFS
- Optimization + subproblems → DP
- Top k / kth → Heap
- All combos → Backtracking
```

**Template 2: Complexity Analysis Decision**
```markdown
**Decision Logic Template: Time-Space Tradeoff**

\```python
def analyze_tradeoff(n, time_complexity, space_complexity):
    """
    Decide if time-space tradeoff is acceptable
    
    n: Input size
    time_complexity: Current time (e.g., "O(n²)")
    space_complexity: Required extra space (e.g., "O(n)")
    """
    
    # Typical interview constraints
    SMALL_N = 100
    MEDIUM_N = 10_000
    LARGE_N = 1_000_000
    
    # DECISION LOGIC
    if n <= SMALL_N:
        verdict = "✅ ACCEPTABLE - Even O(n³) runs instantly"
        action = "Simple brute force is fine"
    
    elif n <= MEDIUM_N:
        if time_complexity in ["O(n)", "O(n log n)", "O(n²)"]:
            verdict = "✅ ACCEPTABLE"
            action = "This complexity is fine for n ≤ 10k"
        else:  # O(n³) or worse
            verdict = "⚠️ TOO SLOW"
            action = "Must optimize to O(n²) or better"
    
    elif n <= LARGE_N:
        if time_complexity in ["O(n)", "O(n log n)"]:
            verdict = "✅ ACCEPTABLE"
            action = "Linear or linearithmic required for n ≤ 1M"
        else:
            verdict = "❌ TOO SLOW"
            action = "Must achieve O(n) or O(n log n)"
    
    else:  # n > 1M
        verdict = "❌ RETHINK PROBLEM"
        action = "Consider streaming, approximation, or distributed approach"
    
    # Space tradeoff
    if space_complexity == "O(n)" and time_improvement > 1:
        space_verdict = "✅ WORTH IT - Trading O(n) space for time is standard"
    elif space_complexity == "O(n²)":
        space_verdict = "⚠️ EXPENSIVE - Justify carefully (memoization table?)"
    
    print(f"n = {n:,}, Time: {time_complexity}, Space: {space_complexity}")
    print(f"Verdict: {verdict}")
    print(f"Action: {action}")
    print(f"Space: {space_verdict}")

# Interview discussion:
# Interviewer: "Can you optimize this O(n²) solution?"
# You: "Yes - use hash table for O(1) lookup. Trade O(n) space for O(n) time."
# Interviewer: "Is that acceptable?"
# You: [Run this analysis] "For n ≤ 10k, O(n) space is negligible (40KB). Acceptable."
\```

**Rules of thumb:**
- n ≤ 100 → Any algorithm works
- n ≤ 10k → O(n²) acceptable, O(n³) too slow
- n ≤ 1M → Need O(n) or O(n log n)
- n > 1M → Streaming/distributed required
```

**Template 3: System Design Decision Workflow**
```markdown
**Decision Logic Template: System Design Choices**

\```python
def system_design_decisions(requirements):
    """
    Design Instagram-like photo sharing service
    
    Requirements:
    - 500M users, 200M daily active
    - 100M photos uploaded per day
    - Read-heavy (100:1 read:write ratio)
    - Low latency (<200ms)
    """
    
    # PHASE 1: Database selection
    read_write_ratio = 100  # Read-heavy
    
    if read_write_ratio > 10:
        db_decision = "SQL (PostgreSQL) + Redis cache"
        reasoning = "Read-heavy → Aggressive caching. SQL for relationships (users, followers)."
    else:
        db_decision = "NoSQL (Cassandra) for write scaling"
        reasoning = "Write-heavy → Eventual consistency, horizontal scaling"
    
    # PHASE 2: Storage strategy
    avg_photo_size_mb = 2
    daily_photos = 100_000_000
    daily_storage_tb = (daily_photos * avg_photo_size_mb) / (1024 * 1024)
    
    if daily_storage_tb > 100:
        storage = "S3 / Blob storage with CDN (CloudFront)"
        reasoning = f"{daily_storage_tb:.1f} TB/day → Object storage + CDN mandatory"
    else:
        storage = "File system + CDN"
    
    # PHASE 3: Caching strategy
    cache_hit_rate_target = 0.8  # 80% cache hits
    
    cache_strategy = {
        "L1": "Browser cache (10 min TTL)",
        "L2": "CDN edge cache (1 hour TTL)",
        "L3": "Redis (metadata, user feeds)",
        "Eviction": "LRU (Least Recently Used)"
    }
    
    # DECISION CHECKPOINT
    print("=" * 60)
    print("SYSTEM DESIGN DECISIONS")
    print("=" * 60)
    print(f"Database: {db_decision}")
    print(f"  Reason: {reasoning}")
    print(f"\nStorage: {storage}")
    print(f"  Daily: {daily_storage_tb:.1f} TB, Yearly: {daily_storage_tb * 365:.0f} TB")
    print(f"\nCaching: {cache_hit_rate_target*100}% target hit rate")
    for level, strategy in cache_strategy.items():
        print(f"  {level}: {strategy}")
    
    return {
        "database": db_decision,
        "storage": storage,
        "cache": cache_strategy
    }

# Interview discussion points:
# Q: "Why PostgreSQL instead of MongoDB?"
# A: "Need ACID for financial transactions (ads revenue). Followers graph benefits from SQL joins."
# Q: "How do you handle 100M photos/day?"
# A: "200 TB/day → S3 with lifecycle policies (archive to Glacier after 1 year). CDN required."
\```

**System design decision checklist:**
- [ ] Database: SQL vs NoSQL (read:write ratio, consistency needs)
- [ ] Storage: File system vs object storage (size, CDN requirements)
- [ ] Caching: Multi-level strategy (browser, CDN, app server)
- [ ] Load balancing: Round-robin vs least-connections
- [ ] Scaling: Horizontal (stateless) vs vertical (stateful)
- [ ] Monitoring: Metrics (latency, error rate, throughput)
```

### Decision Checkpoint Pattern (Interview-Specific)

**Example: After Solving Coding Problem**

```markdown
### DECISION CHECKPOINT — Problem Solved, Optimization Check

**What you just implemented:**
- Two-pointer solution for "Container With Most Water" (LeetCode 11)
- Time: O(n), Space: O(1)
- Code passed all test cases (15/15)

**What it demonstrates:**
- Greedy approach: Move pointer with shorter height (proof of optimality)
- Avoided brute force O(n²) by eliminating impossible candidates early
- Clean code: Clear variable names, edge case handling

**Interview discussion points:**
→ **Complexity proof:** "Why is O(n) optimal?"
  - Answer: "Must examine each height at least once (lower bound Ω(n)). Our algorithm is optimal."

→ **Alternative approaches:** "Could we use dynamic programming?"
  - Answer: "No - no overlapping subproblems. Greedy is sufficient and simpler."

→ **Edge cases handled:**
  - Empty array → Return 0
  - Single element → Return 0
  - All same height → Correct (widest container)

→ **Follow-up preparation:**
  - "What if heights can be negative?" → Adjust comparison logic
  - "What if we want top-k containers?" → Use max heap of size k
  - "Continuous heights (not discrete)?" → Same algorithm, just float arithmetic

**Interviewer feedback loop:**
✅ Optimal solution achieved
✅ Complexity analysis correct
✅ Code is clean and readable
✅ Edge cases considered
→ READY to move to next problem or discuss trade-offs
```

### Files Modified Summary

**Primary authoring guide:** `notes/interview_guides/authoring-guide.md`
- Add "Workflow-Based Chapter Pattern" (nearly ALL chapters are workflow-based)
- Add 7 high-priority workflow chapters (problem recognition, UMPIRE, data structures, algorithms, system design, complexity, behavioral)
- Add Code Snippet Guidelines (4 rules: templates with decisions, pattern recognition, complexity trade-offs, LeetCode walkthroughs)
- Add Industry Tools Decision Framework (when to show from-scratch vs built-ins)
- Add Notebook Exercise Pattern (3 decision templates: algorithm selection, time-space tradeoff, system design)
- Add Decision Checkpoint examples (post-problem optimization check)
- Add Python built-ins cheat sheet (`collections`, `heapq`, `bisect`)
- Add System design tools reference (PostgreSQL, Redis, Kafka, Docker, etc.)

**Chapters requiring workflow restructure:** (Majority of track)
- Ch.1 Problem Type Recognition
- Ch.2 UMPIRE Framework
- Ch.3 Data Structure Selection
- Ch.4 Algorithm Patterns
- Ch.5 System Design (RESHADED)
- Ch.6 Complexity Analysis
- Ch.7 Behavioral (STAR)
- Ch.8 Debugging Strategy
- Ch.9 ML System Design

**Estimated LOC:** +550 lines to authoring guide (largest update - most workflow-heavy track)

---

## Summary: All 4 Tracks Updated

| Track | Workflow Chapters | Effort | LOC Added | Key Additions |
|-------|------------------|--------|-----------|---------------|
| **Multimodal AI** | 4-5 (audio, image, video, fine-tuning) | 3-4 hrs | +450 | Preprocessing pipelines, modality-specific decisions |
| **Advanced DL** | 4-5 (HPO, NAS, distributed, diagnostics) | 3-4 hrs | +500 | Training workflows, convergence checks, optimization |
| **Math** | 2-3 (optimization, factorization) | 2-3 hrs | +350 | Manual→library pattern, numerical stability, mostly concept-based |
| **Interview Guides** | 7+ (problem types, UMPIRE, DS, algo, system design, complexity, behavioral) | 2-3 hrs | +550 | Pattern templates, decision trees, LeetCode walkthroughs, system design |

**Total Effort:** 10-14 hours  
**Total LOC:** +1,850 lines across 4 authoring guides  
**LLM Calls:** 8-10 (2-3 per track for drafting + review)

**Common Patterns Applied Across All Tracks:**
1. ✅ Workflow-based chapter identification (adapted to each track's content)
2. ✅ Code snippet guidelines (4 rules: phase implementation, decision logic, manual→library, real datasets)
3. ✅ Industry tools decision framework (when to show from-scratch vs production tools)
4. ✅ Notebook exercise pattern (decision templates, industry callouts, visual indicators)
5. ✅ Decision checkpoint format (What you saw → What it means → What to do next)

**Track-Specific Adaptations:**
- **Multimodal AI:** Preprocessing pipelines, modality-specific thresholds, fine-tuning strategies
- **Advanced DL:** Training diagnostics, distributed strategies, architecture search, HPO
- **Math:** Limited workflows (mostly concept-based), numerical stability, manual derivation emphasis
- **Interview Guides:** Most workflow-heavy, pattern recognition, complexity analysis, system design frameworks
