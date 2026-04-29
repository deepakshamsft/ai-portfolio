# Multimodal AI / Advanced DL / Math / Interview Guides — Authoring Guide Update Plans

## Multimodal AI Track

**Target:** `notes/05-multimodal_ai/authoring-guide.md`  
**Effort:** 1 hour  
**LLM Calls:** 1

### Quick Context
Add industry tools decision framework. Track already shows diffusers/transformers but lacks explicit when-to-use guidance.

### Single Addition
**Location:** After "Educational vs Production Pattern" section

**Content:**
```markdown
## Industry Tools Decision Framework

### When to Show From-Scratch
- Core concepts (attention mechanism, diffusion forward/reverse)
- Mathematical foundations (VAE loss derivation)
- Educational notebooks (build intuition)

### When to Show Library
- Production pipelines (always use `diffusers`)
- Inference optimization (xFormers, fp16, LCM)
- Pre-trained models (CLIP, Stable Diffusion)

### Required Library Coverage
- `diffusers`: StableDiffusionPipeline, DDIMScheduler, LCMScheduler
- `transformers`: CLIPModel, ViTModel, AutoProcessor
- `torch`: Mixed precision, xFormers integration
- `opencv-python`: Image preprocessing (Canny, resize)

### Pattern
```python
# Educational (Ch.4): DDPM from scratch
def denoise_step(x_t, t, model):
    # Full implementation

# Production (Ch.13): Use diffusers
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-2-1")
image = pipe(prompt).images[0]  # One line!
```
```

**Files Modified:** 1

---

## Advanced Deep Learning Track

**Target:** `notes/02-advanced_deep_learning/authoring-guide.md`  
**Effort:** 1 hour  
**LLM Calls:** 1

### Quick Context
Track is 100% concept-driven (architectures only). Add industry tools framework for TensorFlow/PyTorch/Keras usage.

### Single Addition
**Location:** After chapter template section

**Content:**
```markdown
## Industry Tools Integration (All Chapters)

### Required Pattern Per Chapter
1. Show architecture concept (diagrams, math)
2. Implement from scratch (PyTorch modules, custom layers)
3. Show library equivalent (torchvision, timm, transformers)

### Industry Libraries by Chapter
- Ch.1-4 (Architectures): `torchvision.models`, `timm`
- Ch.5 (GANs): `torch.nn` custom, then `torchgan`
- Ch.6 (VAEs): Custom implementation, then `pytorch-lightning`
- Ch.7-10: `transformers.AutoModel`, `torch.optim`

### Code Pattern
```python
# From scratch (learning)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        # Full implementation

# Industry standard (production)
from torchvision.models import resnet50
model = resnet50(pretrained=True)  # That's it!
```
```

**Files Modified:** 1

---

## Math Under the Hood Track

**Target:** `notes/00-math_under_the_hood/authoring-guide.md`  
**Effort:** 1 hour  
**LLM Calls:** 1

### Quick Context
Add NumPy/SymPy usage pattern. Currently mentions dual approach but lacks framework.

### Single Addition
**Location:** After "Code Style" section

**Content:**
```markdown
## Manual vs Library Implementation Pattern

### When to Show Manual
- First introduction of concept (matrix multiply, gradient)
- Derivations requiring step-by-step arithmetic
- Proof-of-concept for mathematical theorems

### When to Show NumPy/SymPy
- After manual implementation shown once
- Production-scale computations (large matrices)
- Symbolic math (SymPy for derivatives)

### Required Pattern
```python
# Manual (learning): Show the math
result = 0
for i in range(n):
    for j in range(m):
        result += A[i,j] * B[j,i]

# NumPy (production): One line
result = np.dot(A, B.T)  # Same result, 100× faster
```

### Libraries by Chapter
- Linear Algebra: `numpy.linalg`
- Calculus: `scipy.optimize`, `sympy`
- Matrices: `numpy`, `scipy.sparse`
- Statistics: `scipy.stats`, `numpy.random`
```

**Files Modified:** 1

---

## Interview Guides Track

**Target:** `notes/interview_guides/authoring-guide.md`  
**Effort:** 30 min  
**LLM Calls:** 1

### Quick Context
Track is Q&A format only. Add brief note about industry tools in answers.

### Single Addition
**Location:** After "Answer Format" section

**Content:**
```markdown
## Industry Tools in Answers

### Pattern for Technical Questions
1. Explain concept (theory)
2. Show manual implementation (if relevant)
3. Mention industry standard tool

### Example
**Q: How do you detect multicollinearity?**

**Junior:** "Use correlation matrix..."

**Senior:** "Use VIF (Variance Inflation Factor). Manual calculation: regress each feature on others, compute R². Industry standard: `statsmodels.variance_inflation_factor()`. In production, I run VIF before every training job, alert if >5."

### Required Tool Mentions
- Feature engineering: sklearn preprocessing
- Model training: sklearn/PyTorch/TensorFlow
- Evaluation: sklearn.metrics
- Deployment: FastAPI, Docker, Kubernetes
```

**Files Modified:** 1

---

## Summary

| Track | Effort | LLM Calls | Key Addition |
|-------|--------|-----------|--------------|
| Multimodal AI | 1 hr | 1 | Industry tools decision framework |
| Advanced DL | 1 hr | 1 | Library equivalents per architecture |
| Math | 1 hr | 1 | NumPy/SymPy usage pattern |
| Interview Guides | 30 min | 1 | Tool mentions in answers |

**Total:** 3.5 hours, 4 LLM calls  
**All files:** Single-edit operations (insert after existing section)
