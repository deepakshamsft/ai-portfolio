# Exercise 05: PixelSmith — Multimodal AI System

> **Grand Challenge:** Build a production-grade multimodal AI system processing text, images, and audio with <30s generation time, CLIP score >0.7, and WER <15%.

**Scaffolding Level:** 🔴 Minimal (demonstrate independence)

---

## Objective

Implement a complete multimodal AI system with production patterns:
- **CLIP** for text-image similarity and zero-shot classification
- **Whisper** for automatic speech recognition
- **Stable Diffusion** for text-to-image generation
- **BLIP** for image captioning
- Multimodal feature extraction and fusion
- Flask API with multiple endpoints
- Prometheus monitoring and MLflow tracking
- Docker deployment with GPU support

---

## System Architecture

### Supported Modalities

1. **Text Processing**
   - CLIP text embeddings
   - Text-image similarity scoring
   - Zero-shot classification

2. **Image Processing**
   - CLIP image embeddings
   - Image captioning (BLIP)
   - Text-to-image generation (Stable Diffusion)
   - Image search

3. **Audio Processing**
   - Speech-to-text (Whisper)
   - Audio feature extraction
   - Language detection

### Model Details

| Model | Task | Size | Device |
|-------|------|------|--------|
| CLIP ViT-B/32 | Text-Image Similarity | ~350MB | CPU/GPU |
| Whisper Base | Speech Recognition | ~140MB | CPU/GPU |
| Stable Diffusion v1.5 | Text-to-Image | ~4GB | GPU |
| BLIP Base | Image Captioning | ~990MB | CPU/GPU |

---

## Project Structure

```
exercises/05-multimodal_ai/
├── src/
│   ├── __init__.py              # Package exports
│   ├── utils.py                 # Logging, config, file handling
│   ├── data.py                  # Multimodal data loaders
│   ├── features.py              # Feature extraction & fusion
│   ├── evaluate.py              # Evaluation metrics
│   ├── monitoring.py            # Prometheus & MLflow
│   ├── api.py                   # Flask REST API
│   └── models/
│       ├── __init__.py
│       ├── clip.py              # CLIP model wrapper
│       ├── whisper.py           # Whisper model wrapper
│       ├── image_gen.py         # Stable Diffusion wrapper
│       └── image_caption.py     # BLIP model wrapper
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_clip.py             # CLIP tests
│   ├── test_whisper.py          # Whisper tests
│   ├── test_image_gen.py        # Generation tests
│   └── test_api.py              # API endpoint tests
├── config.yaml                  # System configuration
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Service orchestration
├── prometheus.yml               # Monitoring config
├── Makefile                     # Build automation
└── README.md                    # This file
```

---

## Setup

### Local Development

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

### Docker Deployment

**CPU-only:**
```bash
docker-compose up pixelsmith-api
```

**GPU-enabled (requires NVIDIA Docker):**
```bash
docker-compose up pixelsmith-gpu
```

**With monitoring (Prometheus + Grafana):**
```bash
docker-compose up
```

---

## API Endpoints

### 1. Text-Image Similarity
Compute CLIP similarity between text and image.

**Endpoint:** `POST /similarity`

**Request:**
```bash
curl -X POST http://localhost:5000/similarity \
  -F "text=a red car on a highway" \
  -F "image=@car.jpg"
```

**Response:**
```json
{
  "similarity": 0.82,
  "text": "a red car on a highway",
  "threshold": 0.7
}
```

---

### 2. Speech-to-Text
Transcribe audio to text using Whisper.

**Endpoint:** `POST /transcribe`

**Request:**
```bash
curl -X POST http://localhost:5000/transcribe \
  -F "audio=@speech.wav"
```

**Response:**
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "duration": 3.5
}
```

---

### 3. Text-to-Image Generation
Generate images from text prompts using Stable Diffusion.

**Endpoint:** `POST /generate`

**Request:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42
  }' \
  --output generated.png
```

**Response:** PNG image file

---

### 4. Image Captioning
Generate captions for images using BLIP.

**Endpoint:** `POST /caption`

**Request:**
```bash
curl -X POST http://localhost:5000/caption \
  -F "image=@landscape.jpg"
```

**Response:**
```json
{
  "caption": "a beautiful mountain landscape with trees and a lake"
}
```

---

### 5. Multimodal Search
Search using text, image, or audio queries.

**Endpoint:** `POST /search`

**Request:**
```bash
curl -X POST http://localhost:5000/search \
  -F "text=mountain landscape" \
  -F "image=@query.jpg"
```

**Response:**
```json
{
  "results": [
    {"id": 1, "score": 0.85, "type": "image"},
    {"id": 2, "score": 0.78, "type": "text"}
  ],
  "query_modalities": ["text", "image"]
}
```

---

## Performance Metrics

### Success Criteria

| Metric | Target | Measured |
|--------|--------|----------|
| CLIP Score | >0.7 | ✓ |
| Word Error Rate | <15% | ✓ |
| Caption BLEU | >0.3 | ✓ |
| Generation Time | <30s | ✓ |
| API Latency (p99) | <500ms | ✓ |

### Monitoring

**Prometheus Metrics:**
- Request count by endpoint and modality
- Request duration histograms
- Active request gauges
- Model load times
- CLIP score distribution

**MLflow Tracking:**
- Model parameters
- Evaluation metrics
- Generated artifacts

**Access Metrics:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- MLflow UI: http://localhost:5001

---

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  clip:
    model_name: "openai/clip-vit-base-patch32"
    device: "cuda"  # or "cpu"
  
  whisper:
    model_name: "openai/whisper-base"
    device: "cuda"

  stable_diffusion:
    model_name: "runwayml/stable-diffusion-v1-5"
    num_inference_steps: 50
    guidance_scale: 7.5

image:
  max_width: 512
  max_height: 512
  max_file_size_mb: 10

audio:
  sample_rate: 16000
  max_duration_sec: 30

thresholds:
  clip_score_min: 0.7
  wer_max: 0.15
```

---

## Testing

**Run all tests:**
```bash
make test
```

**Run fast tests only:**
```bash
make test-fast
```

**Test coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Development Commands

```bash
# Install dependencies
make install

# Format code
make format

# Run linters
make lint

# Run API locally
make run

# Clean generated files
make clean

# Docker build
make docker-build

# Start services
make docker-up

# View logs
make docker-logs
```

---

## Concepts Covered

From [notes/05-multimodal_ai/](../../notes/05-multimodal_ai/):
- Vision Transformers and CLIP embeddings
- Diffusion models and Stable Diffusion
- Speech recognition with Whisper
- Image captioning with BLIP
- Multimodal feature fusion
- Production API design
- Model monitoring and observability

---

## Troubleshooting

### Out of Memory (GPU)
```python
# In config.yaml, reduce batch size or enable CPU offload
models:
  stable_diffusion:
    device: "cpu"  # Use CPU instead
```

### Slow Generation
```bash
# Reduce inference steps
{
  "num_inference_steps": 20  # Default: 50
}
```

### Model Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
```

---

## Resources

**Concept Review:**
- [notes/05-multimodal_ai/](../../notes/05-multimodal_ai/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)

**Model Documentation:**
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)

---

**Status:** ✅ Complete - Production-ready multimodal AI system  
**Last Updated:** April 28, 2026
