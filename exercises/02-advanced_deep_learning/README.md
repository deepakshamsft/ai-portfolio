# Exercise 02: ProductionCV — Advanced Deep Learning Edge Deployment

> **Grand Challenge:** Compress ResNet-50 (97 MB) into a <100 MB edge-deployable model achieving ≥85% mAP, <50ms inference on NVIDIA Jetson Nano, trained on <1,000 labeled images.

**Scaffolding Level:** 🔴 Minimal (demonstrate independence)

---

## Objective

Build ProductionCV — a complete computer vision pipeline with:
- Object detection (Faster R-CNN, YOLO)
- Instance segmentation (Mask R-CNN)
- Model compression (knowledge distillation + pruning + quantization)
- Edge deployment optimization (ONNX, TensorRT, TFLite)
- Production API with monitoring

---

## Concepts Covered

Ch.1-10 from [notes/02-advanced_deep_learning/](../../notes/02-advanced_deep_learning/):
- ResNets, MobileNets, EfficientNets
- Two-stage & one-stage detectors
- Semantic & instance segmentation
- Knowledge distillation, pruning, mixed precision
- Edge deployment optimization

---

## Grand Challenge

**Success Criteria:**
- ✅ Model size: **<100 MB**
- ✅ mAP (mean Average Precision): **≥85%**
- ✅ Inference latency: **<50ms** per image
- ✅ Edge deployment: Validated on Jetson Nano or equivalent
- ✅ Production API: Flask REST API with monitoring

---

## Setup

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

**Verify Installation:**
```bash
make test
```

---

## Project Structure

```
exercises/02-advanced_deep_learning/
├── src/
│   ├── __init__.py          # Package exports
│   ├── utils.py             # Logging, timing, ONNX helpers
│   ├── data.py              # COCO dataset loading
│   ├── features.py          # Image preprocessing
│   ├── models.py            # Model registry + compression
│   ├── evaluate.py          # mAP calculation, benchmarking
│   ├── edge.py              # Edge deployment utilities
│   ├── monitoring.py        # Prometheus metrics
│   └── api.py               # Flask REST API
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_data.py         # Data loading tests
│   ├── test_models.py       # Model + compression tests
│   ├── test_edge.py         # Edge deployment tests
│   └── test_api.py          # API endpoint tests
├── models/                  # Trained models
├── data/                    # COCO dataset
├── logs/                    # Training logs
├── notebooks/               # Exploratory notebooks
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── Makefile                 # Development commands
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-container orchestration
└── prometheus.yml           # Metrics configuration
```

---

## Quick Start

### 1. Train Baseline Model

```bash
make train
```

This trains a Faster R-CNN with ResNet-50 backbone on COCO subset.

### 2. Apply Compression Pipeline

```bash
make compress
```

Applies the compression pipeline:
1. **Knowledge Distillation**: Student model learns from teacher (ResNet-101)
2. **Pruning**: Remove 30% of least important weights
3. **Quantization**: Convert to INT8 precision

### 3. Evaluate Compressed Model

```bash
make evaluate
```

Outputs:
- mAP @ IoU=0.5
- Inference latency (mean, median, std)
- Model size (MB)
- All success criteria met? ✓/✗

### 4. Deploy to Edge

```bash
make deploy-edge
```

Exports model to:
- **ONNX**: Optimized for inference
- **TFLite**: For mobile/edge devices

### 5. Start Production API

```bash
make serve
```

API endpoints:
- `GET /health` - Health check
- `POST /detect` - Object detection
- `GET /info` - Model information
- `GET /metrics` - Prometheus metrics

---

## Compression Pipeline Details

### Knowledge Distillation

**Goal**: Transfer knowledge from large teacher to small student model.

```yaml
# config.yaml
compression:
  distillation:
    enabled: true
    temperature: 3.0        # Softening factor
    alpha: 0.5              # Balance hard/soft targets
    teacher_model: resnet101
```

**How it works**:
- Teacher model (ResNet-101) produces soft targets
- Student model (ResNet-50) learns from both hard labels and soft targets
- Temperature parameter controls softness of probability distributions

### Pruning

**Goal**: Remove less important weights to reduce model size.

```yaml
compression:
  pruning:
    enabled: true
    pruning_ratio: 0.3      # Remove 30% of weights
    pruning_schedule: polynomial
```

**Strategy**: L1-norm based structured pruning on Conv2D layers.

### Quantization

**Goal**: Reduce precision from FP32 to INT8 to speed up inference.

```yaml
compression:
  quantization:
    enabled: true
    quantization_type: int8
    per_channel: true
```

**Benefits**:
- 4x model size reduction
- 2-4x inference speedup
- Minimal accuracy loss (<2%)

---

## Edge Deployment

### ONNX Export

```python
from src.edge import EdgeDeployer
from src.models import ModelRegistry

registry = ModelRegistry(config)
model = registry.load_model('models/compressed_model.pth')

deployer = EdgeDeployer(config)
onnx_path = deployer.export_to_onnx(model, 'models/onnx/model.onnx')
```

### Validation for Jetson Nano

```bash
python -m src.edge --model-path models/compressed_model.pth --export
```

Validates:
- ✅ Model size <100 MB
- ✅ ONNX compatibility
- ✅ Inference latency <50ms

---

## API Usage

### Start API Server

```bash
# Development
make serve

# Production (Docker)
make docker-compose-up
```

### Detect Objects in Image

```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "max_detections=100"
```

**Response:**
```json
{
  "success": true,
  "num_detections": 3,
  "detections": [
    {
      "x": 50, "y": 50, "width": 100, "height": 150,
      "confidence": 0.95,
      "label": "person",
      "class_id": 0
    },
    ...
  ],
  "inference_time_ms": 32.5,
  "image_shape": [640, 640, 3]
}
```

---

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`:

- `productioncv_predictions_total` - Total predictions
- `productioncv_prediction_latency_seconds` - Latency histogram
- `productioncv_detections_per_image` - Detections distribution
- `productioncv_model_size_mb` - Model size
- `productioncv_inference_fps` - Throughput

### Grafana Dashboard

Access at `http://localhost:3000` (admin/admin):

1. Add Prometheus data source: `http://prometheus:9090`
2. Import dashboard: `dashboards/productioncv.json`
3. Monitor real-time metrics

---

## Testing

```bash
# Run all tests
make test

# Run specific test suite
pytest tests/test_models.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- ✅ Data loading and augmentation
- ✅ Model creation and compression
- ✅ ONNX export and validation
- ✅ API endpoints
- ✅ Edge deployment validation

---

## Docker Deployment

### Build and Run

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose
docker-compose up -d
```

### Multi-Container Stack

```bash
docker-compose up -d
```

Includes:
- **API**: ProductionCV service (port 5000)
- **Prometheus**: Metrics collection (port 9091)
- **Grafana**: Visualization (port 3000)

---

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  base_model: "resnet50"
  architecture: "faster_rcnn"  # or yolov8, mask_rcnn
  target_size_mb: 100
  target_map: 0.85
  target_inference_ms: 50

compression:
  distillation:
    enabled: true
    temperature: 3.0
  pruning:
    enabled: true
    pruning_ratio: 0.3
  quantization:
    enabled: true
    quantization_type: "int8"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

---

## Common Issues

### CUDA Out of Memory

**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### ONNX Export Fails

**Solution**: Install ONNX dependencies:
```bash
pip install onnx onnxruntime onnx-tf
```

### Slow Inference

**Solutions**:
1. Enable quantization in `config.yaml`
2. Use GPU inference (CUDA)
3. Optimize ONNX model with `optimize_onnx()`

---

## Success Checklist

Before considering this exercise complete:

- [ ] All tests pass (`make test`)
- [ ] Compressed model <100 MB
- [ ] mAP ≥85% on test set
- [ ] Inference <50ms per image
- [ ] ONNX export validated
- [ ] API responds to `/detect` requests
- [ ] Prometheus metrics exposed
- [ ] Docker deployment works
- [ ] Edge device validation passed
- [ ] Code passes linting (`make lint`)

---

## Resources

- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [ONNX Documentation](https://onnx.ai/)
- [TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson Nano Setup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

---

## Next Steps

After completing this exercise:

1. **Experiment** with different architectures (YOLOv8, EfficientDet)
2. **Optimize** further with TensorRT for Jetson deployment
3. **Extend** to video object tracking
4. **Add** instance segmentation capabilities
5. **Deploy** to production with Kubernetes

---

## License

This exercise is part of the AI Portfolio project.

**Concept Review:**
- [notes/02-advanced_deep_learning/](../../notes/02-advanced_deep_learning/)
- [notes/02-advanced_deep_learning/grand_solution.md](../../notes/02-advanced_deep_learning/grand_solution.md)

---

**Status:** Phase 3 - Coming soon  
**Last Updated:** April 28, 2026
