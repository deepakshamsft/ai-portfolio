# Advanced Deep Learning — Track Plan

> **Status**: 🟡 PLANNING (not yet started)
> **Created**: 2026-04-26
> **Target Launch**: TBD

---

## Mission Statement

**Build ProductionCV** — a computer vision system that goes beyond image classification to solve real-world tasks: object detection, semantic segmentation, and instance segmentation. Master advanced CNN architectures (ResNet, EfficientNet), self-supervised pretraining strategies, and production optimization techniques (knowledge distillation, pruning, mixed precision training).

**Gap Being Filled**: The Neural Networks track (03_neural_networks) teaches CNN fundamentals (convolution, pooling, Ch.5) and ends with Transformers (Ch.10). This track bridges the gap between "basic CNNs" and "production computer vision systems" by covering:
- Advanced architectures that power modern CV (ResNet family)
- Tasks beyond classification (detection, segmentation)
- Self-supervised learning (leverage unlabeled data at scale)
- Model compression for edge deployment (distillation, pruning)

**Prerequisites**: 
- ✅ Complete [03_neural_networks](../03_neural_networks/README.md) (especially Ch.5 CNNs, Ch.10 Transformers)
- ✅ Comfortable with PyTorch/Keras
- ✅ Understand backpropagation, regularization, convolutional layers

**What's NOT in this track** (covered elsewhere):
- Vision Transformers (ViT) → [Multimodal AI](../../04-multimodal_ai/ch02_vision_transformers/README.md)
- Generative models (GANs, VAEs, Diffusion) → [Multimodal AI](../../04-multimodal_ai/README.md)
- Fine-tuning & transfer learning (LLMs) → [AI/Fine-Tuning](../../02-ai/ch10_fine_tuning/fine-tuning.md)
- Quantization (FP16, INT8) → [AI Infrastructure/Quantization](../../05-ai_infrastructure/ch03_quantization_and_precision/README.md)
- Distributed training (DDP, FSDP) → [AI Infrastructure/Parallelism](../../05-ai_infrastructure/ch04_parallelism_and_distributed_training/README.md)

---

## The Grand Challenge

**ProductionCV System** — Autonomous Retail Shelf Monitoring

| Constraint | Target | Why It Matters |
|------------|--------|----------------|
| **#1 Detection Accuracy** | mAP@0.5 ≥ 85% | Detect products on retail shelves (empty slots, misplaced items) |
| **#2 Segmentation Quality** | IoU ≥ 70% | Pixel-level product boundary detection for planogram compliance |
| **#3 Inference Latency** | <50ms per frame | Real-time monitoring on in-store edge devices (NVIDIA Jetson) |
| **#4 Model Size** | <100 MB | Deploy on edge hardware (memory-constrained devices) |
| **#5 Data Efficiency** | <1,000 labeled images | Self-supervised pretraining on 50k unlabeled shelf photos |

**Dataset**: Custom synthetic retail shelf dataset (to be created)
- **Labeled**: 1,000 annotated images (bounding boxes + segmentation masks)
- **Unlabeled**: 50,000 shelf photos for self-supervised pretraining
- **Classes**: 20 product types (soda cans, cereal boxes, milk cartons, etc.)

**Why This Challenge?**
- Combines all track skills: detection (#1), segmentation (#2), optimization (#3–4), self-supervised learning (#5)
- Realistic production constraints (edge deployment, limited labels)
- Mirrors real-world CV deployments (retail, manufacturing, surveillance)

---

## Proposed Chapter Structure

### Act 1: Advanced CNN Architectures (Ch.1–2)

**Ch.1 — Residual Networks (ResNets)**
- **Core Idea**: Skip connections solve vanishing gradients, enable 100+ layer networks
- **Architecture**: ResNet-18, ResNet-50, ResNet-101 (basic blocks vs bottleneck blocks)
- **Breakthrough**: He et al. (2015) — first 100+ layer network, ImageNet winner
- **Hands-On**: Build ResNet-18 from scratch, train on CIFAR-10, visualize activation flow
- **Production Insight**: Why ResNet is still the backbone for object detection (Faster R-CNN, Mask R-CNN)
- **Challenge Progress**: Establish baseline architecture for shelf product classification

**Ch.2 — Efficient Architectures (MobileNet, EfficientNet)**
- **Core Idea**: Depthwise separable convolutions, compound scaling (depth + width + resolution)
- **MobileNetV2**: Inverted residual blocks, linear bottlenecks (Sandler et al., 2018)
- **EfficientNet**: Neural Architecture Search (NAS) + compound scaling (Tan & Le, 2019)
- **Hands-On**: Compare ResNet-50 (25M params) vs EfficientNet-B0 (5M params, same accuracy)
- **Production Insight**: When to choose efficiency over raw accuracy (mobile, edge devices)
- **Challenge Progress**: Reduce model size from 100 MB → 20 MB (constraint #4 progress)

---

### Act 2: Object Detection (Ch.3–4)

**Ch.3 — Two-Stage Detectors (R-CNN Family)**
- **Core Idea**: Region proposals → CNN features → classification + bounding box regression
- **Evolution**: R-CNN (2014) → Fast R-CNN (2015) → Faster R-CNN (2015, region proposal network)
- **Architecture**: Region Proposal Network (RPN), RoI pooling, multi-task loss (classification + bbox regression)
- **Hands-On**: Implement Faster R-CNN with ResNet-50 backbone on PASCAL VOC
- **Production Insight**: When accuracy matters more than speed (medical imaging, autonomous driving)
- **Challenge Progress**: First object detection baseline (products on shelves)

**Ch.4 — One-Stage Detectors (YOLO, SSD, RetinaNet)**
- **Core Idea**: Direct prediction (no region proposals) → 10–100× faster than Faster R-CNN
- **YOLO**: Grid-based prediction, single forward pass (Redmon et al., 2016)
- **RetinaNet**: Focal loss solves class imbalance (easy negatives dominate) (Lin et al., 2017)
- **Hands-On**: Train YOLOv5 on custom retail shelf dataset, compare mAP vs Faster R-CNN
- **Production Insight**: Real-time detection on edge devices (surveillance, retail)
- **Challenge Progress**: ✅ Constraint #1 (mAP@0.5 ≥ 85%), ⚡ Constraint #3 progress (<100ms → target <50ms)

---

### Act 3: Semantic & Instance Segmentation (Ch.5–6)

**Ch.5 — Semantic Segmentation (FCN, U-Net, DeepLab)**
- **Core Idea**: Pixel-level classification (assign class to every pixel, no object boundaries)
- **FCN**: Fully Convolutional Networks, replace FC layers with conv layers (Long et al., 2015)
- **U-Net**: Encoder-decoder with skip connections (medical imaging standard) (Ronneberger et al., 2015)
- **DeepLabV3+**: Atrous convolutions (dilated conv), ASPP (Atrous Spatial Pyramid Pooling) (Chen et al., 2018)
- **Hands-On**: Train U-Net on retail shelf images (segment empty space vs products)
- **Production Insight**: Medical imaging, satellite imagery, autonomous driving
- **Challenge Progress**: ⚡ Constraint #2 progress (IoU 60% → target 70%)

**Ch.6 — Instance Segmentation (Mask R-CNN)**
- **Core Idea**: Detect objects AND segment each instance (combination of detection + segmentation)
- **Architecture**: Faster R-CNN + segmentation branch (RoIAlign, mask prediction head)
- **Hands-On**: Train Mask R-CNN on retail shelf dataset (detect + segment individual products)
- **Production Insight**: When you need object boundaries + instance differentiation (robotics, AR)
- **Challenge Progress**: ✅ Constraint #2 (IoU ≥ 70% achieved)

---

### Act 4: Self-Supervised Learning (Ch.7–8)

**Ch.7 — Contrastive Learning (SimCLR, MoCo)**
- **Core Idea**: Learn representations from unlabeled data by contrasting augmented views
- **SimCLR**: Contrastive loss, large batch sizes, data augmentation (Chen et al., 2020)
- **MoCo**: Momentum encoder, queue-based negative sampling (He et al., 2020)
- **Hands-On**: Pretrain ResNet-50 on 50k unlabeled shelf photos (SimCLR), fine-tune on 1k labeled
- **Production Insight**: Leverage massive unlabeled datasets (reduces labeling cost 10×)
- **Challenge Progress**: ✅ Constraint #5 (data efficiency, <1,000 labeled images)

**Ch.8 — Self-Supervised Vision (DINO, MAE)**
- **Core Idea**: Self-distillation (DINO) and masked autoencoding (MAE) for vision
- **DINO**: Self-distillation with no labels (attention maps emerge automatically) (Caron et al., 2021)
- **MAE**: Mask 75% of image patches, reconstruct (Masked Autoencoder, He et al., 2022)
- **Hands-On**: Train DINO on unlabeled shelf images, visualize emergent attention maps
- **Production Insight**: State-of-art pretraining for ViT (bridges to Multimodal AI track)
- **Challenge Progress**: Further improve data efficiency (mAP 82% → 86% with same 1k labels)

---

### Act 5: Production Optimization (Ch.9–10)

**Ch.9 — Knowledge Distillation**
- **Core Idea**: Transfer knowledge from large teacher model → small student model
- **Technique**: Soft labels (teacher's probability distribution), temperature scaling (Hinton et al., 2015)
- **Hands-On**: Distill ResNet-50 teacher → MobileNetV2 student (maintain 90% accuracy, 5× smaller)
- **Production Insight**: Edge deployment, mobile apps (complementary to quantization)
- **Challenge Progress**: ⚡ Constraint #4 progress (20 MB → 10 MB, maintain mAP)

**Ch.10 — Pruning & Mixed Precision Training**
- **Pruning**: Remove redundant weights (structured vs unstructured, magnitude-based vs learned)
- **Mixed Precision**: FP16 training with FP32 master weights (AMP, Automatic Mixed Precision)
- **Hands-On**: Prune distilled student model (10 MB → 5 MB), train with mixed precision (2× speedup)
- **Production Insight**: Final optimization stack (pruning + quantization + distillation)
- **Challenge Progress**: ✅ All constraints met:
  - ✅ #1 mAP@0.5 = 85.4%
  - ✅ #2 IoU = 71.2%
  - ✅ #3 Latency = 48ms (NVIDIA Jetson Nano)
  - ✅ #4 Model size = 8.2 MB
  - ✅ #5 Data efficiency (<1k labels via self-supervised pretraining)

---

## Progressive Capability Table

| Ch | Title | Key Concept | Grand Challenge Progress |
|----|-------|-------------|--------------------------|
| **1** | Residual Networks | Skip connections, 100+ layers | Baseline architecture (ResNet-50, 80% mAP) |
| **2** | Efficient Architectures | MobileNet, EfficientNet, depthwise conv | Model size 100 MB → 20 MB |
| **3** | Two-Stage Detectors | Region proposals, Faster R-CNN | First detection baseline (products on shelves) |
| **4** | One-Stage Detectors | YOLO, RetinaNet, focal loss | ✅ Constraint #1 (mAP 85%), latency 95ms |
| **5** | Semantic Segmentation | FCN, U-Net, atrous convolutions | Pixel-level product boundaries (IoU 62%) |
| **6** | Instance Segmentation | Mask R-CNN, RoIAlign | ✅ Constraint #2 (IoU 71%) |
| **7** | Contrastive Learning | SimCLR, MoCo, self-supervised | ✅ Constraint #5 (data efficiency, 1k labels) |
| **8** | Self-Supervised Vision | DINO, MAE, masked autoencoding | mAP 82% → 86% (same 1k labels) |
| **9** | Knowledge Distillation | Teacher-student, soft labels | Model size 20 MB → 10 MB (maintain mAP) |
| **10** | Pruning & Mixed Precision | Structured pruning, AMP | ✅ All constraints met (8.2 MB, 48ms, 85% mAP) |

---

## Timeline & Milestones

**Phase 1: Planning & Dataset Prep** (Current)
- [x] Identify gaps in 03_neural_networks coverage
- [x] Define grand challenge (ProductionCV retail shelf monitoring)
- [ ] Create synthetic retail shelf dataset (1k labeled + 50k unlabeled)
- [ ] Establish evaluation metrics (mAP, IoU, latency, model size)

**Phase 2: Act 1–2 (Architectures)** — ETA TBD
- [ ] Ch.1: ResNets (skip connections, residual blocks)
- [ ] Ch.2: Efficient architectures (MobileNet, EfficientNet)
- [ ] Milestone: Classification baseline (80% mAP on shelf products)

**Phase 3: Act 3 (Detection & Segmentation)** — ETA TBD
- [ ] Ch.3: Two-stage detectors (Faster R-CNN)
- [ ] Ch.4: One-stage detectors (YOLO, RetinaNet)
- [ ] Ch.5: Semantic segmentation (FCN, U-Net, DeepLab)
- [ ] Ch.6: Instance segmentation (Mask R-CNN)
- [ ] Milestone: ✅ Constraints #1–2 (detection + segmentation)

**Phase 4: Act 4 (Self-Supervised)** — ETA TBD
- [ ] Ch.7: Contrastive learning (SimCLR, MoCo)
- [ ] Ch.8: Self-supervised vision (DINO, MAE)
- [ ] Milestone: ✅ Constraint #5 (data efficiency <1k labels)

**Phase 5: Act 5 (Optimization)** — ETA TBD
- [ ] Ch.9: Knowledge distillation
- [ ] Ch.10: Pruning & mixed precision
- [ ] Milestone: ✅ All constraints met (ProductionCV deployed)

**Phase 6: Integration & Testing** — ETA TBD
- [ ] End-to-end ProductionCV system (detection + segmentation + edge deployment)
- [ ] Benchmark against commercial solutions (AWS Rekognition, Azure Custom Vision)
- [ ] GRAND_CHALLENGE.md validation
- [ ] Track README finalization

---

## Unification Narrative

This track demonstrates a key production truth: **Modern computer vision is a layered stack, not a single model.**

```
Layer 5: Production Optimization
├─ Distillation (Ch.9) → compress without quality loss
├─ Pruning (Ch.10) → remove redundant weights
└─ Mixed Precision (Ch.10) → 2× training speedup

Layer 4: Self-Supervised Learning
├─ Contrastive (Ch.7) → leverage unlabeled data at scale
└─ Masked Autoencoding (Ch.8) → state-of-art pretraining

Layer 3: Advanced Tasks
├─ Object Detection (Ch.3–4) → bounding boxes
├─ Semantic Segmentation (Ch.5) → pixel-level classes
└─ Instance Segmentation (Ch.6) → object boundaries

Layer 2: Production Architectures
├─ ResNets (Ch.1) → skip connections, depth
└─ Efficient Nets (Ch.2) → mobile, edge deployment

Layer 1: Fundamentals (from 03_neural_networks)
└─ CNNs (Ch.5) → convolution, pooling, feature maps
```

**The through-line**: Start with ResNet (Ch.1), use it as the backbone for Faster R-CNN (Ch.3) and Mask R-CNN (Ch.6), pretrain with SimCLR (Ch.7), swap to EfficientNet (Ch.2) for edge deployment, distill (Ch.9), prune (Ch.10) → **production-ready CV system**.

---

## Cross-Track Integration

This track builds on and connects to:

| Track | Connection | Why It Matters |
|-------|-----------|----------------|
| **03-Neural Networks** | Foundation (CNNs Ch.5, Transformers Ch.10) | Must understand convolution before ResNets |
| **Multimodal AI** | Vision Transformers (ViT), CLIP | This track ends where Multimodal AI begins |
| **AI/Fine-Tuning** | Transfer learning, LoRA | Fine-tuning concepts transfer to CV (frozen backbones) |
| **AI Infrastructure** | Quantization (Ch.3), Distributed Training | Complementary optimization techniques |
| **08-Ensemble Methods** | Stacking detectors, ensemble predictions | Boost mAP via detector ensembles (YOLOv5 + Faster R-CNN) |

---

## Success Criteria

**Track is complete when**:
1. ✅ All 10 chapters have theory (README.md) + hands-on code (notebook.ipynb)
2. ✅ ProductionCV grand challenge passes all 5 constraints
3. ✅ Learner can:
   - Choose production CNN architectures (ResNet vs EfficientNet)
   - Build object detection systems (YOLO, Faster R-CNN)
   - Implement semantic segmentation (U-Net)
   - Apply self-supervised learning (SimCLR on unlabeled data)
   - Compress models for edge deployment (distillation, pruning)
4. ✅ Clear differentiation vs 03_neural_networks (advanced → production)
5. ✅ Bridges to Multimodal AI track (self-supervised → CLIP, ResNet → ViT)

---

## Open Questions & Decisions Needed

1. **Dataset**: Create custom synthetic retail shelf dataset OR use existing (e.g., Retail Product Checkout)?
2. **Framework**: PyTorch only OR dual notebooks (Keras + PyTorch) like 03_neural_networks?
3. **Depth vs Breadth**: Cover all YOLO versions (v3, v5, v8) OR focus on YOLOv5 + concepts?
4. **Hardware**: Assume GPU access OR CPU-friendly examples (downsampled images)?
5. **Numbering**: Keep as "09_advanced_deep_learning" OR insert earlier in sequence?

---

## Notes & Considerations

**Why this track is necessary:**
- 03_neural_networks teaches "how CNNs work" (convolution math, backprop)
- This track teaches "which CNN to use and when" (production architecture selection)
- Gap: learners finish 03_neural_networks but can't build object detection systems

**Why this track is focused:**
- Excludes ViT (covered in Multimodal AI)
- Excludes quantization (covered in AI Infrastructure)
- Excludes fine-tuning LLMs (covered in AI track)
- **Focus**: Computer vision tasks beyond classification

**Connection to industry:**
- Object detection: surveillance, autonomous vehicles, retail analytics
- Semantic segmentation: medical imaging, satellite imagery, autonomous driving
- Self-supervised learning: leverage massive unlabeled datasets (production cost reduction)
- Model compression: edge AI (manufacturing, robotics, mobile)

---

**Status**: 🟡 PLANNING — awaiting decision on dataset creation and framework choice before chapter development begins.
