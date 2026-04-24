# Ch.8 — TensorBoard

![Animation: instrumentation-guided training decisions move test MAE from about 78k to 51k.](img/ch08-tensorboard-needle.gif)

*Visual takeaway: monitoring first reveals divergence early, then early stopping and scheduler tuning turn that visibility into measurable validation gains.*

> **The story.** Until **2015**, training a deep network was a black box: you watched a loss number tick down in a terminal and hoped. **TensorBoard** shipped with TensorFlow 0.6 in November 2015, and for the first time engineers could *see* what was happening inside training — weight and gradient histograms, activation distributions, the live computation graph, the **embedding projector** that visualised learned representations in 2D/3D. The idea spread fast: PyTorch added native TensorBoard support in 2019 (`torch.utils.tensorboard`); **Weights & Biases** (2017) and **MLflow** (2018) extended the model to remote experiment tracking with hosted dashboards — the foundations of the MLOps notes. Today, training a serious model without instrumentation is considered an unforced error.
>
> **Where you are in the curriculum.** Your network from [Ch.3](../ch03-backprop-optimisers/) trained to convergence — but what actually happened inside? Loss curves show the output; TensorBoard shows the internals: weight distributions drifting (or vanishing), gradients exploding (or dead), and the embedding projector revealing whether the network learned meaningful feature representations. If [Ch.3](../ch03-backprop-optimisers/) was "turn on the engine," this chapter is "read the instruments" — the practical skill that separates working data scientists from people who copy-paste training scripts.
>
> **Notation in this chapter.** Most of TensorBoard is software, not symbols, but the four objects you log are: **scalars** — single numbers per training step (loss, accuracy, learning rate); **histograms** — the empirical distribution of a tensor over time (typically the weights $W^{(\ell)}$ or gradients $\nabla_{W^{(\ell)}}\mathcal{L}$ at each layer $\ell$); **embeddings** — high-dimensional vectors $\mathbf{e}\in\mathbb{R}^d$ projected with PCA / t-SNE / UMAP for inspection (see [Ch.13](../../07-UnsupervisedLearning/ch02-dimensionality-reduction/)); **graphs / profiles** — the computation graph and per-op GPU/CPU timeline. The two diagnostic numbers you always watch: $\|\nabla_{W^{(\ell)}}\mathcal{L}\|$ — gradient magnitude per layer (vanishing or exploding?), and $\|W^{(\ell)}\|$ — weight magnitude (drifting or collapsing?).

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Launch **UnifiedAI** — a production home valuation system satisfying 5 constraints:
> 1. **ACCURACY**: <$50k MAE — 2. **GENERALIZATION**: Unseen districts — 3. **MULTI-TASK**: Value + Segment — 4. **INTERPRETABILITY**: Explainable — 5. **PRODUCTION**: Scale + Monitor

**What we know so far:**
- ✅ Ch.1-15: Achieved Constraints #1-4, understand loss functions from first principles
- ✅ Can train accurate, generalizable, interpretable models
- ❌ **But training is still a black box!**

**What's blocking us:**
⚠️ **Can't debug training failures**

Engineer reports: "Model trained for 50 epochs, validation loss stopped decreasing at epoch 30, but I kept training — wasted 20 epochs and $50 in compute!"

**Common training failures with no visibility:**
1. **Vanishing gradients**: Loss decreases slowly, then plateaus — but why?
2. **Exploding gradients**: Loss = NaN at epoch 3 — which layer caused it?
3. **Dead neurons**: Accuracy stuck at 75% — are neurons dying (ReLU outputting 0 always)?
4. **Overfitting starts early**: Validation loss increases after epoch 15 — but we only check at epoch 50!

**Why this matters for production:**
- **Cost**: Wasted compute = wasted money ($50/run × 20 wasted epochs = $1000 wasted)
- **Time**: Debugging training failures takes days without diagnostics
- **Constraint #5 partial**: Production requires **monitoring** — need to see what's happening during training

**What this chapter unlocks:**
⚡ **TensorBoard — training instrumentation:**
1. **Loss curves**: Plot train/val loss per epoch → see overfitting start
2. **Weight histograms**: Track weight distributions → detect dead neurons, weight collapse
3. **Gradient histograms**: Monitor gradient magnitudes → catch vanishing/exploding gradients
4. **Embedding projector**: Visualize learned representations → validate feature learning

⚡ **Constraint #5 PARTIAL**: Monitoring infrastructure in place — still need versioning, A/B testing (Ch.19)

---

## 1 · Core Idea

TensorBoard is TensorFlow's (and PyTorch's) training dashboard. It reads event files written during training and renders interactive visualisations in a browser. The key panels:

| Panel | What it shows | Primary diagnostic use |
|---|---|---|
| **Scalars** | Loss and metrics per epoch/step | Detect overfitting, underfitting, learning rate issues |
| **Histograms** | Weight and gradient distributions over time | Detect vanishing/exploding gradients, dead neurons |
| **Distributions** | Same as histograms but as an overlay area chart | See the spread of activations evolve |
| **Projector** | High-dimensional embeddings reduced via PCA or t-SNE | Validate that learned representations cluster meaningfully |
| **Images** | Logged tensors rendered as images | Inspect feature maps, sample predictions, data augmentation |
| **Graphs** | Computational graph of the model | Verify architecture, spot unexpected operations |
| **Profile** | GPU/CPU utilisation timeline | Identify training bottlenecks |

TensorBoard is not a debugging tool for code errors. It is a **training diagnostics tool** for model behaviour.

---

## 2 · Running Example

We return to the **Ch.5 training loop**: a small Keras neural network trained on California Housing. We instrument it with the `TensorBoard` callback to emit:

1. Training and validation MSE per epoch (Scalars)
2. Weight and bias distributions per layer (Histograms)
3. Gradient histograms (Histograms — requires manual summary writing or `histogram_freq`)
4. Intermediate activations as a custom embedding (Projector)

Dataset: **California Housing** (`sklearn.datasets.fetch_california_housing`)  
Network: 3-layer dense network (same as Ch.5)

---

## 3 · Math

TensorBoard itself has no mathematics — it logs tensors and renders them. But the diagnostics it reveals connect to the mathematical concepts from earlier chapters:

### 3.1 What Histograms Reveal

Each layer's weight matrix $\mathbf{W}$ at epoch $t$ has a distribution. Healthy training shows this distribution shifting and narrowing as learning proceeds. Warning signs:

**Vanishing gradients:** weight histograms stop changing in early layers (Ch.5 — vanishing gradient problem). The gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(L)}} \cdot \prod_{k=2}^{L} \frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(k-1)}}$ shrinks exponentially for deep networks with sigmoid activations.

**Exploding gradients:** weight histograms spread wider and wider; $\|\nabla\|$ grows uncontrollably.

**Dead neurons:** if the gradient histogram for a layer is a spike at zero, those neurons have died (ReLU outputs $\max(0, x)$; once input is always negative, gradient is always 0).

### Numeric Example — Gradient Health Check

| Training step | Mean gradient | Std gradient | Max \|grad\| | Health indicator |
|---------------|---------------|--------------|--------------|------------------|
| 100 | −0.003 | 0.052 | 0.21 | ✅ Healthy |
| 500 | −0.001 | 0.008 | 0.04 | ⚠️ Possibly vanishing |
| 1000 | 0.000 | 0.001 | 0.003 | ❌ Vanishing — check lr/init |

Rule of thumb: if `std(grad) < 1e-3` at mid-training, gradients are vanishing. If `max(|grad|) > 10`, gradients are exploding. TensorBoard's histogram tab shows this at a glance.

### 3.2 Embedding Projector

The projector takes a tensor $\mathbf{Z} \in \mathbb{R}^{n \times d}$ (the internal representation of $n$ samples from a hidden layer of dim $d$) and reduces it to 2D/3D via PCA or t-SNE for visualisation. If the network learned useful features, samples of the same class should cluster in the embedding space.

---

## 4 · Step by Step

```
Setting up TensorBoard logging:
1. Define log_dir = 'logs/run_<timestamp>'
2. Create tf.keras.callbacks.TensorBoard(
       log_dir=log_dir,
       histogram_freq=1,       # log weight histograms every epoch
       write_graph=True,       # log the computational graph
       write_images=False,
       update_freq='epoch'     # log scalars every epoch (not every batch)
   )
3. Pass callback to model.fit(callbacks=[tb_callback])
4. After training: tensorboard --logdir logs/

Reading the dashboard:
5. Scalars: is val_loss decreasing? Does it flatten (underfitting) or diverge from train_loss (overfitting)?
6. Histograms: do early-layer weights change between epochs? If frozen → vanishing gradients.
7. Check gradient histograms: are they wide (normal) or spike at 0 (dead neurons)?
8. Projector: load the embedding tensor and metadata file; visualise with t-SNE in browser.

Custom summaries (beyond the callback):
9. tf.summary.scalar('custom_metric', value, step=epoch)
10. tf.summary.histogram('layer_output', tensor, step=epoch)
11. tf.summary.image('predictions', img_tensor, step=epoch)
```

---

## 5 · Key Diagrams

### TensorBoard data flow

```mermaid
flowchart LR
    A["model.fit\n+ TensorBoard callback"] --> B["SummaryWriter\nwrites event files\nto log_dir/"]
    B --> C["tensorboard --logdir log_dir/\n(starts local HTTP server)"]
    C --> D["Browser\nlocalhost:6006"]
    D --> E1["Scalars tab\ntrain/val loss curve"]
    D --> E2["Histograms tab\nweight + gradient dists"]
    D --> E3["Projector tab\nembedding UMAP/t-SNE"]
```

### Vanishing gradient signature in histograms

```
Epoch 1   │ ████████████████  (weights moving, wide gradient)
Epoch 5   │ ██████████        (gradients shrinking)
Epoch 10  │ ████              (early layers barely moving)
Epoch 20  │ █                 (first layer frozen — vanishing gradient!)
Layer 4   │ ████████████████  (last layer still learning)
```

### Healthy vs unhealthy scalar curves

```
Healthy:                    Overfitting:            Underfitting:
 train_loss = val_loss       train_loss ↓           both curves flat
 both decreasing             val_loss ↑             or very slowly ↓
 (small gap)                 (widening gap)
```

---

## 6 · Hyperparameter Dial

| Parameter | Too low / off | Sweet spot | Too high / on |
|---|---|---|---|
| **histogram_freq** | 0 — no weight histograms | 1 (every epoch) | Every batch — huge disk usage, no value |
| **update_freq** | 'epoch' — one scalar per epoch | 'epoch' for long runs | 'batch' — floods Scalars; only useful for debugging a single epoch |
| **profile_batch** | 0 — no profiling | 2 (discard warmup batch 1) | Multiple batches — significant training overhead |
| **write_graph** | False | True once to verify architecture | — |
| **embeddings_freq** | 0 — no projector | 5–10 — every few epochs | 1 — slow; embedding data can be large |

---

## 7 · Code Skeleton

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y    = housing.data, housing.target.reshape(-1, 1)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
```

```python
# ── Build model (same architecture as Ch.5) ───────────────────────────────────
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(64, activation='relu', name='hidden_1'),
        keras.layers.Dense(32, activation='relu', name='hidden_2'),
        keras.layers.Dense(16, activation='relu', name='hidden_3'),
        keras.layers.Dense(1,                     name='output'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model
```

```python
# ── TensorBoard callback ──────────────────────────────────────────────────────
log_dir = 'logs/ch16_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

tb_callback = keras.callbacks.TensorBoard(
    log_dir        = log_dir,
    histogram_freq = 1,          # weight + bias histograms every epoch
    write_graph    = True,       # log computational graph
    update_freq    = 'epoch',    # scalars per epoch, not per batch
    profile_batch  = 0,          # disabled — enable with profile_batch=2 for perf profiling
)

model = build_model()
history = model.fit(
    X_tr, y_tr,
    validation_data = (X_va, y_va),
    epochs          = 50,
    batch_size      = 256,
    callbacks       = [tb_callback],
    verbose         = 0
)
print(f"Logs written to: {log_dir}")
print("Run: tensorboard --logdir logs/")
```

```python
# ── Custom scalar: learning rate ──────────────────────────────────────────────
summary_writer = tf.summary.create_file_writer(log_dir + '/custom')

class LRLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with summary_writer.as_default():
            tf.summary.scalar('learning_rate',
                              self.model.optimizer.learning_rate.numpy(),
                              step=epoch)

model2  = build_model()
model2.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=30, batch_size=256,
           callbacks=[tb_callback, LRLogger()], verbose=0)
```

```python
# ── Projector: log intermediate embeddings ────────────────────────────────────
import os

embedding_model = keras.Model(inputs=model.input,
                               outputs=model.get_layer('hidden_3').output)
embeddings = embedding_model.predict(X_te[:500], verbose=0)  # (500, 16)

log_emb_dir = os.path.join(log_dir, 'projector')
os.makedirs(log_emb_dir, exist_ok=True)

# Write embeddings as numpy checkpoint
np.savetxt(os.path.join(log_emb_dir, 'feature_vecs.tsv'),
           embeddings, delimiter='\t')

# Optional metadata (true house values for colouring)
with open(os.path.join(log_emb_dir, 'metadata.tsv'), 'w') as f:
    f.write('MedHouseVal\n')
    for v in y_te[:500, 0]:
        f.write(f'{v:.3f}\n')

print("Embedding + metadata written for Projector tab")
```

---

## 8 · What Can Go Wrong

- **Logging every batch for Scalars.** `update_freq='batch'` can create hundreds of thousands of scalar events per run. The TensorBoard UI becomes unresponsive and disk usage bloats. Use `'epoch'` for all but single-epoch debugging where you need per-step visibility.

- **histogram_freq=1 on very large models or datasets.** Computing histograms requires a forward pass through the data and extraction of all weight tensors. On a large model with many layers, this can double training time. Set `histogram_freq=5` (every 5 epochs) if it's too slow.

- **Comparing runs without clearing the log directory.** If you re-run training into the same `log_dir`, TensorBoard aggregates both runs in the same Scalars view — curves overlap confusingly. Always use a timestamped subdirectory (`logs/run_YYYYMMDD_HHMMSS/`) for each experiment.

- **Hard-coding `log_dir='logs/'` in a cloud or shared environment.** Write to a unique path per run. In cloud training jobs, use environment variables or job IDs.

- **Interpreting a dead gradient histogram as "converged."** If the gradient histogram for a layer is a spike at zero from epoch 5 onwards, the layer is not converged — it is frozen due to the dying-ReLU problem or a vanishing-gradient issue. Fix: use LeakyReLU or He initialisation and check earlier-layer learning rates.

---

## Bridge to Ch.9 — From Sequences to Attention

Ch.8 gave you the instruments to diagnose a trained network. Ch.9 is a short **bridge chapter** that introduces attention as a soft dictionary lookup — dot product + softmax + weighted sum of values — without any transformer machinery. It exists so the full transformer architecture in Ch.10 lands softly: every symbol you will meet there ($Q$, $K$, $V$, the $T\times T$ attention matrix, positional encoding) is introduced first in pure-numpy form in Ch.9.

The full 10-chapter arc:

```
Ch.1:     XOR Problem (why linear models fail; need for depth)
Ch.2:     Neural Networks (forward pass, activation functions)
Ch.3:     Backprop & Optimisers (how networks actually learn)
Ch.4:     Regularisation (preventing overfitting)
Ch.5:     CNNs (spatial feature extraction)
Ch.6:     RNNs & LSTMs (sequence modelling)
Ch.7:     MLE & Loss Functions (why the losses are what they are)
Ch.8:     TensorBoard (diagnosing training with instruments)
Ch.9:     From Sequences to Attention (the bridge to transformers)
Ch.10:    Transformers & Attention (the architecture behind modern LLMs)
```


## Illustrations

![TensorBoard panels — scalars, weight histograms, embedding projector, and diagnostics dashboard](img/ch16%20tensorboard.png)

## 9 · Where This Reappears

Instrumentation and monitoring patterns reappear across the repo and in production notes:

- Experiment tracking and logs in AIInfrastructure (MLOps and experiment management).
- Diagnostic recipes referenced in training notebooks across ML topics.
- Production monitoring and profiling in deployment guides.

Refine these cross-references during editorial cleanup.
