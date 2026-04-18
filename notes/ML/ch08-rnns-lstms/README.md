# Ch.8 — RNNs / LSTMs / GRUs

> **The story.** **John Hopfield** (1982) introduced recurrent dynamics into neural nets; **Jeffrey Elman** (1990) gave us the simple RNN we still teach today — a hidden state that gets fed back as input on the next time step. The architecture was elegant and almost completely useless in practice: **Sepp Hochreiter** showed in his 1991 diploma thesis that gradients vanish (or explode) exponentially through time, so RNNs couldn't learn dependencies more than 5–10 steps apart. The fix came from Hochreiter himself, with **Jürgen Schmidhuber**, in **1997**: the **Long Short-Term Memory** cell — a tiny network of gates (input, forget, output) that lets information flow unchanged across hundreds of steps. **GRUs** (Cho et al., 2014) trimmed the gate count to two and matched LSTM performance with fewer parameters. From the late 1990s through 2017 the LSTM was the standard answer for sequence modelling — speech, translation, captioning — until the transformer in [Ch.18](../ch18-transformers/) replaced it almost overnight.
>
> **Where you are in the curriculum.** A dense network ([Ch.4](../ch04-neural-networks/)) sees a flat vector with no sense of order. A CNN ([Ch.7](../ch07-cnns/)) exploits spatial locality. Sequential data has a third structure: **temporal ordering and long-range dependencies**. The platform now tracks how district median house values change month by month. RNNs carry a hidden state forward through time; LSTMs add gated memory to preserve what matters over many steps. Master this chapter and the motivation for attention in [Ch.17](../ch17-sequences-to-attention/) will feel inevitable.
>
> **Notation in this chapter.** $\mathbf{x}_t$ — input at time step $t$; $\mathbf{h}_t$ — the **hidden state** (the recurrent network's memory); $\mathbf{c}_t$ — the **cell state** (LSTM's long-term memory bus); $f_t,i_t,o_t$ — LSTM **forget**, **input**, and **output** gates; $\tilde{\mathbf{c}}_t$ — the candidate cell update; $W,U$ — weight matrices applied to the input and to the previous hidden state respectively; $\sigma$ — sigmoid (used inside gates); $\tanh$ — the squashing nonlinearity; $T$ — sequence length; **BPTT** — *back-propagation through time*.

---

## 1 · Core Idea

A **Recurrent Neural Network** processes a sequence one step at a time, updating a hidden state that summarises everything seen so far:

```
Dense (Ch.4): input → output (no memory of previous inputs)
RNN: [x_1, x_2, ..., x_T] → h_1 → h_2 → ... → h_T → output
 each h_t depends on h_{t-1} and x_t simultaneously
```

The problem: gradients of the loss with respect to early steps shrink exponentially as they flow back through each time step (vanishing gradient). **LSTMs** solve this with a separate cell state — a "conveyor belt" that carries information across many steps with minimal transformation.

---

## 2 · Running Example

The platform's analytics team wants a **monthly housing price index forecaster**. Given the last $T$ months of median house values for a district, predict next month's value.

Dataset: **Synthetic monthly price index** 
Features: normalised median house values, time index, 12-month seasonal signal 
Target: next month's normalised median house value 
Sequence length: 12 months look-back (one full year of context)

A single district's price index follows a trend + seasonal cycle + noise — exactly the structure an LSTM captures (trend in the cell state, seasonal pattern in the hidden state, noise suppressed).

---

## 3 · Math

### 3.1 Vanilla RNN

At each time step $t$, the RNN computes a new hidden state from the previous hidden state $\mathbf{h}_{t-1}$ and the current input $\mathbf{x}_t$:

$$\mathbf{h}_t = \tanh \left(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h\right)$$

$$\hat{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\mathbf{x}_t$ | $(d,)$ | Input at step $t$ — one feature per district per month |
| $\mathbf{h}_t$ | $(H,)$ | Hidden state — compressed summary of the sequence so far |
| $\mathbf{W}_{hh}$ | $(H, H)$ | Recurrent weight — how much the past hidden state contributes |
| $\mathbf{W}_{xh}$ | $(H, d)$ | Input weight — how much the current input contributes |
| $H$ | scalar | Hidden size — the main capacity dial |

The **same weights** $\mathbf{W}_{hh}$, $\mathbf{W}_{xh}$ are shared across every time step. An RNN on $T$ steps is a deep network with $T$ identical layers — and backprop runs through all of them.

### 3.2 Vanishing Gradient

Backprop through time (BPTT) computes $\partial \mathcal{L} / \partial \mathbf{h}_0$ by chaining Jacobians:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_0} = \prod_{t=1}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \prod_{t=1}^{T} \mathbf{W}_{hh}^\top \cdot \mathrm{diag} \left(1 - \mathbf{h}_t^2\right)$$

If the spectral radius of $\mathbf{W}_{hh}$ is $< 1$, this product shrinks exponentially with $T$. Gradients from early steps become numerically zero — the network cannot learn dependencies longer than ~10 steps. **Exploding gradients** occur when the radius is $> 1$: gradients blow up. Fix: gradient clipping.

### 3.3 LSTM Cell

The Long Short-Term Memory adds a **cell state** $\mathbf{c}_t$ — a direct, gated highway that lets gradients flow without repeated multiplication by $\mathbf{W}_{hh}$.

Let $[\mathbf{h}_{t-1}; \mathbf{x}_t]$ denote the concatenated vector of shape $(H+d,)$.

**Forget gate** — what fraction of the old cell state to discard:

$$\mathbf{f}_t = \sigma \left(\mathbf{W}_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f\right)$$

**Input gate** — how much new information to write:

$$\mathbf{i}_t = \sigma \left(\mathbf{W}_i [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i\right)$$

**Candidate cell** — the new information to potentially add:

$$\tilde{\mathbf{c}}_t = \tanh \left(\mathbf{W}_c [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c\right)$$

**Cell state update** — the conveyor belt:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output gate** — what portion of the cell state to expose as hidden state:

$$\mathbf{o}_t = \sigma \left(\mathbf{W}_o [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o\right)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

$\odot$ is element-wise multiplication. The cell state $\mathbf{c}_t$ flows through with only element-wise operations — no matrix multiply — so gradients can flow back many steps without vanishing.

**Parameter count per LSTM layer:** $4 \times (H^2 + H \cdot d + H)$ — four gate weight matrices plus biases.

### 3.4 GRU

The Gated Recurrent Unit is a lighter alternative with two gates (no separate cell state):

**Reset gate** — how much of the past hidden state to use when computing the candidate:

$$\mathbf{r}_t = \sigma \left(\mathbf{W}_r [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_r\right)$$

**Update gate** — how much of the old hidden state to keep (analogous to LSTM forget + input):

$$\mathbf{z}_t = \sigma \left(\mathbf{W}_z [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_z\right)$$

**Candidate hidden state:**

$$\tilde{\mathbf{h}}_t = \tanh \left(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_h\right)$$

**Output hidden state:**

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**GRU vs LSTM:**

| Property | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate $\mathbf{c}_t$ | None — $\mathbf{h}_t$ does both |
| Parameters | $4 \times (H^2 + Hd + H)$ | $3 \times (H^2 + Hd + H)$ |
| Training speed | Slower | ~25% faster |
| Long sequences | Slight edge | Comparable |
| When to use | Default choice | When training time is tight |

---

## 4 · Step by Step

```
1. Build the dataset as sliding windows
 └─ given T months of prices as x, predict month T+1 as y
 └─ normalise: (x - mean) / std (fit on training split only)

2. Define the model
 └─ LSTM(units=H, return_sequences=False) if single-output regression
 └─ LSTM(units=H, return_sequences=True) if predicting all T+1 steps

3. Compile
 └─ loss = MSE (regression target: next month's price)
 └─ optimizer = Adam (default lr=1e-3)

4. Train with early stopping
 └─ monitor val_loss, patience=10, restore_best_weights=True

5. Predict
 └─ feed the last T real values → get ŷ for month T+1
 └─ inverse-transform: (ŷ × std) + mean

6. Evaluate with RMSE and MAE (Ch.9 gives the full metrics toolkit)
```

---

## 5 · Key Diagrams

### Unrolled RNN (3 steps)

```
 x_1 x_2 x_3
 │ │ │
 ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
h_0 │ RNN │ │ RNN │ │ RNN │
───►│ cell │───►│ cell │───►│ cell │───► ŷ
 └───────┘ └───────┘ └───────┘
 h_1 h_2 h_3

Same W_hh and W_xh used at every step — shared weights across time.
```

### LSTM cell internals

```mermaid
flowchart LR
 subgraph Inputs
 H_prev["h_{t-1}"]
 X["x_t"]
 C_prev["c_{t-1}"]
 end

 F["Forget gate\nσ(W_f·[h,x]+b_f)"]
 I["Input gate\nσ(W_i·[h,x]+b_i)"]
 G["Candidate\ntanh(W_c·[h,x]+b_c)"]
 O["Output gate\nσ(W_o·[h,x]+b_o)"]
 C_new["c_t = f⊙c_{t-1} + i⊙g̃"]
 H_new["h_t = o⊙tanh(c_t)"]

 H_prev --> F & I & G & O
 X --> F & I & G & O
 C_prev --> C_new
 F --> C_new
 I --> C_new
 G --> C_new
 C_new --> H_new
 O --> H_new
```

### Vanishing gradient: RNN vs LSTM

```
Time steps → 1 5 10 20 50
 │ │ │ │ │
RNN gradient: 1.0 0.3 0.01 0.0 0.0 (× W_hh at each step — decays)
LSTM gradient: 1.0 0.9 0.8 0.7 0.5 (additive cell path — preserved)
```

### Sequence window construction

```
Price series: [p1, p2, p3, p4, p5, p6, p7, ...]

Window T=3:
 Input [p1, p2, p3] → target p4
 Input [p2, p3, p4] → target p5
 Input [p3, p4, p5] → target p6
 ...
```

### RNN vs LSTM vs GRU — when to use

```mermaid
flowchart TD
 A["Sequence data?"] -->|yes| B["Dependencies\n> 10 steps?"]
 B -->|no| C["Vanilla RNN\n(simple, fast)"]
 B -->|yes| D["Training time\nconstrained?"]
 D -->|no| E["LSTM\n(default choice)"]
 D -->|yes| F["GRU\n(~25% fewer params)"]
 A -->|no| G["Dense / CNN\n(no recurrence needed)"]
```

---

## 6 · Hyperparameter Dial

| Dial | Too low | Sweet spot | Too high |
|---|---|---|---|
| **Hidden units** $H$ | underfits long patterns | 32–128 for most time series | overfits, slow |
| **Sequence length** $T$ | misses long dependencies | 12–52 steps (month/week) | slow BPTT, more vanishing gradient |
| **Stacked layers** | shallow temporal hierarchy | 1–2 for most tasks | vanishing gradient without residual |
| **Dropout** (on recurrent connections) | no regularisation | 0.1–0.3 between LSTM layers | underfits |
| **Gradient clip** | exploding gradient | 1.0–5.0 | clips too aggressively, slows learning |

The single most impactful dial for sequence length is **hidden units** — double it before adding a second LSTM layer.

---

## 7 · Code Skeleton

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Synthetic monthly price index (120 months = 10 years) ────────────────────
def make_price_series(n_months=120, seed=42):
 """Synthetic district median house value index.
 Components: linear trend + 12-month seasonality + noise.
 """
 rng = np.random.default_rng(seed)
 t = np.arange(n_months)
 trend = 0.005 * t # slow upward drift
 seasonal = 0.15 * np.sin(2 * np.pi * t / 12) # annual cycle
 noise = rng.normal(0, 0.05, n_months)
 return 2.0 + trend + seasonal + noise # base value ~2.0 ($200k)

prices = make_price_series()

# ── Sliding window dataset ────────────────────────────────────────────────────
def make_windows(series, T=12):
 """Convert 1-D series to (X, y) sliding windows.
 X: (N, T, 1) y: (N,)
 """
 X, y = [], []
 for i in range(len(series) - T):
 X.append(series[i:i+T, np.newaxis])
 y.append(series[i+T])
 return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

T = 12
X, y = make_windows(prices, T=T)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
 shuffle=False) # no shuffle for time series!

# Normalise using training statistics only
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
y_train = (y_train - mean) / std
y_test = (y_test - mean) / std

print(f"X_train: {X_train.shape} X_test: {X_test.shape}")
```

```python
# ── Manual RNN forward pass (NumPy) ──────────────────────────────────────────
def rnn_forward(X_seq, W_xh, W_hh, b_h, W_hy, b_y):
 """Single-step RNN forward pass for one sequence.
 X_seq: (T, d)
 Returns: h_sequence (T, H), y_hat (scalar)
 """
 H = W_hh.shape[0]
 h = np.zeros(H)
 hs = []
 for x_t in X_seq:
 h = np.tanh(W_xh @ x_t + W_hh @ h + b_h)
 hs.append(h)
 y_hat = W_hy @ hs[-1] + b_y
 return np.array(hs), y_hat

# Tiny demo: H=4, d=1
H, d = 4, 1
rng = np.random.default_rng(0)
W_xh = rng.normal(0, 0.1, (H, d))
W_hh = rng.normal(0, 0.1, (H, H))
b_h = np.zeros(H)
W_hy = rng.normal(0, 0.1, (1, H))
b_y = np.zeros(1)

hs, y_hat = rnn_forward(X_train[0], W_xh, W_hh, b_h, W_hy, b_y)
print(f"Hidden states shape: {hs.shape} Prediction: {y_hat[0]:.4f}")
```

```python
# ── LSTM with Keras ───────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

lstm_model = keras.Sequential([
 layers.Input(shape=(T, 1)),
 layers.LSTM(64, return_sequences=False),
 layers.Dense(32, activation='relu'),
 layers.Dense(1), # regression — no activation
], name='HousePriceForecaster_LSTM')

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

early_stop = keras.callbacks.EarlyStopping(
 monitor='val_loss', patience=15, restore_best_weights=True)

history = lstm_model.fit(
 X_train, y_train,
 epochs=200, batch_size=16,
 validation_split=0.15,
 callbacks=[early_stop],
 verbose=0,
)

y_pred = lstm_model.predict(X_test, verbose=0).ravel()

# Inverse-transform
y_pred_real = y_pred * std + mean
y_test_real = y_test * std + mean

rmse = np.sqrt(np.mean((y_pred_real - y_test_real) ** 2))
mae = np.mean(np.abs(y_pred_real - y_test_real))
print(f"RMSE: {rmse:.4f} MAE: {mae:.4f} (units: $100k)")
```

```python
# ── GRU (drop-in replacement) ─────────────────────────────────────────────────
gru_model = keras.Sequential([
 layers.Input(shape=(T, 1)),
 layers.GRU(64, return_sequences=False),
 layers.Dense(32, activation='relu'),
 layers.Dense(1),
], name='HousePriceForecaster_GRU')

gru_model.compile(optimizer='adam', loss='mse')
# GRU trains ~25% faster for the same hidden size
```

---

## 8 · What Can Go Wrong

- **Shuffling a time-series split.** Using `train_test_split` with `shuffle=True` on sequential data leaks future information into training. Always split chronologically: train on earlier months, test on later months. The validation split inside `model.fit` should also be the `validation_split` fraction from the **end** of the training data.

- **Forgetting to normalise the target.** MSE on raw house prices (range 0.5–5.0 × $100k) is fine, but MSE on un-normalised sequences with a strong trend drives the optimiser to focus on the trend rather than the pattern, producing inflated apparent losses. Normalise both X and y using training statistics.

- **Not clipping gradients on long sequences.** For $T > 50$, vanilla LSTM training without gradient clipping (`clipnorm=1.0` in the optimiser) can explode within the first few epochs — the loss goes to NaN. Always add `clipnorm` when the sequence is long.

- **Using `return_sequences=True` on the last LSTM layer before a Dense output.** This outputs `(N, T, H)` — a sequence prediction — rather than the single `(N, H)` summary needed for regression. Use `return_sequences=False` on the final LSTM layer, or add a `Flatten` / `GlobalAveragePooling1D`.

- **Treating RNN hidden size and CNN filter count as equivalent dials.** An LSTM with `H=64` has $(4 × (64^2 + 64 + 64)) = 16,900$ parameters per step. Jumping straight to `H=256` quadruples parameters and training time. Start small, increase if validation loss is still decreasing.

---

## 9 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| RNN recurrence equation: $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})$ | Why does the vanilla RNN suffer from vanishing gradient? (product of Jacobians decays exponentially in $T$) | "LSTM solves vanishing gradient by using attention" — wrong; it uses a gated cell state |
| What the four LSTM gates do (forget, input, candidate, output) | Why is the cell state $\mathbf{c}_t$ the key to long-term memory? (additive update avoids repeated multiplication) | Claiming `return_sequences=True` is always needed — it depends on whether you want step-by-step or a single final prediction |
| GRU uses two gates (reset, update) and no separate cell state | When would you choose GRU over LSTM? (training speed / parameter budget) | Shuffling time-series validation data — always split chronologically |
| `shuffle=False` in `train_test_split` for time series | How does gradient clipping fix exploding gradients? (caps the norm; doesn't fix vanishing) | Confusing sequence length $T$ with hidden size $H$ when asked "how do you increase model capacity?" |
| **Bidirectional RNN:** runs two RNNs in opposite directions over the sequence and concatenates hidden states at each step; doubles parameters and compute, but gives each position access to both past and future context — critical for NER and classification tasks | "When would you use a bidirectional RNN?" | "Bidirectional RNNs are always better" — they cannot be used autoregressively (generation, online inference) because future tokens are unavailable at the time of prediction; transformer encoder attention achieves the same effect more efficiently |
| **Teacher forcing:** during training, feed the ground-truth token from step $t-1$ as input at step $t$ instead of the model's own prediction; speeds convergence but creates **exposure bias** — the model never learns to recover from its own mistakes. **Scheduled sampling** gradually replaces teacher tokens with model predictions during training | "What is exposure bias and how do you mitigate it?" | "Teacher forcing makes the model more accurate" — it makes training faster but widens the train-inference distribution gap; inference errors compound because the model trained on gold inputs, not its own |

---

## Bridge to Chapter 9

Ch.8 showed how to train recurrent models and get predictions. But predictions alone are not enough — a model that's 80% accurate on a balanced test set and a model that's 80% accurate on an imbalanced test set are telling you very different things. Ch.9 — **Metrics Deep Dive** — closes the loop: it takes the classifier from Ch.2, the regressor from Ch.1, and examines every angle through which a model can look good or bad on paper while failing in production.


## Illustrations

![RNN and LSTM architecture — hidden-state flow and gating through time steps](img/ch8%20rnn.png)
