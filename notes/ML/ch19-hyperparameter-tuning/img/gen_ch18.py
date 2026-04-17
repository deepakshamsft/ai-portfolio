"""Generate ch18 hyperparameter tuning.png — 9-panel cheat-sheet of
the core hyperparameters you tune when training a neural network.

Panel grid (3x3):
    [01 Learning rate]        [02 Optimisers]           [03 Batch size]
    [04 Initializers]         [05 Dropout]              [06 Loss functions]
    [07 Layer types]          [08 Depth (# layers)]     [09 When to get more data]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Gentle xkcd: keep sketchy look but keep text legible (no stroke outlines).
plt.xkcd(scale=0.3, length=100, randomness=1)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(20, 16), facecolor="white")
fig.suptitle("Hyperparameter Tuning — What Each Dial Does",
             fontsize=24, fontweight="bold", y=0.985)

gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38,
                      left=0.05, right=0.97, top=0.94, bottom=0.06)

# ── palette ──────────────────────────────────────────────────────
BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#C0392B", "#8E44AD", "#1ABC9C",
    "#2C3E50", "#BDC3C7",
)

rng = np.random.default_rng(0)


def annotate(ax, text, color="#2C3E50", loc=(0.03, 0.97), fs=9):
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, lw=0.8, alpha=0.9))


# ═════════════════════════════════════════════════════════════════
# 01 · Learning rate
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])
epochs = np.arange(0, 60)

# too low
loss_low = 1.0 * np.exp(-epochs / 120) + 0.45
# just right
loss_good = 1.0 * np.exp(-epochs / 10) + 0.08
# too high - diverges / oscillates
rng2 = np.random.default_rng(1)
loss_high = 1.0 * np.exp(-epochs / 30) + 0.2 + 0.3 * np.abs(np.sin(epochs / 2)) * np.exp(epochs / 80)

ax.plot(epochs, loss_low, color=BLUE, lw=2.2, label="η too low")
ax.plot(epochs, loss_good, color=GREEN, lw=2.2, label="η just right")
ax.plot(epochs, loss_high, color=RED, lw=2.2, label="η too high")
ax.set_title("01 · Learning rate (η)", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend(loc="upper right", fontsize=8, frameon=True)
ax.set_ylim(0, 2.0)
annotate(ax, "controls step size:\n   too small → slow\n   too large → diverges")

# ═════════════════════════════════════════════════════════════════
# 02 · Optimisers (trajectory on a saddle-ish surface)
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
xs = np.linspace(-2.5, 2.5, 60)
ys = np.linspace(-2.0, 2.0, 60)
X, Y = np.meshgrid(xs, ys)
# elongated valley: narrow in y, wide in x
Z = 0.15 * X ** 2 + 1.5 * Y ** 2
ax.contour(X, Y, Z, levels=12, colors=GREY, linewidths=0.7, alpha=0.6)

# SGD: zig-zags in narrow direction
sgd_pts = [(-2.2, 1.6)]
for _ in range(40):
    x0, y0 = sgd_pts[-1]
    g = np.array([0.30 * x0, 3.0 * y0])
    sgd_pts.append((x0 - 0.08 * g[0], y0 - 0.08 * g[1]))
sgd = np.array(sgd_pts)
ax.plot(sgd[:, 0], sgd[:, 1], "-o", color=RED, lw=1.5, ms=2, label="SGD")

# Momentum: accumulates, overshoots then settles
mom_pts = [(-2.2, 1.6)]
v = np.zeros(2)
for _ in range(30):
    x0, y0 = mom_pts[-1]
    g = np.array([0.30 * x0, 3.0 * y0])
    v = 0.85 * v + g
    mom_pts.append((x0 - 0.05 * v[0], y0 - 0.05 * v[1]))
mom = np.array(mom_pts)
ax.plot(mom[:, 0], mom[:, 1], "-o", color=ORANGE, lw=1.5, ms=2, label="Momentum")

# Adam: adaptive per-param, smooth straight shot
ad_pts = [(-2.2, 1.6)]
m, v2 = np.zeros(2), np.zeros(2)
for t in range(1, 26):
    x0, y0 = ad_pts[-1]
    g = np.array([0.30 * x0, 3.0 * y0])
    m = 0.9 * m + 0.1 * g
    v2 = 0.999 * v2 + 0.001 * g ** 2
    mh = m / (1 - 0.9 ** t)
    vh = v2 / (1 - 0.999 ** t)
    step = 0.30 * mh / (np.sqrt(vh) + 1e-8)
    ad_pts.append((x0 - step[0], y0 - step[1]))
adam = np.array(ad_pts)
ax.plot(adam[:, 0], adam[:, 1], "-o", color=GREEN, lw=1.8, ms=2.5, label="Adam")

ax.plot(0, 0, "*", color=DARK, ms=14, zorder=5)
ax.set_title("02 · Optimisers", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("w₁")
ax.set_ylabel("w₂")
ax.legend(loc="upper right", fontsize=8)
annotate(ax, "SGD: zig-zags · Momentum: accel.\nAdam: per-param adaptive scale")

# ═════════════════════════════════════════════════════════════════
# 03 · Batch size — noise vs speed tradeoff
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 2])
bs = np.array([8, 16, 32, 64, 128, 256, 512, 1024])

# per-step time: ~constant for small, then grows
step_time = 1.0 + 0.0015 * bs
# noise in gradient ~ 1/sqrt(bs)
noise = 1.0 / np.sqrt(bs) * 3.0
# generalisation gap tends to grow for very large batches (sharp minima)
gen_gap = 0.02 + 0.0003 * bs

ax.plot(bs, noise, "-o", color=BLUE, lw=2, label="grad noise ∝ 1/√B")
ax.plot(bs, step_time, "-s", color=ORANGE, lw=2, label="step time")
ax.plot(bs, gen_gap * 30, "-^", color=RED, lw=2, label="gen. gap ×30")
ax.set_xscale("log", base=2)
ax.set_title("03 · Batch size", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("batch size (log₂)")
ax.set_ylabel("relative")
ax.legend(loc="upper left", fontsize=8)
annotate(ax, "small: noisy updates\nlarge: smoother but sharp minima\nsweet spot: 64–256",
         loc=(0.35, 0.45))

# ═════════════════════════════════════════════════════════════════
# 04 · Initializers — weight distributions
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
rng3 = np.random.default_rng(42)
n_in = 128
zeros = np.zeros(5000)
big = rng3.normal(0, 1.0, 5000)          # too big
xavier = rng3.normal(0, np.sqrt(1 / n_in), 5000)
he = rng3.normal(0, np.sqrt(2 / n_in), 5000)

ax.hist(big, bins=50, alpha=0.45, color=RED, label="N(0,1) — explodes")
ax.hist(he, bins=50, alpha=0.7, color=GREEN, label="He (ReLU)")
ax.hist(xavier, bins=50, alpha=0.7, color=BLUE, label="Xavier (tanh)")
ax.axvline(0, color=DARK, lw=1)
ax.set_title("04 · Initializers", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("initial weight value")
ax.set_ylabel("count")
ax.set_xlim(-1.2, 1.2)
ax.legend(loc="upper right", fontsize=8)
annotate(ax, "zeros → symmetry, no learning\nlarge → explode · Xavier (tanh)\nHe (ReLU): σ = √(2/nᵢₙ)")

# ═════════════════════════════════════════════════════════════════
# 05 · Dropout — train vs test error
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
p = np.linspace(0, 0.9, 50)
# train error rises with dropout (harder to fit)
train_err = 0.05 + 0.8 * p ** 2
# test error: U-shape — low p underfits memorisation, high p underfits capacity
test_err = 0.35 - 0.85 * p + 1.4 * p ** 2

ax.plot(p, train_err, color=BLUE, lw=2.5, label="train error")
ax.plot(p, test_err, color=RED, lw=2.5, label="test error")
best = p[np.argmin(test_err)]
ax.axvline(best, color=GREEN, ls="--", lw=1.5)
ax.text(best + 0.02, 0.55, f"sweet spot\np ≈ {best:.2f}", color=GREEN, fontsize=9,
        fontweight="bold")
ax.set_title("05 · Dropout rate (p)", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("dropout probability p")
ax.set_ylabel("error")
ax.set_ylim(0, 0.8)
ax.legend(loc="upper left", fontsize=8)
annotate(ax, "random zeroing during train\nforces redundancy\ntypical: 0.2–0.5", loc=(0.55, 0.25))

# ═════════════════════════════════════════════════════════════════
# 06 · Loss (cost) functions
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 2])
x = np.linspace(-3, 3, 200)
mse = x ** 2
mae = np.abs(x)
huber = np.where(np.abs(x) < 1, 0.5 * x ** 2, np.abs(x) - 0.5)

ax.plot(x, mse, color=BLUE, lw=2, label="MSE")
ax.plot(x, mae, color=ORANGE, lw=2, label="MAE (robust)")
ax.plot(x, huber, color=GREEN, lw=2, label="Huber (mix)")
ax.set_title("06 · Regression loss functions", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("residual  ŷ − y")
ax.set_ylabel("loss")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 6)
ax.legend(loc="upper center", fontsize=8)
annotate(ax,
         "MSE: smooth, outlier-sensitive\n"
         "MAE: robust, non-smooth\n"
         "Classification losses (CE, hinge)\n"
         "live on p̂ / margin axes — see ch15",
         loc=(0.02, 0.98))

# ═════════════════════════════════════════════════════════════════
# 07 · Layer types (schematic cards)
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[2, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("07 · Layer types", fontsize=13, fontweight="bold", color=DARK, pad=8)

cards = [
    ("Dense / FC",      "tabular, last layers",      BLUE,   8.5),
    ("Conv2D",          "images, local patterns",    ORANGE, 7.0),
    ("RNN / LSTM / GRU","sequences, time",           GREEN,  5.5),
    ("Attention",       "long-range dep., tokens",   PURPLE, 4.0),
    ("BatchNorm",       "stabilise deep nets",       TEAL,   2.5),
    ("Dropout",         "regularise",                RED,    1.0),
]
for name, use, col, y in cards:
    ax.add_patch(plt.Rectangle((0.3, y - 0.55), 3.2, 1.1, facecolor=col,
                               edgecolor="white", alpha=0.85))
    ax.text(1.9, y, name, ha="center", va="center", color="white",
            fontweight="bold", fontsize=10)
    ax.text(3.7, y, "→  " + use, ha="left", va="center", color=DARK, fontsize=10)
annotate(ax, "pick the layer that matches\nyour data's structure", loc=(0.02, 0.12))

# ═════════════════════════════════════════════════════════════════
# 08 · Number of layers (depth)
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[2, 1])
depth = np.arange(1, 21)
# underfits shallow, sweet spot middle, overfits / vanishing at high depth w/o tricks
train_loss = 0.9 * np.exp(-depth / 4) + 0.05
val_loss = 0.9 * np.exp(-depth / 4) + 0.05 + 0.015 * (depth - 6) ** 2 * (depth > 6)
params = depth ** 1.8 * 20   # scaled

ax.plot(depth, train_loss, color=BLUE, lw=2.2, label="train loss")
ax.plot(depth, val_loss, color=RED, lw=2.2, label="val loss")
ax2 = ax.twinx()
ax2.plot(depth, params, color=GREY, lw=1.5, ls="--", label="params (right)")
ax2.set_ylabel("# params (scaled)")
best_d = depth[np.argmin(val_loss)]
ax.axvline(best_d, color=GREEN, ls="--", lw=1.5)
ax.text(best_d + 0.3, 0.5, f"best\ndepth ≈ {best_d}",
        color=GREEN, fontsize=9, fontweight="bold")
ax.set_title("08 · Number of layers (depth)", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("# hidden layers")
ax.set_ylabel("loss")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=8)
annotate(ax, "shallow → underfits\ndeep → overfits / vanishing\n(fix: residuals, BN)",
         loc=(0.25, 0.98))

# ═════════════════════════════════════════════════════════════════
# 09 · When to get more data (learning curves)
# ═════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[2, 2])
n = np.linspace(100, 10000, 80)

# high-variance model: val loss keeps improving with n → GET MORE DATA
val_hv = 0.5 / np.sqrt(n / 100) + 0.12
train_hv = np.full_like(n, 0.05)

# high-bias model: gap is tiny, both plateau high → MORE DATA WON'T HELP
val_hb = 0.45 + 0.0 * n
train_hb = 0.42 + 0.0 * n

ax.plot(n, train_hv, color=BLUE, lw=2, label="train (high var.)")
ax.plot(n, val_hv, color=RED, lw=2, label="val (high var.)")
ax.plot(n, train_hb, color=BLUE, lw=2, ls="--", label="train (high bias)")
ax.plot(n, val_hb, color=RED, lw=2, ls="--", label="val (high bias)")

ax.fill_between(n, train_hv, val_hv, color=RED, alpha=0.12)
ax.annotate("big gap +\nstill narrowing\n→ MORE DATA HELPS",
            xy=(6000, 0.2), xytext=(3500, 0.7),
            fontsize=8, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
ax.annotate("flat +\nsmall gap\n→ MORE DATA WON'T HELP\n(fix bias: bigger model,\nbetter features)",
            xy=(7000, 0.44), xytext=(3500, 0.05),
            fontsize=8, color=PURPLE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))

ax.set_title("09 · When to get more data", fontsize=13, fontweight="bold", color=DARK, pad=8)
ax.set_xlabel("# training samples")
ax.set_ylabel("loss")
ax.set_ylim(0, 0.9)
ax.legend(loc="upper right", fontsize=7)

# ── footer caption ───────────────────────────────────────────────
fig.text(0.5, 0.012,
         "Tune in this rough order: learning rate  →  batch size  →  optimiser  →  "
         "initialiser  →  architecture (layers/units)  →  regularisation (dropout, weight decay)  →  "
         "loss choice  →  more data.",
         ha="center", va="bottom", fontsize=11, color="#555", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9F9",
                   edgecolor="#BDC3C7", lw=0.8))

out = r"c:\repos\AI learning\ai-portfolio\notes\ML\ch19-hyperparameter-tuning\img\ch19 hyperparameter tuning.png"
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
