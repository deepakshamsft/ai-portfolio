"""
Generate three Pearson/covariance animations for ch07 § 4b.

Outputs (all to ../img/):
  ch07-pearson-covariance.gif   – signed-rectangle covariance build-up
  ch07-pearson-correlation.gif  – four scatter plots at different ρ values
  ch07-covariance-matrix.gif    – 8×8 California Housing correlation matrix build

Run from repo root:
    python notes/math_under_the_hood/ch07_probability_statistics/gen_scripts/gen_ch07_pearson.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io

OUT = Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(exist_ok=True)

# ── shared palette (matches rest of ch07) ────────────────────────────────────
BG   = "#1a1a2e"
FG   = "#e2e8f0"
BLUE = "#3b82f6"
RED  = "#ef4444"
GREEN= "#22c55e"
GOLD = "#f59e0b"
GREY = "#64748b"

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  COVARIANCE BUILD-UP — signed rectangle per session
# ═══════════════════════════════════════════════════════════════════════════════
X = np.array([3., 5., 7., 4., 6.])
Y = np.array([4., 6., 8., 5., 7.])
xbar, ybar = X.mean(), Y.mean()
dx = X - xbar
dy = Y - ybar
products = dx * dy

frames_cov = []

for k in range(1, len(X) + 1):
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)

    # mean lines
    ax.axvline(xbar, color=GREY, lw=1, ls="--", alpha=0.6)
    ax.axhline(ybar, color=GREY, lw=1, ls="--", alpha=0.6)
    ax.text(xbar + 0.1, 3.3, f"$\\bar{{x}}={xbar:.0f}$",
            color=GREY, fontsize=9)
    ax.text(2.2, ybar + 0.1, f"$\\bar{{y}}={ybar:.0f}$",
            color=GREY, fontsize=9)

    # draw rectangles for sessions 1..k
    for i in range(k):
        colour = BLUE if products[i] >= 0 else RED
        rx = min(X[i], xbar)
        ry = min(Y[i], ybar)
        rw = abs(dx[i])
        rh = abs(dy[i])
        rect = mpatches.FancyArrowPatch
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.2, edgecolor=colour,
            facecolor=colour, alpha=0.25
        ))
        ax.plot(X[i], Y[i], "o", color=colour, ms=9, zorder=5)
        ax.annotate(
            f"  S{i+1}\n  Δx={dx[i]:+.0f}, Δy={dy[i]:+.0f}\n  prod={products[i]:+.0f}",
            xy=(X[i], Y[i]), xytext=(X[i] + 0.15, Y[i] - 0.05),
            color=colour, fontsize=7.5
        )

    running_cov = products[:k].mean()
    ax.set_xlim(1.5, 9)
    ax.set_ylim(2.5, 10)
    ax.set_xlabel("km run  ($x$)", color=FG)
    ax.set_ylabel("tiredness  ($y$)", color=FG)
    ax.tick_params(colors=FG)
    ax.set_title(
        f"Covariance build-up — session {k} of {len(X)}\n"
        f"Running Cov(x,y) = {running_cov:.2f}  "
        f"[blue = same direction, red = opposite]",
        color=FG, fontsize=10, pad=8
    )
    fig.tight_layout()
    frames_cov.append(fig_to_pil(fig))
    plt.close(fig)

# duplicate last frame for a pause
frames_cov += [frames_cov[-1]] * 4
frames_cov[0].save(
    OUT / "ch07-pearson-covariance.gif",
    save_all=True, append_images=frames_cov[1:],
    duration=900, loop=0
)
print("✓ ch07-pearson-covariance.gif")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FOUR SCATTER PLOTS — different ρ values
# ═══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
N = 120

def make_corr_data(r, n=N, seed=42):
    rng2 = np.random.default_rng(seed)
    cov = [[1, r], [r, 1]]
    data = rng2.multivariate_normal([0, 0], cov, n)
    return data[:, 0], data[:, 1]

configs = [
    (0.97,  "+0.97 — near-perfect positive\n(like standardised MedInc vs house value)",  BLUE),
    (0.69,  "+0.69 — strong positive\n(MedInc ↔ MedHouseVal actual ρ)",                  GREEN),
    (0.03,  "≈ 0 — no linear relationship\n(Population ↔ MedHouseVal actual ρ)",         GOLD),
    (-0.80, "−0.80 — strong negative\n(Latitude ↔ price in northern CA cheaper)",        RED),
]

frames_scatter = []
for rho, title, colour in configs:
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    xs, ys = make_corr_data(rho)
    ax.scatter(xs, ys, color=colour, alpha=0.45, s=22, edgecolors="none")
    # best-fit line
    m, b_ = np.polyfit(xs, ys, 1)
    xl = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(xl, m * xl + b_, color=colour, lw=2, alpha=0.9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.axvline(0, color=GREY, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("feature $x$ (standardised)", color=FG)
    ax.set_ylabel("target $y$ (standardised)", color=FG)
    ax.tick_params(colors=FG)
    rho_label = f"$\\rho = {rho:+.2f}$"
    ax.text(0.05, 0.93, rho_label, transform=ax.transAxes,
            fontsize=16, color=colour, fontweight="bold",
            va="top")
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    fig.tight_layout()
    # hold each panel for 1.8 s
    pil_frame = fig_to_pil(fig)
    frames_scatter += [pil_frame] * 2
    plt.close(fig)

frames_scatter[0].save(
    OUT / "ch07-pearson-correlation.gif",
    save_all=True, append_images=frames_scatter[1:],
    duration=1800, loop=0
)
print("✓ ch07-pearson-correlation.gif")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COVARIANCE MATRIX BUILD — California Housing 8×8
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    corr = df.corr().values
    labels = list(df.columns)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  (sklearn not available — covariance matrix gif skipped)")

if HAS_SKLEARN:
    p = len(labels)
    frames_mat = []

    # Build frames: reveal one column at a time (plus target col always shown)
    for reveal_up_to in range(p):
        fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
        ax.set_facecolor(BG)

        # mask unrevealed cells
        display = np.full_like(corr, np.nan)
        for col in range(reveal_up_to + 1):
            display[:, col] = corr[:, col]
            display[col, :] = corr[col, :]   # symmetric

        im = ax.imshow(display, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           color=FG, fontsize=8)
        ax.set_yticklabels(labels, color=FG, fontsize=8)

        # annotate revealed cells
        for r in range(p):
            for c in range(p):
                if not np.isnan(display[r, c]):
                    val = display[r, c]
                    txt_col = "white" if abs(val) > 0.55 else BG
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            color=txt_col, fontsize=6.5)

        ax.set_title(
            f"California Housing correlation matrix\n"
            f"(revealing column {reveal_up_to + 1}/{p}: {labels[reveal_up_to]})",
            color=FG, fontsize=10, pad=8
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=FG)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
        fig.tight_layout()
        pil_frame = fig_to_pil(fig)
        frames_mat.append(pil_frame)
        plt.close(fig)

    # pause on final frame
    frames_mat += [frames_mat[-1]] * 5
    frames_mat[0].save(
        OUT / "ch07-covariance-matrix.gif",
        save_all=True, append_images=frames_mat[1:],
        duration=700, loop=0
    )
    print("✓ ch07-covariance-matrix.gif")

print("\nAll Pearson/covariance animations written to", OUT)
