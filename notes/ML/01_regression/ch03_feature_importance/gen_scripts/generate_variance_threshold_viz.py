"""
Gen script: variance-threshold-comparison.png
Static side-by-side histogram comparing:
- Left: Near-constant feature (tall spike, barely any spread)
- Right: MedInc feature (broad distribution)
Output: ../img/variance-threshold-comparison.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

HERE = Path(__file__).parent
OUT = HERE.parent / "img" / "variance-threshold-comparison.png"

# Color palette (matching existing conventions)
BG = "#1a1a2e"
PANEL_BG = "#12122a"
LABEL_CLR = "#e2e8f0"
RED = "#dc2626"
GREEN = "#16a34a"
BLUE = "#2563eb"
GREY = "#64748b"

# Load California Housing data
data = fetch_california_housing()
X = data.data
feature_names = data.feature_names

# Create a near-constant feature (synthetic)
np.random.seed(42)
n_samples = X.shape[0]
near_constant = np.full(n_samples, 2.0) + np.random.normal(0, 0.01, n_samples)

# Get MedInc feature (first column)
medinc = X[:, 0]

# Compute variances
var_constant = np.var(near_constant)
var_medinc = np.var(medinc)

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

# Left panel: Near-constant feature
ax_left = axes[0]
ax_left.set_facecolor(PANEL_BG)
ax_left.hist(near_constant, bins=50, color=RED, alpha=0.8, edgecolor=LABEL_CLR, linewidth=0.5)
ax_left.set_xlabel("Feature Value", fontsize=11, color=LABEL_CLR, fontweight="bold")
ax_left.set_ylabel("Frequency", fontsize=11, color=LABEL_CLR, fontweight="bold")
ax_left.set_title("Near-Constant Feature", fontsize=13, color=RED, fontweight="bold", pad=12)
ax_left.tick_params(colors=LABEL_CLR, labelsize=9)
ax_left.grid(axis='y', alpha=0.2, color=GREY)

# Annotate variance on left
ax_left.text(0.95, 0.95, f"Variance = {var_constant:.4f}",
             transform=ax_left.transAxes, fontsize=11, color=RED,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.5", fc="#2d2d4e", ec=RED, linewidth=2, alpha=0.95),
             fontweight="bold")

# Add annotation arrow pointing to spike
ax_left.annotate("Tall spike\nat 2.00", xy=(2.0, 4000), xytext=(1.95, 6000),
                arrowprops=dict(arrowstyle="->", color=LABEL_CLR, lw=1.5),
                fontsize=9, color=LABEL_CLR,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, alpha=0.8))

# Right panel: MedInc feature
ax_right = axes[1]
ax_right.set_facecolor(PANEL_BG)
ax_right.hist(medinc, bins=50, color=GREEN, alpha=0.8, edgecolor=LABEL_CLR, linewidth=0.5)
ax_right.set_xlabel("Median Income (× $10k)", fontsize=11, color=LABEL_CLR, fontweight="bold")
ax_right.set_ylabel("Frequency", fontsize=11, color=LABEL_CLR, fontweight="bold")
ax_right.set_title("MedInc — Varied Distribution", fontsize=13, color=GREEN, fontweight="bold", pad=12)
ax_right.tick_params(colors=LABEL_CLR, labelsize=9)
ax_right.grid(axis='y', alpha=0.2, color=GREY)

# Annotate variance on right
ax_right.text(0.95, 0.95, f"Variance = {var_medinc:.1f}",
              transform=ax_right.transAxes, fontsize=11, color=GREEN,
              verticalalignment="top", horizontalalignment="right",
              bbox=dict(boxstyle="round,pad=0.5", fc="#2d2d4e", ec=GREEN, linewidth=2, alpha=0.95),
              fontweight="bold")

# Add annotation arrow pointing to spread
ax_right.annotate("Broad\nspread", xy=(6.0, 2000), xytext=(8.0, 3500),
                 arrowprops=dict(arrowstyle="->", color=LABEL_CLR, lw=1.5),
                 fontsize=9, color=LABEL_CLR,
                 bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, alpha=0.8))

# Add main caption at bottom
caption = (
    "Variance Threshold Filtering: Drop features with Var < 0.01 to remove near-constants.\n"
    "Near-constant features carry almost no information → safe to remove before training."
)
fig.text(0.5, -0.05, caption, ha="center", fontsize=10, color=LABEL_CLR,
         wrap=True, bbox=dict(boxstyle="round,pad=0.5", fc="#2d2d4e", alpha=0.9))

# Save
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print(f"✓ Generated: {OUT}")
