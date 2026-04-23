"""
Gen script: ch03-vif-instability.png
Scatter showing weight instability for high-VIF features across 50 bootstrap resamples.
Left panel: stable low-VIF features (MedInc, Latitude, Longitude).
Right panel: unstable high-VIF features (AveRooms, AveBedrms).
Output: ../img/ch03-vif-instability.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "ch03-vif-instability.png"

rng    = np.random.default_rng(42)
data   = fetch_california_housing()
X      = data.data
y      = data.target
n      = len(y)
N_BOOT = 50

weights_A = np.zeros((N_BOOT, 8))
weights_B = np.zeros((N_BOOT, 8))

for k in range(N_BOOT):
    idxA = rng.choice(n, size=n, replace=True)
    idxB = rng.choice(n, size=n, replace=True)
    XsA  = StandardScaler().fit_transform(X[idxA])
    XsB  = StandardScaler().fit_transform(X[idxB])
    weights_A[k] = LinearRegression().fit(XsA, y[idxA]).coef_
    weights_B[k] = LinearRegression().fit(XsB, y[idxB]).coef_

feat_names   = list(data.feature_names)
# One accent colour per feature (8 total)
ACCENT = [
    "#2563eb",  # MedInc
    "#16a34a",  # HouseAge
    "#d97706",  # AveRooms
    "#dc2626",  # AveBedrms
    "#9333ea",  # Population
    "#0891b2",  # AveOccup
    "#f59e0b",  # Latitude
    "#10b981",  # Longitude
]
BG        = "#1a1a2e"
LABEL_CLR = "#e2e8f0"

# Panel indices
# MedInc=0, HouseAge=1, AveRooms=2, AveBedrms=3, Population=4,
# AveOccup=5, Latitude=6, Longitude=7
stable_idx   = [0, 6, 7]   # MedInc, Latitude, Longitude
unstable_idx = [2, 3]      # AveRooms, AveBedrms

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle(
    "Weight Stability Across 50 Bootstrap Resamples\n"
    "Tight diagonal = stable  ·  Scattered cloud = VIF-inflated",
    color=LABEL_CLR, fontsize=11, y=1.01,
)

for ax, indices, panel_title in [
    (axes[0], stable_idx,   "Low-VIF Features (stable weights)"),
    (axes[1], unstable_idx, "High-VIF Features (unstable weights)"),
]:
    ax.set_facecolor(BG)
    ax.set_title(panel_title, color=LABEL_CLR, fontsize=10)

    # Reference diagonal y = x
    all_vals = np.concatenate([
        weights_A[:, indices].ravel(),
        weights_B[:, indices].ravel(),
    ])
    lo, hi = all_vals.min() - 0.05, all_vals.max() + 0.05
    ax.plot([lo, hi], [lo, hi], color="#ffffff", lw=0.8, alpha=0.4, zorder=1)

    for feat_idx in indices:
        ax.scatter(
            weights_A[:, feat_idx],
            weights_B[:, feat_idx],
            color=ACCENT[feat_idx],
            alpha=0.7, s=30, zorder=3,
            label=feat_names[feat_idx],
        )

    ax.set_xlabel("Weight — resample A", color=LABEL_CLR, fontsize=9)
    ax.set_ylabel("Weight — resample B", color=LABEL_CLR, fontsize=9)
    ax.tick_params(colors=LABEL_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d4e")
    ax.legend(facecolor="#2d2d4e", labelcolor=LABEL_CLR, fontsize=8, loc="upper left")

axes[0].text(0.05, 0.95, "Tight diagonal\n= stable weight",
             transform=axes[0].transAxes, color="#16a34a", fontsize=8, va="top")
axes[1].text(0.05, 0.95, "Scattered cloud\n= VIF-inflated",
             transform=axes[1].transAxes, color="#dc2626", fontsize=8, va="top")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
