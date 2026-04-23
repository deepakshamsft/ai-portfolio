"""
gen_ch05_l1_l2_geometry.py
Generates: ../img/ch05-l1-l2-geometry.png

Side-by-side L2 (Ridge) and L1 (Lasso) constraint regions with MSE contours.
Uses the exact 2D worked example from §3.5:
  OLS optimum = (2.0, 0.5),  budget t = 1.0
  Lasso solution = (1.0, 0.0)
  Ridge solution = (0.970, 0.243)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-l1-l2-geometry.png")

W_OLS = np.array([2.0, 0.5])
W_RIDGE = np.array([0.970, 0.243])
W_LASSO = np.array([1.0, 0.0])
T = 1.0

XLIM = (-1.5, 2.5)
YLIM = (-1.5, 1.5)


def mse_contour(w1g, w2g):
    """Spherical MSE centered at OLS optimum (matches §3.5 example)."""
    return (w1g - W_OLS[0])**2 + (w2g - W_OLS[1])**2


fig, (ax_l2, ax_l1) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")

for ax in (ax_l2, ax_l1):
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhline(0, color="white", alpha=0.2, linewidth=0.8)
    ax.axvline(0, color="white", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("$w_1$", color="white", fontsize=12)
    ax.set_ylabel("$w_2$", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4a6a")

    # MSE contours
    w1g = np.linspace(*XLIM, 300)
    w2g = np.linspace(*YLIM, 300)
    W1, W2 = np.meshgrid(w1g, w2g)
    Z = mse_contour(W1, W2)
    ax.contour(W1, W2, Z, levels=[0.5, 1.0, 1.5, 2.5],
               colors="gray", linestyles="--", linewidths=0.9, alpha=0.6)

    # OLS optimum star
    ax.plot(*W_OLS, marker="*", markersize=14, color="white", zorder=6,
            label="OLS optimum (2.0, 0.5)")
    ax.annotate("OLS optimum", W_OLS, xytext=(W_OLS[0] + 0.08, W_OLS[1] + 0.1),
                color="white", fontsize=9)

# ── Left panel: L2 (Ridge) ────────────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 400)
cx, cy = np.cos(theta) * T, np.sin(theta) * T
ax_l2.fill(cx, cy, color="#1e3a8a", alpha=0.5, zorder=2)
ax_l2.plot(cx, cy, color="#60a5fa", linewidth=2, zorder=3)

ax_l2.plot(*W_RIDGE, "o", markersize=11, color="#60a5fa", zorder=7,
           label=f"Ridge solution ({W_RIDGE[0]:.3f}, {W_RIDGE[1]:.3f})")
ax_l2.annotate(f"Ridge solution\n({W_RIDGE[0]:.3f}, {W_RIDGE[1]:.3f})",
               W_RIDGE, xytext=(W_RIDGE[0] - 0.9, W_RIDGE[1] - 0.4),
               color="#60a5fa", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=1.2))
ax_l2.text(0.0, -0.5, "Neither $w_j = 0$", color="#60a5fa", fontsize=10,
           ha="center", style="italic")
ax_l2.set_title(r"L2 (Ridge): $w_1^2 + w_2^2 \leq 1$",
                color="white", fontsize=12, pad=10)
ax_l2.legend(loc="upper left", fontsize=8, framealpha=0.3,
             labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

# ── Right panel: L1 (Lasso) ───────────────────────────────────────────────────
diamond_x = [T, 0, -T, 0, T]
diamond_y = [0, T, 0, -T, 0]
ax_l1.fill(diamond_x, diamond_y, color="#14532d", alpha=0.55, zorder=2)
ax_l1.plot(diamond_x, diamond_y, color="#4ade80", linewidth=2, zorder=3)

ax_l1.plot(*W_LASSO, "o", markersize=11, color="#4ade80", zorder=7,
           label=f"Lasso solution ({W_LASSO[0]:.1f}, {W_LASSO[1]:.1f})")
ax_l1.annotate(f"Lasso solution\n({W_LASSO[0]:.1f}, {W_LASSO[1]:.1f})",
               W_LASSO, xytext=(W_LASSO[0] - 0.9, W_LASSO[1] + 0.35),
               color="#4ade80", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.2))
ax_l1.text(0.0, -0.55, "$w_2 = 0$ ✓  (corner hit)", color="#4ade80", fontsize=10,
           ha="center", style="italic")
ax_l1.set_title(r"L1 (Lasso): $|w_1| + |w_2| \leq 1$",
                color="white", fontsize=12, pad=10)
ax_l1.legend(loc="upper left", fontsize=8, framealpha=0.3,
             labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

fig.suptitle("Why L1 Produces Exact Zeros: The Geometry",
             color="white", fontsize=14, y=1.01)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
