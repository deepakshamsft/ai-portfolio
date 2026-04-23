"""
gen_ch05_regularization_path.py
Generates: ../img/ch05-regularization-path.png

Regularization path — coefficient magnitudes of 7 representative features
as a function of log10(λ), using Lasso on 44-feature polynomial expansion
of California Housing data.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import lasso_path

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-regularization-path.png")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_poly)
feature_names = poly.get_feature_names_out(data.feature_names)

# ── Lasso path ────────────────────────────────────────────────────────────────
alphas_grid = np.logspace(-4, 2, 100)
alphas, coefs, _ = lasso_path(X_scaled, y, alphas=alphas_grid, max_iter=10000)
# coefs shape: (n_features, n_alphas)

# ── Named features to highlight ───────────────────────────────────────────────
TRACKED = {
    "MedInc":               "#f59e0b",
    "MedInc^2":             "#fcd34d",
    "Latitude":             "#60a5fa",
    "AveRooms AveBedrms":   "#a78bfa",
    "Population AveBedrms": "#f87171",
    "AveOccup^2":           "#f87171",
    "HouseAge AveOccup":    "#fb923c",
}

# Map tracked names to feature indices (flexible partial match)
name_list = list(feature_names)
tracked_idx = {}
for tname, color in TRACKED.items():
    for i, fn in enumerate(name_list):
        if fn == tname or fn.replace(" ", "_") == tname.replace(" ", "_"):
            tracked_idx[tname] = (i, color)
            break
    else:
        # Try case-insensitive
        for i, fn in enumerate(name_list):
            if fn.lower().replace(" ", "") == tname.lower().replace(" ", ""):
                tracked_idx[tname] = (i, color)
                break

log_alphas = np.log10(alphas)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# All features — thin background lines
for j in range(coefs.shape[0]):
    ax.plot(log_alphas, coefs[j], color="#4a4a6a", alpha=0.25, linewidth=0.7)

# Highlighted named features
for tname, (idx, color) in tracked_idx.items():
    vals = coefs[idx]
    ax.plot(log_alphas, vals, color=color, linewidth=2.2, label=tname, zorder=3)
    # Right-side annotation at the first alpha
    y_end = vals[0]
    ax.annotate(tname, xy=(log_alphas[0], y_end),
                xytext=(log_alphas[0] + 0.1, y_end),
                fontsize=7.5, color=color, va="center",
                clip_on=True)

# Zero line
ax.axhline(0, color="white", alpha=0.25, linewidth=1)

# Optimal λ vertical line (approximately log10(0.001) = -3 for Lasso on this data)
optimal_log_alpha = -3.0
ax.axvline(optimal_log_alpha, color="white", linestyle="--", linewidth=1.2,
           alpha=0.7, label="optimal λ ≈ 0.001")

ax.set_xlim(log_alphas[0], log_alphas[-1])
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("log₁₀(λ)", color="white", fontsize=12)
ax.set_ylabel("Coefficient value (standardized)", color="white", fontsize=12)
ax.set_title("Regularization Path: Feature Weights vs λ\n"
             "Lasso on 44 polynomial features (California Housing)",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#4a4a6a")

legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.3,
                   labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e")
plt.close()
print(f"Saved: {OUT_PATH}")
