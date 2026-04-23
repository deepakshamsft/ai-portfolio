"""
gen_ch05_weight_shrinkage.py
Generates: ../img/ch05-weight-shrinkage.png

Side-by-side grouped bar chart comparing OLS, Ridge, Lasso, and Elastic Net
weights for 7 representative features from the 44-feature California Housing
polynomial expansion. Shows how each regularizer shrinks/zeroes weights.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-weight-shrinkage.png")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

X_poly = poly.fit_transform(X_train)
X_scaled = scaler.fit_transform(X_poly)
feature_names = list(poly.get_feature_names_out(data.feature_names))

# ── Fit models ────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression

models = {
    "OLS":         LinearRegression().fit(X_scaled, y_train),
    "Ridge α=1":   Ridge(alpha=1.0).fit(X_scaled, y_train),
    "Lasso α=0.001": Lasso(alpha=0.001, max_iter=20000).fit(X_scaled, y_train),
    "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000).fit(X_scaled, y_train),
}

# ── Track features (use original 8 features only, always present at indices 0-7) ─
# The degree-2 polynomial of 8 features produces 44; the first 8 are the originals.
ORIG_FEATURES = data.feature_names  # ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
tracked_idx = list(range(len(ORIG_FEATURES)))  # always valid
tracked_labels = list(ORIG_FEATURES)

# Extract weights for tracked features
coefs = {}
for mname, model in models.items():
    coefs[mname] = np.array([model.coef_[i] for i in tracked_idx])

# ── Plot ──────────────────────────────────────────────────────────────────────
n_features = len(tracked_labels)
n_models = len(models)
x = np.arange(n_features)
bar_w = 0.18
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

MODEL_COLORS = {
    "OLS":           "#94a3b8",
    "Ridge α=1":     "#60a5fa",
    "Lasso α=0.001": "#f87171",
    "Elastic Net":   "#a78bfa",
}

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

for i, (mname, offset) in enumerate(zip(models.keys(), offsets)):
    vals = coefs[mname]
    bars = ax.bar(x + offset, vals, width=bar_w, label=mname,
                  color=MODEL_COLORS[mname], alpha=0.85, zorder=3)
    # Mark exact zeros
    for j, v in enumerate(vals):
        if abs(v) < 1e-6:
            ax.text(x[j] + offset, 0.02, "0", ha="center", va="bottom",
                    color=MODEL_COLORS[mname], fontsize=7, fontweight="bold")

ax.axhline(0, color="white", alpha=0.3, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(tracked_labels, rotation=18, ha="right", color="white", fontsize=9)
ax.set_ylabel("Coefficient (standardized)", color="white", fontsize=11)
ax.set_title("Weight Shrinkage: OLS vs Ridge vs Lasso vs Elastic Net\n"
             "7 representative features from 44-feature California Housing expansion",
             color="white", fontsize=12, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#4a4a6a")

legend = ax.legend(loc="upper right", fontsize=10, framealpha=0.3,
                   labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e")
plt.close()
print(f"Saved: {OUT_PATH}")
