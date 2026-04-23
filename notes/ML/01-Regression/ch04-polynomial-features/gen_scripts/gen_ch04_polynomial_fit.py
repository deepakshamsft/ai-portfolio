"""gen_ch04_polynomial_fit.py — Ch.4 income vs price scatter with degree 1/2/3 curves."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

BG = "#1a1a2e"
SCATTER_COLOR = "#64748b"   # slate — data points (low alpha)
COLORS = {1: "#ef4444", 2: "#22c55e", 3: "#f97316"}
LABELS = {1: "Degree 1 (linear)", 2: "Degree 2 (quadratic)", 3: "Degree 3 (cubic)"}

data = fetch_california_housing()
X_raw = data.data[:, 0:1]   # MedInc only (column 0)
y     = data.target          # MedHouseVal in $100k units

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

# scatter (subsample for performance)
rng   = np.random.default_rng(42)
idx   = rng.choice(len(y), size=2000, replace=False)
ax.scatter(X_raw[idx, 0], y[idx] * 100_000, s=6, alpha=0.25,
           color=SCATTER_COLOR, rasterized=True)

x_line = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)

for deg in [1, 2, 3]:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipe.fit(X_raw, y)
    y_line = pipe.predict(x_line) * 100_000
    ax.plot(x_line.ravel(), y_line, color=COLORS[deg], linewidth=2.2,
            label=LABELS[deg])

ax.set_xlabel("MedInc (×$10k)", color="#e2e8f0", fontsize=11)
ax.set_ylabel("MedHouseVal ($)", color="#e2e8f0", fontsize=11)
ax.set_title("Ch.4 — Polynomial Fits: MedInc → House Value", color="#e2e8f0",
             fontsize=13, pad=14)
ax.set_ylim(-20_000, 620_000)
ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0",
          fontsize=10)
ax.tick_params(colors="#94a3b8")
for spine in ax.spines.values():
    spine.set_edgecolor("#334155")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"${v/1000:.0f}k"
))

plt.tight_layout()
plt.savefig("../img/ch04-polynomial-fit.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Saved ch04-polynomial-fit.png")
