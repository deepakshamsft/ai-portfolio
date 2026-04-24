"""gen_ch04_degree_sweep.py — Ch.4 degree sweep: train vs test MAE for degrees 1–8."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

BG = "#1a1a2e"
ACCENT_TRAIN = "#38bdf8"   # sky-blue — train MAE
ACCENT_TEST  = "#f97316"   # orange   — test MAE
TARGET_LINE  = "#22c55e"   # green    — $40k target

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = list(range(1, 6))   # degrees 1–5; degree 6+ creates 3000+ features and is impractical
train_maes, test_maes = [], []

for deg in degrees:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    train_maes.append(mean_absolute_error(y_train, pipe.predict(X_train)) * 100_000)
    test_maes.append( mean_absolute_error(y_test,  pipe.predict(X_test))  * 100_000)

Y_MAX = 100_000  # cap axis; annotate values that exceed this

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

# clamp values for plotting
train_plot = [min(v, Y_MAX * 0.95) for v in train_maes]
test_plot  = [min(v, Y_MAX * 0.95) for v in test_maes]

ax.plot(degrees, train_plot, "o-", color=ACCENT_TRAIN, linewidth=2.5,
        markersize=7, label="Train MAE")
ax.plot(degrees, test_plot,  "s-", color=ACCENT_TEST,  linewidth=2.5,
        markersize=7, label="Test MAE")
ax.axhline(40_000, color=TARGET_LINE, linestyle="--", linewidth=1.5,
           label="$40k target")

# annotate clipped test values
for deg, raw, clipped in zip(degrees, test_maes, test_plot):
    if raw > Y_MAX:
        ax.annotate(f"${raw/1000:.0f}k ↑",
                    xy=(deg, Y_MAX * 0.90),
                    ha="center", va="bottom", color=ACCENT_TEST, fontsize=7.5)

# annotate sweet spot
best_deg = int(np.argmin(test_maes)) + 1
ax.annotate(
    f"Sweet spot\ndeg={best_deg}",
    xy=(best_deg, test_maes[best_deg - 1]),
    xytext=(best_deg + 0.5, test_maes[best_deg - 1] + 8_000),
    color=TARGET_LINE, fontsize=9,
    arrowprops=dict(arrowstyle="->", color=TARGET_LINE),
)

ax.set_xlabel("Polynomial Degree", color="#e2e8f0", fontsize=11)
ax.set_ylabel("MAE ($)", color="#e2e8f0", fontsize=11)
ax.set_title("Ch.4 — Bias-Variance Trade-off: Degree Sweep", color="#e2e8f0",
             fontsize=13, pad=14)
ax.set_ylim(0, Y_MAX)
ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0",
          fontsize=10)
ax.tick_params(colors="#94a3b8")
for spine in ax.spines.values():
    spine.set_edgecolor("#334155")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"${v/1000:.0f}k"
))

plt.tight_layout()
plt.savefig("../img/ch04-degree-sweep.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Saved ch04-degree-sweep.png")
