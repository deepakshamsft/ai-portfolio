import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path

DARK_BG = "#1a1a2e"
TEXT_COLOR = "#e2e8f0"
COLORS = [
    "#e94560", "#0f3460", "#16213e", "#533483",
    "#e2e8f0", "#f5a623", "#7ed321", "#bd10e0",
]


def gen():
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    alphas, coefs, _ = lasso_path(X, y, eps=5e-4, n_alphas=100)
    log_alphas = np.log10(alphas)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    feature_names = data.feature_names
    for i, (name, c) in enumerate(zip(feature_names, COLORS)):
        ax.plot(log_alphas, coefs[i], color=c, linewidth=2, label=name)

    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_xlabel("log\u2081\u2080(\u03b1)  [\u2190 stronger regularisation]", color=TEXT_COLOR)
    ax.set_ylabel("Coefficient value", color=TEXT_COLOR)
    ax.set_title(
        "Lasso Coefficient Path \u2014 Features Shrink to Zero as \u03bb Increases",
        color=TEXT_COLOR,
    )
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR, framealpha=0.3)

    plt.tight_layout()
    plt.savefig("../img/ch03-lasso-path.png", dpi=120, facecolor=DARK_BG)
    plt.close()
    print("Saved img/ch03-lasso-path.png")


if __name__ == "__main__":
    gen()
