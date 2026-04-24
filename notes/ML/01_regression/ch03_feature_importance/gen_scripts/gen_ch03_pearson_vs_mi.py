import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

np.random.seed(42)
DARK_BG = "#1a1a2e"
ACCENT1 = "#e94560"
ACCENT2 = "#16213e"
TEXT_COLOR = "#e2e8f0"


def gen():
    n = 200
    x1 = np.linspace(-3, 3, n)
    y1 = 0.8 * x1 + np.random.randn(n) * 0.6   # linear
    y2 = x1**2 + np.random.randn(n) * 0.5        # U-shaped

    rho1 = np.corrcoef(x1, y1)[0, 1]
    rho2 = np.corrcoef(x1, y2)[0, 1]
    mi1 = mutual_info_regression(x1.reshape(-1, 1), y1, random_state=42)[0]
    mi2 = mutual_info_regression(x1.reshape(-1, 1), y2, random_state=42)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(TEXT_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].scatter(x1, y1, c=ACCENT1, alpha=0.5, s=15)
    axes[0].set_title(f"Linear relationship\n\u03c1 = {rho1:.2f}, MI = {mi1:.2f}", color=TEXT_COLOR)
    axes[0].set_xlabel("x", color=TEXT_COLOR)
    axes[0].set_ylabel("y", color=TEXT_COLOR)

    axes[1].scatter(x1, y2, c="#e2e8f0", alpha=0.5, s=15)
    axes[1].set_title(f"U-shaped (non-linear)\n\u03c1 \u2248 {rho2:.2f}, MI = {mi2:.2f}", color=TEXT_COLOR)
    axes[1].set_xlabel("x", color=TEXT_COLOR)

    plt.suptitle("Pearson Misses Non-Linear Associations; MI Catches Them",
                 color=TEXT_COLOR, fontsize=11)
    plt.tight_layout()
    plt.savefig("../img/ch03-pearson-vs-mi.png", dpi=120, facecolor=DARK_BG)
    plt.close()
    print("Saved img/ch03-pearson-vs-mi.png")


if __name__ == "__main__":
    gen()
