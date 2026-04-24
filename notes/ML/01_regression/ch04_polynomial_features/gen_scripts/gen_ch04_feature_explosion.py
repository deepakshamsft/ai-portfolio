"""gen_ch04_feature_explosion.py — Ch.4 visual showing feature count explosion."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG      = "#1a1a2e"
COLORS  = {"raw": "#1e3a8a", "deg2": "#b45309", "deg3": "#b91c1c"}
TEXT_C  = "#e2e8f0"
SUB_C   = "#94a3b8"

rows = [
    # (d, D_deg2, D_deg3)
    (2,   5,    9),
    (4,  14,   34),
    (8,  44,  164),
    (20, 230, 1540),
]

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

# header
ax.text(0.5, 0.93, "Ch.4 — Feature Explosion by Degree",
        ha="center", va="center", color=TEXT_C, fontsize=14,
        fontweight="bold", transform=ax.transAxes)

col_xs = [0.12, 0.38, 0.62, 0.86]
headers = ["Raw Features (d)", "Degree 2 features", "Degree 3 features", "Formula (deg=2)"]
for cx, h in zip(col_xs, headers):
    ax.text(cx, 0.80, h, ha="center", va="center", color=SUB_C,
            fontsize=10, transform=ax.transAxes)

for i, (d, d2, d3) in enumerate(rows):
    y_pos = 0.68 - i * 0.16

    # raw box
    rect = mpatches.FancyBboxPatch((col_xs[0]-0.06, y_pos-0.045), 0.12, 0.09,
                                    boxstyle="round,pad=0.01",
                                    facecolor=COLORS["raw"], edgecolor="none",
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(col_xs[0], y_pos, str(d), ha="center", va="center",
            color=TEXT_C, fontsize=13, fontweight="bold",
            transform=ax.transAxes)

    # degree-2 box
    rect2 = mpatches.FancyBboxPatch((col_xs[1]-0.06, y_pos-0.045), 0.12, 0.09,
                                     boxstyle="round,pad=0.01",
                                     facecolor=COLORS["deg2"], edgecolor="none",
                                     transform=ax.transAxes)
    ax.add_patch(rect2)
    ax.text(col_xs[1], y_pos, str(d2), ha="center", va="center",
            color=TEXT_C, fontsize=13, fontweight="bold",
            transform=ax.transAxes)

    # degree-3 box
    rect3 = mpatches.FancyBboxPatch((col_xs[2]-0.06, y_pos-0.045), 0.12, 0.09,
                                     boxstyle="round,pad=0.01",
                                     facecolor=COLORS["deg3"], edgecolor="none",
                                     transform=ax.transAxes)
    ax.add_patch(rect3)
    ax.text(col_xs[2], y_pos, str(d3), ha="center", va="center",
            color=TEXT_C, fontsize=13, fontweight="bold",
            transform=ax.transAxes)

    # formula annotation
    formula = f"C({d}+2,2)−1={d2}"
    ax.text(col_xs[3], y_pos, formula, ha="center", va="center",
            color=SUB_C, fontsize=9, transform=ax.transAxes)

# legend
legend_items = [
    (COLORS["raw"],  "Raw features"),
    (COLORS["deg2"], "Degree 2 expansion"),
    (COLORS["deg3"], "Degree 3 expansion"),
]
for li, (color, label) in enumerate(legend_items):
    ax.text(0.02 + li * 0.33, 0.04, "█ " + label, ha="left", va="center",
            color=color, fontsize=9, transform=ax.transAxes)

plt.tight_layout()
plt.savefig("../img/ch04-feature-explosion.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Saved ch04-feature-explosion.png")
