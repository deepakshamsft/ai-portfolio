"""
gen_ch07_grid_vs_random.py
Generates: ../img/ch07-grid-vs-random.png
Two side-by-side subplots comparing 9-trial grid search vs 9-trial random search
in the (alpha, degree) parameter space, colored by MAE.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# ── Data (from plan A·2) ────────────────────────────────────────────────
grid_alpha  = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
grid_degree = [1,    2,    3,    1,   2,   3,   1,   2,   3  ]
grid_mae    = [55.1, 39.8, 43.2, 54.9, 38.3, 39.1, 55.4, 38.9, 39.6]

rand_alpha  = [0.0034, 0.47, 8.3, 0.091, 0.014, 0.0031, 1.83, 0.23, 3.9]
rand_degree = [2,      1,    3,   2,     1,     2,      2,    3,    1  ]
rand_mae    = [40.5, 55.2, 42.1, 38.5, 56.0, 37.5, 39.1, 39.8, 57.0]

# ── Colour scale: green = low MAE, red = high MAE ───────────────────────
vmin, vmax = 37.0, 62.0
cmap = plt.get_cmap('RdYlGn_r')
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

BG      = '#1a1a2e'
GRID_C  = '#1e293b'
TEXT_C  = '#e2e8f0'
STAR_C  = '#fbbf24'

fig, (ax_grid, ax_rand) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.patch.set_facecolor(BG)

x_ticks = [0.001, 0.01, 0.1, 1, 10, 100]


def _style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_xscale('log')
    ax.set_xlim(0.0008, 130)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1', '2', '3'], color=TEXT_C, fontsize=11)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks], color=TEXT_C, fontsize=9)
    ax.set_xlabel('Regularization strength α', color=TEXT_C, fontsize=11)
    ax.set_ylabel('Polynomial degree', color=TEXT_C, fontsize=11)
    ax.set_title(title, color=TEXT_C, fontsize=12, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.tick_params(colors=TEXT_C)
    ax.grid(True, which='both', color=GRID_C, linewidth=0.7, linestyle='--')


# ── Left: Grid Search ───────────────────────────────────────────────────
_style_ax(ax_grid, 'Grid Search — 3 unique α values')

# grey overlay for untested α region (outside {0.01, 0.1, 1.0})
ax_grid.axvspan(0.0008, 0.009, color='#374151', alpha=0.45, zorder=0)
ax_grid.axvspan(0.011,  0.09,  color='#374151', alpha=0.45, zorder=0)
ax_grid.axvspan(0.11,   0.9,   color='#374151', alpha=0.45, zorder=0)
ax_grid.axvspan(1.1,    130,   color='#374151', alpha=0.45, zorder=0)

# Vertical lines at tested α values
for a in [0.01, 0.1, 1.0]:
    ax_grid.axvline(a, color='#94a3b8', linewidth=0.8, linestyle=':', alpha=0.6)

sc_grid = ax_grid.scatter(
    grid_alpha, grid_degree,
    c=grid_mae, cmap=cmap, norm=norm,
    s=200, edgecolors=TEXT_C, linewidths=1.2, zorder=5
)

# Best point star
best_g_idx = int(np.argmin(grid_mae))
ax_grid.scatter([grid_alpha[best_g_idx]], [grid_degree[best_g_idx]],
                marker='*', s=350, color=STAR_C, zorder=6)
ax_grid.annotate(f'Best\n${grid_mae[best_g_idx]:.1f}k',
                 xy=(grid_alpha[best_g_idx], grid_degree[best_g_idx]),
                 xytext=(0.04, 2.55), color=STAR_C, fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=STAR_C, lw=1))

# "Miss optimal!" annotation
ax_grid.annotate('← Misses optimal\nα ≈ 0.003!',
                 xy=(0.003, 2.0), xytext=(0.0009, 2.6),
                 color='#f87171', fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#f87171', lw=1))

# ── Right: Random Search ─────────────────────────────────────────────────
_style_ax(ax_rand, 'Random Search — 9 unique α values')

sc_rand = ax_rand.scatter(
    rand_alpha, rand_degree,
    c=rand_mae, cmap=cmap, norm=norm,
    s=200, edgecolors=TEXT_C, linewidths=1.2, zorder=5
)

best_r_idx = int(np.argmin(rand_mae))
ax_rand.scatter([rand_alpha[best_r_idx]], [rand_degree[best_r_idx]],
                marker='*', s=350, color=STAR_C, zorder=6)
ax_rand.annotate(f'Best\n${rand_mae[best_r_idx]:.1f}k',
                 xy=(rand_alpha[best_r_idx], rand_degree[best_r_idx]),
                 xytext=(0.006, 2.55), color=STAR_C, fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=STAR_C, lw=1))

# ── Colourbar ────────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_grid, ax_rand], fraction=0.025, pad=0.02)
cbar.set_label('CV MAE ($k)', color=TEXT_C, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=TEXT_C)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_C)
cbar.outline.set_edgecolor(GRID_C)

fig.suptitle(
    'Why Random Search Beats Grid: Same Budget, More α Coverage',
    color=TEXT_C, fontsize=13, fontweight='bold', y=1.01
)
fig.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch07-grid-vs-random.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
