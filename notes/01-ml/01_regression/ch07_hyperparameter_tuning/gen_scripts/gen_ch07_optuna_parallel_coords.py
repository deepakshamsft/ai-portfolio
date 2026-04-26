"""
gen_ch07_optuna_parallel_coords.py
Generates: ../img/ch07-optuna-parallel-coords.png
Parallel coordinates plot for top-10 Optuna trials across hyperparameters.
Uses real Optuna study if available, else synthetic fallback data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

BG     = '#1a1a2e'
GRID_C = '#1e293b'
TEXT_C = '#e2e8f0'

N_TRIALS = 100

# ── Build/load trial data ────────────────────────────────────────────────
try:
    import optuna
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    import pandas as pd

    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        degree   = trial.suggest_int('degree', 1, 3)
        alpha    = trial.suggest_float('alpha', 1e-4, 10, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        pipe = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)

    df = study.trials_dataframe()
    df['mae_k'] = -df['value'] * 100_000
    df = df.rename(columns={
        'params_degree':   'degree',
        'params_alpha':    'alpha',
        'params_l1_ratio': 'l1_ratio',
    })
    top10 = df.nsmallest(10, 'mae_k')[['degree', 'alpha', 'l1_ratio', 'mae_k']].reset_index(drop=True)
    print('Used real Optuna study.')

except Exception as e:
    print(f'Optuna unavailable ({e}), using synthetic data.')
    import pandas as pd
    rng = np.random.default_rng(7)
    top10 = pd.DataFrame({
        'degree':   rng.integers(2, 4, size=10).astype(float),
        'alpha':    np.exp(rng.uniform(np.log(5e-4), np.log(0.05), size=10)),
        'l1_ratio': rng.uniform(0.1, 0.9, size=10),
        'mae_k':    rng.uniform(36.5, 39.5, size=10),
    })
    top10 = top10.sort_values('mae_k').reset_index(drop=True)

# ── Normalize each axis to [0,1] for drawing ────────────────────────────
cols   = ['degree', 'alpha', 'l1_ratio', 'mae_k']
labels = ['degree', 'log(α)', 'l1_ratio', 'MAE ($k)']

data_arr = top10[cols].values.astype(float)

# log-scale alpha for display
data_arr[:, 1] = np.log10(data_arr[:, 1])

col_min = data_arr.min(axis=0)
col_max = data_arr.max(axis=0)
# Avoid division by zero
col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
norm_arr = (data_arr - col_min) / col_range

# ── Colour by MAE (green=low, red=high) ──────────────────────────────────
cmap     = cm.RdYlGn_r
mae_norm = mcolors.Normalize(vmin=data_arr[:, 3].min(), vmax=data_arr[:, 3].max())
colors   = [cmap(mae_norm(v)) for v in data_arr[:, 3]]

n_axes = len(cols)
x_pos  = np.arange(n_axes)

fig, axes = plt.subplots(1, n_axes - 1, figsize=(10, 6), sharey=False, facecolor=BG)
fig.patch.set_facecolor(BG)

if n_axes - 1 == 1:
    axes = [axes]

# Draw axes
for k, ax in enumerate(axes):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# We draw on a single axis spanning all columns via lines
fig_ax = fig.add_subplot(1, 1, 1, facecolor=BG)
fig_ax.set_facecolor(BG)
fig_ax.set_xlim(-0.1, n_axes - 1 + 0.1)
fig_ax.set_ylim(-0.1, 1.1)
for sp in fig_ax.spines.values():
    sp.set_edgecolor(GRID_C)
fig_ax.tick_params(colors=TEXT_C, bottom=False, left=False)
fig_ax.set_xticks(x_pos)
fig_ax.set_xticklabels(labels, color=TEXT_C, fontsize=12)
fig_ax.set_yticks([])

# Draw vertical axis lines
for j in range(n_axes):
    fig_ax.axvline(j, color='#475569', linewidth=1.5, zorder=1)
    # Tick marks at min/max
    raw_min = col_min[j]
    raw_max = col_max[j]
    if j == 1:  # alpha: show log10 values
        lbl_min = f'{10**raw_min:.4f}'
        lbl_max = f'{10**raw_max:.3f}'
    elif j == 3:  # MAE
        lbl_min = f'${raw_min:.1f}k'
        lbl_max = f'${raw_max:.1f}k'
    elif j == 0:  # degree (int)
        lbl_min = f'{int(round(raw_min))}'
        lbl_max = f'{int(round(raw_max))}'
    else:
        lbl_min = f'{raw_min:.2f}'
        lbl_max = f'{raw_max:.2f}'
    fig_ax.text(j, -0.08, lbl_min, ha='center', va='top', color='#94a3b8', fontsize=8)
    fig_ax.text(j, 1.07,  lbl_max, ha='center', va='bottom', color='#94a3b8', fontsize=8)

# Draw lines for each trial
best_idx = int(np.argmin(data_arr[:, 3]))
for i, (nrow, color) in enumerate(zip(norm_arr, colors)):
    lw    = 3.5 if i == best_idx else 1.2
    alpha = 1.0 if i == best_idx else 0.7
    fig_ax.plot(x_pos, nrow, color=color, linewidth=lw, alpha=alpha, zorder=2)
    if i == best_idx:
        fig_ax.annotate(f'Best trial\nMAE=${data_arr[i,3]:.1f}k',
                        xy=(n_axes - 1, nrow[-1]),
                        xytext=(n_axes - 1.1, nrow[-1] + 0.15),
                        color=TEXT_C, fontsize=8.5,
                        arrowprops=dict(arrowstyle='->', color=TEXT_C, lw=1))

fig_ax.set_title('Top 10 Optuna Trials — Hyperparameter Configurations vs MAE',
                 color=TEXT_C, fontsize=12, fontweight='bold', pad=15)

# Colourbar
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=mae_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=fig_ax, fraction=0.025, pad=0.03)
cbar.set_label('MAE ($k)', color=TEXT_C, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=TEXT_C)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_C)
cbar.outline.set_edgecolor(GRID_C)
cbar.ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f'${v:.1f}k')
)

# Remove sub-axes (they were just placeholders)
for ax in axes:
    ax.remove()

fig.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch07-optuna-parallel-coords.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
