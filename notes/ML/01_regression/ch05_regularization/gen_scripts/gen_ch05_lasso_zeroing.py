"""
Lasso Zeroing Animation
Shows how weights get "reeled in" to zero at different λ thresholds.
Unimportant features hit zero first, important features persist.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from pathlib import Path

# Get absolute path to img directory
script_dir = Path(__file__).parent
img_dir = script_dir.parent / 'img'
img_dir.mkdir(exist_ok=True)
np.random.seed(42)

# 6 features with different importance levels
features = [
    'MedInc',
    'Latitude', 
    'MedInc²',
    'Lat × MedInc',
    'AveBedrms²',
    'Pop × AveOccup'
]

# Initial weights (OLS, λ=0)
ols_weights = np.array([0.68, -0.42, 0.31, -0.28, -0.18, 0.21])

# Lambda values (log scale) - extended range to show all zeros
lambdas = np.logspace(-3, 1.5, 100)  # 0.001 to ~31.6

# Simulate Lasso path - features zero out at different thresholds
# Each feature has a "zero threshold" based on its importance
zero_thresholds = np.array([10.0, 1.5, 0.8, 0.5, 0.08, 0.06])  # λ where feature zeros

lasso_paths = []
for lam in lambdas:
    weights = []
    for w_ols, thresh in zip(ols_weights, zero_thresholds):
        if lam >= thresh:
            # Feature zeroed out (hard threshold)
            weights.append(0.0)
        else:
            # Soft thresholding: |w| = max(0, |w_ols| - λ) * sign(w_ols)
            # Simplified: linear decay to zero as λ approaches threshold
            shrinkage = 1 - (lam / thresh) ** 0.7  # Non-linear decay
            weights.append(w_ols * shrinkage)
    lasso_paths.append(weights)

lasso_paths = np.array(lasso_paths)  # Shape: (100, 6)

# Create animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Lasso Regularization: Zeroing Out Unimportant Features', 
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Weight trajectories
colors = plt.cm.tab10(np.linspace(0, 0.9, len(features)))

lines = []
zero_markers = []  # Mark when each feature hits zero
for i, (feat, color) in enumerate(zip(features, colors)):
    line, = ax1.plot([], [], '-', linewidth=2.5, color=color, 
                     label=feat, alpha=0.85)
    lines.append(line)
    
    # Add X marker at zero threshold
    zero_lam = zero_thresholds[i]
    marker, = ax1.plot([], [], 'X', markersize=10, color=color, 
                       markeredgecolor='black', markeredgewidth=1.5)
    zero_markers.append((marker, zero_lam))

ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax1.set_xlim(np.log10(lambdas[0]), np.log10(lambdas[-1]))
ax1.set_ylim(-0.75, 0.75)
ax1.set_xlabel('log₁₀(λ)  [regularization strength]', fontsize=11)
ax1.set_ylabel('Weight Value', fontsize=11)
ax1.set_title('Weights Hit Zero at Different Thresholds', fontsize=12)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add annotations
ax1.text(-2.5, 0.65, '← All features active', fontsize=8, alpha=0.7)
ax1.text(1, 0.65, 'Only important\nfeatures remain →', fontsize=8, alpha=0.7)

# Panel 2: Current weights bar chart with "ZEROED" markers
bars = ax2.barh(features, ols_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax2.set_xlim(-0.75, 0.75)
ax2.set_xlabel('Weight Value', fontsize=11)
ax2.set_title('Current Weights at λ = 0.001', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

# Add text annotations for zeroed features
zero_texts = [ax2.text(0, i, '', ha='center', va='center', fontsize=10,
                       fontweight='bold', color='red', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='red', linewidth=2))
              for i in range(len(features))]

# Lambda indicator text
lambda_text = fig.text(0.5, 0.01, 'λ = 0.001', 
                       ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# Feature count text
feature_count_text = fig.text(0.85, 0.96, 'Active: 6/6', 
                              ha='center', fontsize=11, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

def init():
    for line in lines:
        line.set_data([], [])
    for marker, _ in zero_markers:
        marker.set_data([], [])
    return lines + [m for m, _ in zero_markers] + [lambda_text, feature_count_text]

def update(frame):
    lam = lambdas[frame]
    log_lam = np.log10(lam)
    
    # Update weight trajectories
    for i, line in enumerate(lines):
        line.set_data(np.log10(lambdas[:frame+1]), lasso_paths[:frame+1, i])
    
    # Show zero markers for features that have crossed threshold
    for marker, zero_lam in zero_markers:
        if lam >= zero_lam:
            marker.set_data([np.log10(zero_lam)], [0])
    
    # Update bar chart
    current_weights = lasso_paths[frame]
    n_active = np.sum(np.abs(current_weights) > 1e-6)
    
    for i, (bar, w, color, zero_text) in enumerate(zip(bars, current_weights, colors, zero_texts)):
        bar.set_width(w)
        
        if abs(w) < 1e-6:  # Zeroed out
            bar.set_color('lightgray')
            bar.set_alpha(0.3)
            zero_text.set_text('ZEROED')
            zero_text.set_visible(True)
        else:
            bar.set_color(color)
            bar.set_alpha(0.7)
            zero_text.set_visible(False)
    
    # Update titles
    ax2.set_title(f'Current Weights at λ = {lam:.3f}', fontsize=12)
    lambda_text.set_text(f'λ = {lam:.3f}  ({n_active}/{len(features)} features active)')
    feature_count_text.set_text(f'Active: {n_active}/{len(features)}')
    
    # Color code feature count
    if n_active >= 5:
        feature_count_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    elif n_active >= 3:
        feature_count_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    else:
        feature_count_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    return (lines + [m for m, _ in zero_markers] + list(bars) + zero_texts + 
            [lambda_text, feature_count_text])

# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(lambdas), 
                               init_func=init, interval=60, blit=False)

# Save
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_path = img_dir / 'ch05-lasso-zeroing.gif'
anim.save(str(output_path), writer=PillowWriter(fps=15))
print(f"✅ Saved: {output_path}")
plt.close()
