"""
Ridge Shrinkage Animation
Shows how ALL weights gradually shrink as λ increases,
with important features resisting more than noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from pathlib import Path
import os

# Get absolute path to img directory
script_dir = Path(__file__).parent
img_dir = script_dir.parent / 'img'
img_dir.mkdir(exist_ok=True)

# Simulate feature importance and regularization path
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

# Initial weights (OLS, λ=0) - realistic magnitudes
ols_weights = np.array([0.68, -0.42, 0.31, -0.28, -0.18, 0.21])

# Lambda values (log scale)
lambdas = np.logspace(-3, 2, 80)  # 0.001 to 100

# Compute Ridge path: w_ridge = w_ols / (1 + λ)
# More accurately: simulate w(λ) = w_OLS / (1 + λ/eigenvalue)
# For visualization, use importance-based decay rates
importance = np.array([1.0, 0.7, 0.6, 0.5, 0.2, 0.15])  # How much each feature resists shrinkage

ridge_paths = []
for lam in lambdas:
    # Ridge shrinks inversely proportional to importance
    weights = ols_weights / (1 + lam / importance)
    ridge_paths.append(weights)

ridge_paths = np.array(ridge_paths)  # Shape: (80, 6)

# Create animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ridge Regularization: Reeling In Unimportant Features', 
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Weight trajectories
colors = plt.cm.tab10(np.linspace(0, 0.9, len(features)))

lines = []
for i, (feat, color) in enumerate(zip(features, colors)):
    line, = ax1.plot([], [], '-', linewidth=2.5, color=color, 
                     label=feat, alpha=0.85)
    lines.append(line)

ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax1.set_xlim(np.log10(lambdas[0]), np.log10(lambdas[-1]))
ax1.set_ylim(-0.75, 0.75)
ax1.set_xlabel('log₁₀(λ)  [regularization strength]', fontsize=11)
ax1.set_ylabel('Weight Value', fontsize=11)
ax1.set_title('All Weights Shrink (None Reach Zero)', fontsize=12)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add shading for weak λ (overfitting) vs strong λ (underfitting)
ax1.axvspan(np.log10(lambdas[0]), -1, alpha=0.1, color='red', label='_nolegend_')
ax1.axvspan(1, np.log10(lambdas[-1]), alpha=0.1, color='blue', label='_nolegend_')
ax1.text(-2.5, 0.65, '← Weak penalty\n(overfitting risk)', fontsize=8, alpha=0.7)
ax1.text(1.2, 0.65, 'Strong penalty →\n(underfitting)', fontsize=8, alpha=0.7)

# Panel 2: Current weights bar chart
bars = ax2.barh(features, ols_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax2.set_xlim(-0.75, 0.75)
ax2.set_xlabel('Weight Value', fontsize=11)
ax2.set_title('Current Weights at λ = 0.001', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

# Lambda indicator text
lambda_text = fig.text(0.5, 0.01, 'λ = 0.001 (no regularization)', 
                       ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

def init():
    for line in lines:
        line.set_data([], [])
    return lines + [lambda_text]

def update(frame):
    lam = lambdas[frame]
    log_lam = np.log10(lam)
    
    # Update weight trajectories (show path up to current frame)
    for i, line in enumerate(lines):
        line.set_data(np.log10(lambdas[:frame+1]), ridge_paths[:frame+1, i])
    
    # Update bar chart with current weights
    current_weights = ridge_paths[frame]
    for bar, w, color in zip(bars, current_weights, colors):
        bar.set_width(w)
        bar.set_color(color)
        bar.set_alpha(0.7 if abs(w) > 0.05 else 0.3)  # Fade small weights
    
    # Update title
    ax2.set_title(f'Current Weights at λ = {lam:.3f}', fontsize=12)
    
    # Update lambda indicator
    if lam < 0.1:
        regime = "(weak penalty, overfitting risk)"
    elif lam < 10:
        regime = "(balanced)"
    else:
        regime = "(strong penalty, underfitting)"
    lambda_text.set_text(f'λ = {lam:.3f} {regime}')
    
    return lines + list(bars) + [lambda_text]

# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(lambdas), 
                               init_func=init, interval=80, blit=False)

# Save
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_path = img_dir / 'ch05-ridge-shrinkage.gif'
anim.save(str(output_path), writer=PillowWriter(fps=12))
print(f"✅ Saved: {output_path}")
plt.close()
