#!/usr/bin/env python3
"""
Generate "Weight Divergence Animation" — shows how weights diverge dramatically
with polynomial powers when fitting data with different true relationships.

Demonstrates: The volume knob concept - model cranks up helpful shapes, silences others.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_weight_divergence_animation():
    """Create animation showing weight evolution during training."""
    
    # Generate data: y = 2x²
    np.random.seed(42)
    x_train = np.linspace(-2, 2, 50).reshape(-1, 1)
    y_train = 2 * x_train.ravel()**2 + np.random.normal(0, 0.5, len(x_train))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Data and fit
    ax1.scatter(x_train, y_train, color='#1e3a8a', alpha=0.6, s=30, label='Data (y = 2x²)')
    line, = ax1.plot([], [], 'r-', linewidth=2.5, label='Model fit')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-1, 10)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Model Fit Over Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Weight bars
    feature_names = ['x', 'x²', 'x³']
    x_pos = np.arange(len(feature_names))
    bars = ax2.bar(x_pos, [0, 0, 0], color=['#3b82f6', '#ef4444', '#8b5cf6'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.set_title('Weight Evolution ("Volume Knobs")', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, fontsize=12)
    ax2.set_ylim(-0.5, 3)
    ax2.axhline(y=0, color='k', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add status text
    status_text = ax2.text(0.5, 0.95, '', transform=ax2.transAxes,
                          fontsize=11, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Simulate training progression
    frames = 80
    x_smooth = np.linspace(-2, 2, 100).reshape(-1, 1)
    
    # Weight trajectories (simulated convergence)
    # Target: w1≈0, w2≈2.0, w3≈0
    w1_trajectory = np.concatenate([
        np.linspace(1.0, 0.8, 20),
        np.linspace(0.8, 0.2, 30),
        np.linspace(0.2, 0.05, 30)
    ])
    w2_trajectory = np.concatenate([
        np.linspace(1.0, 1.2, 20),
        np.linspace(1.2, 1.8, 30),
        np.linspace(1.8, 2.0, 30)
    ])
    w3_trajectory = np.concatenate([
        np.linspace(1.0, 0.6, 20),
        np.linspace(0.6, 0.1, 30),
        np.linspace(0.1, 0.001, 30)
    ])
    
    def init():
        line.set_data([], [])
        for bar in bars:
            bar.set_height(0)
        status_text.set_text('')
        return [line] + list(bars) + [status_text]
    
    def animate(frame):
        w1 = w1_trajectory[frame]
        w2 = w2_trajectory[frame]
        w3 = w3_trajectory[frame]
        
        # Update fit line
        y_pred = w1 * x_smooth.ravel() + w2 * x_smooth.ravel()**2 + w3 * x_smooth.ravel()**3
        line.set_data(x_smooth.ravel(), y_pred)
        
        # Update weight bars
        bars[0].set_height(w1)  # x
        bars[1].set_height(w2)  # x²
        bars[2].set_height(w3)  # x³
        
        # Status message
        if frame < 20:
            status = "⚙️ Early training\n(all weights similar)"
        elif frame < 50:
            status = "🎚️ Adjusting...\n(x² rising, others falling)"
        else:
            status = "✅ Converged!\n🔊 x² amplified\n🔇 x, x³ silenced"
        
        status_text.set_text(status)
        
        return [line] + list(bars) + [status_text]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=frames, interval=80,
                                  blit=True, repeat=True)
    
    # Save as GIF
    output_path = IMG_DIR / "ch04-weight-divergence.gif"
    print(f"Saving weight divergence animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=12, dpi=100)
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Weight Divergence Animation...")
    create_weight_divergence_animation()
    print("Done!")
