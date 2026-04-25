#!/usr/bin/env python3
"""
Generate "Bending Animation" — shows a straight line bending into a curve
as the weight for x² is gradually increased.

Demonstrates: How the x² "volume knob" transforms a straight line into a parabola.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# Generate parabolic data y = x²
np.random.seed(42)
x_data = np.linspace(-3, 3, 30)
y_data = x_data**2 + np.random.normal(0, 0.5, len(x_data))

# For plotting the fitted curve
x_smooth = np.linspace(-3, 3, 200)

def create_bending_animation():
    """Create animation showing line bending into curve."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(x_data, y_data, color='#1e3a8a', alpha=0.6, s=50, 
               label='Data (y ≈ x²)', zorder=3)
    
    # Initialize line
    line, = ax.plot([], [], 'r-', linewidth=3, label='Model fit', zorder=2)
    weight_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1, 12)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('How the x² Volume Knob Bends a Straight Line', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Animation frames: gradually increase w2 from 0 to 1.0
    frames = 60
    w2_values = np.concatenate([
        np.linspace(0, 0, 10),      # Hold at 0 (flat line)
        np.linspace(0, 1.0, 40),    # Gradual bend
        np.linspace(1.0, 1.0, 10)   # Hold at final curve
    ])
    
    def init():
        line.set_data([], [])
        weight_text.set_text('')
        return line, weight_text
    
    def animate(frame):
        w2 = w2_values[frame]
        
        # Model: ŷ = w₁·x + w₂·x²
        # Keep w1 small so the curve is dominated by x²
        w1 = 0.1
        y_smooth = w1 * x_smooth + w2 * x_smooth**2
        
        line.set_data(x_smooth, y_smooth)
        
        # Update weight display
        if w2 < 0.01:
            status = "🔇 FLAT LINE (w₂ = 0)"
        elif w2 < 0.5:
            status = "🎚️ BENDING..."
        else:
            status = "🔊 PARABOLA (w₂ = 1.0)"
        
        weight_text.set_text(
            f'Model: ŷ = 0.1·x + {w2:.2f}·x²\n'
            f'{status}'
        )
        
        return line, weight_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(w2_values), interval=100,
                                  blit=True, repeat=True)
    
    # Save as GIF
    output_path = IMG_DIR / "ch04-bending-animation.gif"
    print(f"Saving bending animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Bending Animation...")
    create_bending_animation()
    print("Done!")
