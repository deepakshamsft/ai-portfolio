#!/usr/bin/env python3
"""
Generate "Survival of the Fittest" static image — before/after weight comparison
showing how the model beats down unhelpful features and amplifies useful ones.

Demonstrates: The natural selection process of feature weights during training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_survival_chart():
    """Create before/after bar chart showing weight evolution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    feature_names = ['x', 'x²', 'x³', 'x⁴', 'x⁵']
    x_pos = np.arange(len(feature_names))
    
    # Before training: all weights initialized similarly (random small values)
    weights_before = [0.8, 1.0, 0.9, 1.1, 0.85]
    
    # After training on y = 2x² data: x² dominates, others silenced
    weights_after = [0.05, 2.0, 0.01, 0.001, 0.0001]
    
    colors = ['#3b82f6', '#ef4444', '#8b5cf6', '#f59e0b', '#10b981']
    
    # BEFORE subplot
    bars1 = ax1.bar(x_pos, weights_before, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Weight Value', fontsize=14, fontweight='bold')
    ax1.set_title('BEFORE Training\n(Initial: All Features Equal)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, fontsize=14)
    ax1.set_ylim(0, 2.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
                label='Starting baseline')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=11)
    
    # Add emoji annotations
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                '🎚️', ha='center', va='bottom', fontsize=20)
    
    # AFTER subplot
    bars2 = ax2.bar(x_pos, weights_after, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Weight Value', fontsize=14, fontweight='bold')
    ax2.set_title('AFTER Training\n(Final: Only x² Survives)', 
                  fontsize=16, fontweight='bold', color='#15803d')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, fontsize=14)
    ax2.set_ylim(0, 2.5)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5,
                label='Noise threshold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=11)
    
    # Add outcome annotations
    annotations = ['🔇', '🔊', '🔇', '🔇', '🔇']
    labels = ['Silenced', 'AMPLIFIED!', 'Silenced', 'Silenced', 'Silenced']
    
    for i, (bar, emoji, label) in enumerate(zip(bars2, annotations, labels)):
        height = bar.get_height()
        y_offset = 0.15 if emoji == '🔊' else 0.05
        ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                emoji, ha='center', va='bottom', fontsize=20)
        ax2.text(bar.get_x() + bar.get_width()/2., -0.15,
                label, ha='center', va='top', fontsize=9,
                fontweight='bold' if emoji == '🔊' else 'normal',
                color='#15803d' if emoji == '🔊' else '#7f1d1d')
    
    # Add main caption
    fig.text(0.5, 0.02, 
             'The model "beats down" shapes that don\'t fit the data (y = 2x²) and "cranks up" the one that does.',
             ha='center', fontsize=13, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save as PNG
    output_path = IMG_DIR / "ch04-survival-of-fittest.png"
    print(f"Saving survival chart to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Survival of the Fittest Chart...")
    create_survival_chart()
    print("Done!")
