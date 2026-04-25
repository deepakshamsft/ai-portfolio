"""
Generate skewness spectrum visualization showing three distribution types.

This script creates a static image with three distribution shapes side by side:
- Negative skew (left tail)
- Symmetric / normal
- Positive skew (right tail) - the problem case

Each distribution shows the relationship between mean and median.

Output: ../img/skewness-spectrum.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
OUTPUT_PATH = "../img/skewness-spectrum.png"
DPI = 150
FIGSIZE = (15, 5)
N_SAMPLES = 10000


def generate_negative_skew():
    """Generate a negatively skewed distribution (left tail)."""
    # Use beta distribution with high alpha, low beta
    data = stats.beta.rvs(8, 2, size=N_SAMPLES)
    return data


def generate_symmetric():
    """Generate a symmetric normal distribution."""
    data = np.random.normal(loc=0.5, scale=0.12, size=N_SAMPLES)
    # Clip to [0, 1] range for consistency
    data = np.clip(data, 0, 1)
    return data


def generate_positive_skew():
    """Generate a positively skewed distribution (right tail)."""
    # Use beta distribution with low alpha, high beta
    data = stats.beta.rvs(2, 8, size=N_SAMPLES)
    return data


def plot_distribution(ax, data, title, skew_type):
    """
    Plot a distribution with mean and median markers.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : array
        Distribution data
    title : str
        Title for the subplot
    skew_type : str
        Type of skew: 'negative', 'symmetric', or 'positive'
    """
    # Calculate statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    skew_val = stats.skew(data)
    
    # Set color based on skew type
    colors = {
        'negative': 'skyblue',
        'symmetric': 'lightgreen',
        'positive': 'coral'
    }
    color = colors[skew_type]
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=40, color=color, alpha=0.7, 
                                edgecolor='black', density=True)
    
    # Plot mean line
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_val:.3f}', zorder=10)
    
    # Plot median line
    ax.axvline(median_val, color='blue', linestyle='-.', linewidth=2.5, 
               label=f'Median = {median_val:.3f}', zorder=10)
    
    # Add KDE curve for smoothness
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 300)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.6)
    
    # Set title and labels
    ax.set_title(f'{title}\nSkewness = {skew_val:.2f}', 
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add annotations based on skew type
    if skew_type == 'negative':
        # Median > Mean (median is to the right)
        ax.annotate('Long left tail\npulls mean left', 
                   xy=(0.2, ax.get_ylim()[1] * 0.7), 
                   fontsize=10, color='darkblue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                            alpha=0.8, edgecolor='black'),
                   ha='center')
        # Arrow showing relationship
        mid_y = ax.get_ylim()[1] * 0.85
        ax.annotate('', xy=(mean_val, mid_y), xytext=(median_val, mid_y),
                   arrowprops=dict(arrowstyle='<-', color='purple', lw=2))
        ax.text((mean_val + median_val) / 2, mid_y + ax.get_ylim()[1] * 0.05, 
               'Mean < Median', ha='center', fontsize=9, fontweight='bold', 
               color='purple')
    
    elif skew_type == 'symmetric':
        # Mean ≈ Median
        ax.annotate('Balanced distribution\nMean ≈ Median', 
                   xy=(0.5, ax.get_ylim()[1] * 0.7), 
                   fontsize=10, color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                            alpha=0.8, edgecolor='black'),
                   ha='center')
        ax.text(mean_val, ax.get_ylim()[1] * 0.9, '✓ Ideal for\nStandardScaler', 
               ha='center', fontsize=9, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    elif skew_type == 'positive':
        # Mean > Median (mean is to the right)
        ax.annotate('Long right tail\npulls mean right', 
                   xy=(0.8, ax.get_ylim()[1] * 0.7), 
                   fontsize=10, color='darkred', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                            alpha=0.8, edgecolor='black'),
                   ha='center')
        # Arrow showing relationship
        mid_y = ax.get_ylim()[1] * 0.85
        ax.annotate('', xy=(mean_val, mid_y), xytext=(median_val, mid_y),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax.text((mean_val + median_val) / 2, mid_y + ax.get_ylim()[1] * 0.05, 
               'Mean > Median', ha='center', fontsize=9, fontweight='bold', 
               color='purple')
        # Warning box
        ax.text(median_val, ax.get_ylim()[1] * 0.9, '⚠ Problem Case\n(e.g., Population)', 
               ha='center', fontsize=9, fontweight='bold', color='darkred',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))


def main():
    """Generate the skewness spectrum visualization."""
    print("Generating skewness spectrum visualization...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate distributions
    print("  Generating negative skew distribution...")
    neg_skew_data = generate_negative_skew()
    
    print("  Generating symmetric distribution...")
    symmetric_data = generate_symmetric()
    
    print("  Generating positive skew distribution...")
    pos_skew_data = generate_positive_skew()
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    
    print("  Plotting distributions...")
    # Plot each distribution
    plot_distribution(axes[0], neg_skew_data, 'Negative Skew\n(Left Tail)', 'negative')
    plot_distribution(axes[1], symmetric_data, 'Symmetric\n(Normal)', 'symmetric')
    plot_distribution(axes[2], pos_skew_data, 'Positive Skew\n(Right Tail)', 'positive')
    
    # Overall title
    fig.suptitle('Understanding Skewness: How Distribution Shape Affects Mean vs Median', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add footer explanation
    fig.text(0.5, 0.02, 
             'Key Concept: In positively skewed distributions (like Population), a few extreme outliers pull the mean far from the median,\n'
             'causing StandardScaler to compress most data points into a narrow range. Log transformation makes the distribution symmetric.',
             ha='center', fontsize=11, style='italic', 
             bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                      edgecolor='orange', linewidth=2, alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save figure
    print(f"  Saving figure to {OUTPUT_PATH}...")
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Skewness spectrum saved to {OUTPUT_PATH}")
    print(f"  Resolution: {DPI} DPI")
    print(f"  Distributions: 3 (negative, symmetric, positive)")


if __name__ == "__main__":
    main()
