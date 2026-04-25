"""
Generate before/after histogram animation showing the impact of skewness on StandardScaler.

This script creates a two-frame GIF demonstrating why log transformation helps
with positively skewed features like Population in the California Housing dataset.

Output: ../img/skew-before-after.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import imageio.v2 as imageio
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
OUTPUT_PATH = "../img/skew-before-after.gif"
DURATION = 2000  # milliseconds per frame
DPI = 100
FIGSIZE = (12, 5)


def create_frame_1():
    """Create frame showing raw Population with StandardScaler compression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    
    # Load California Housing data
    housing = fetch_california_housing()
    population = housing.data[:, 4]  # Population feature (index 4)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    population_scaled = scaler.fit_transform(population.reshape(-1, 1)).flatten()
    
    # Left subplot: Raw Population histogram
    ax1.hist(population, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Population', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Population Distribution\n(Positively Skewed)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add annotation for long tail
    ax1.annotate('Long right tail\n(few outliers at 30,000+)', 
                 xy=(25000, 50), xytext=(20000, 800),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Add annotation for bulk of data
    ax1.annotate('Bulk of data\nclustered near zero', 
                 xy=(1000, 2000), xytext=(8000, 3000),
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                 fontsize=10, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Right subplot: StandardScaler result
    ax2.hist(population_scaled, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Standardized Value (z-score)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('After StandardScaler\n(Problem: Data Compressed)', fontsize=13, fontweight='bold', color='darkred')
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight the compression zone
    ax2.axvspan(-0.5, 0.5, alpha=0.2, color='yellow', label='90% of data\ncompressed here')
    
    # Show outlier region
    outlier_max = population_scaled.max()
    ax2.annotate(f'Outlier at\n+{outlier_max:.1f}σ', 
                 xy=(outlier_max, 50), xytext=(outlier_max - 3, 800),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # Overall figure title
    fig.suptitle('FRAME 1: The Skewness Problem', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to temporary file
    temp_path_1 = "temp_frame_1.png"
    plt.savefig(temp_path_1, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return temp_path_1


def create_frame_2():
    """Create frame showing log-transformed Population with better StandardScaler result."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    
    # Load California Housing data
    housing = fetch_california_housing()
    population = housing.data[:, 4]  # Population feature
    
    # Apply log1p transformation
    population_log = np.log1p(population)
    
    # Apply StandardScaler to log-transformed data
    scaler = StandardScaler()
    population_log_scaled = scaler.fit_transform(population_log.reshape(-1, 1)).flatten()
    
    # Left subplot: log1p(Population) histogram
    ax1.hist(population_log, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('log1p(Population)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Log-Transformed Population\n(Roughly Bell-Shaped)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add annotation for better shape
    ax1.annotate('Much more symmetric!\nNo extreme outliers', 
                 xy=(7.5, 1500), xytext=(8.5, 2500),
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                 fontsize=10, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Right subplot: StandardScaler result on log-transformed data
    ax2.hist(population_log_scaled, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Standardized Value (z-score)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('After StandardScaler\n(Solution: Data Well-Distributed)', fontsize=13, fontweight='bold', color='darkgreen')
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight the well-distributed zone
    ax2.axvspan(-2, 2, alpha=0.2, color='lightgreen', label='Data spreads across\n[−2, +2] range')
    
    # Show range
    data_min, data_max = population_log_scaled.min(), population_log_scaled.max()
    ax2.annotate(f'Range: [{data_min:.1f}, {data_max:.1f}]', 
                 xy=(0, ax2.get_ylim()[1] * 0.7), 
                 fontsize=10, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                 ha='center')
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # Overall figure title
    fig.suptitle('FRAME 2: The Log Transformation Solution', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to temporary file
    temp_path_2 = "temp_frame_2.png"
    plt.savefig(temp_path_2, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return temp_path_2


def create_caption_frame():
    """Create a final caption frame explaining the key insight."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    # Main message
    caption_text = (
        "Key Insight:\n\n"
        "Skew means StandardScaler compresses your real data into a tiny band\n"
        "while one outlier sets the scale.\n\n"
        "Log transformation fixes this by making the distribution more symmetric,\n"
        "allowing StandardScaler to distribute the data evenly."
    )
    
    ax.text(0.5, 0.5, caption_text, 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                     edgecolor='orange', linewidth=3),
            linespacing=1.8)
    
    plt.tight_layout()
    
    # Save to temporary file
    temp_path_caption = "temp_frame_caption.png"
    plt.savefig(temp_path_caption, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return temp_path_caption


def main():
    """Generate the before/after animation GIF."""
    print("Generating skew before/after animation...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate frames
    print("  Creating frame 1 (raw distribution)...")
    frame_1_path = create_frame_1()
    
    print("  Creating frame 2 (log-transformed distribution)...")
    frame_2_path = create_frame_2()
    
    print("  Creating caption frame...")
    caption_path = create_caption_frame()
    
    # Read frames as images
    frame_1 = imageio.imread(frame_1_path)
    frame_2 = imageio.imread(frame_2_path)
    caption_frame = imageio.imread(caption_path)
    
    # Ensure all frames have the same dimensions by resizing to the largest
    max_height = max(frame_1.shape[0], frame_2.shape[0], caption_frame.shape[0])
    max_width = max(frame_1.shape[1], frame_2.shape[1], caption_frame.shape[1])
    
    from PIL import Image
    def resize_to_match(img_array, target_height, target_width):
        """Resize image array to target dimensions."""
        img = Image.fromarray(img_array)
        # Create new image with white background
        new_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        # Paste original image centered
        x_offset = (target_width - img.width) // 2
        y_offset = (target_height - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        return np.array(new_img)
    
    frame_1_resized = resize_to_match(frame_1, max_height, max_width)
    frame_2_resized = resize_to_match(frame_2, max_height, max_width)
    caption_resized = resize_to_match(caption_frame, max_height, max_width)
    
    # Create GIF with frames shown twice each and caption at the end
    frames = [frame_1_resized, frame_1_resized, frame_2_resized, frame_2_resized, caption_resized, caption_resized]
    durations = [DURATION, DURATION, DURATION, DURATION, DURATION * 2, DURATION * 2]
    
    print(f"  Saving GIF to {OUTPUT_PATH}...")
    imageio.mimsave(OUTPUT_PATH, frames, duration=durations, loop=0)
    
    # Clean up temporary files
    os.remove(frame_1_path)
    os.remove(frame_2_path)
    os.remove(caption_path)
    
    print(f"✓ Animation saved to {OUTPUT_PATH}")
    print(f"  Frames: 3 (before, after, caption)")
    print(f"  Duration: {DURATION}ms per frame")


if __name__ == "__main__":
    main()
