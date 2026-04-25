"""
Gen script: three-lens-convergence.png
Static 2×2 grid diagram showing the three-lens framing for feature importance.
X-axis: Univariate R² (Low → High)
Y-axis: Methods 2+3 Joint (Low → High)
Four quadrants with interpretations and California Housing feature examples:
- High/High → "Strong independent signal" (MedInc)
- High/Low → "Shared signal (correlated)" (AveRooms)
- Low/High → "Jointly irreplaceable" (Latitude/Longitude)
- Low/Low → "Genuinely uninformative" (Population)
Output: ../img/three-lens-convergence.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "three-lens-convergence.png"

# Color scheme matching the chapter style
BG        = "#1a1a2e"
PANEL_BG  = "#12122a"
LABEL_CLR = "#e2e8f0"
AMBER     = "#d97706"
GREEN     = "#16a34a"
BLUE      = "#2563eb"
RED       = "#dc2626"
GRAY      = "#6b7280"

def main():
    """Generate the three-lens convergence diagram."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    
    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Univariate R²", color=LABEL_CLR, fontsize=14, fontweight="bold")
    ax.set_ylabel("Methods 2+3 Joint", color=LABEL_CLR, fontsize=14, fontweight="bold")
    ax.set_title("Three-Lens Convergence Framework\nFeature Importance Interpretation", 
                 color=LABEL_CLR, fontsize=16, fontweight="bold", pad=20)
    
    # Style the axes
    ax.tick_params(colors=LABEL_CLR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(2)
    
    # Add grid lines at the midpoint
    ax.axhline(0.5, color=GRAY, linewidth=1.5, linestyle='--', alpha=0.5)
    ax.axvline(0.5, color=GRAY, linewidth=1.5, linestyle='--', alpha=0.5)
    
    # Add "Low" and "High" labels on axes
    ax.text(0.15, -0.08, "Low", color=GRAY, fontsize=12, ha="center", transform=ax.transData)
    ax.text(0.85, -0.08, "High", color=GRAY, fontsize=12, ha="center", transform=ax.transData)
    ax.text(-0.08, 0.15, "Low", color=GRAY, fontsize=12, ha="center", va="center", rotation=90, transform=ax.transData)
    ax.text(-0.08, 0.85, "High", color=GRAY, fontsize=12, ha="center", va="center", rotation=90, transform=ax.transData)
    
    # Define quadrants with colors and content
    quadrants = [
        {
            "x": 0.75, "y": 0.75,
            "color": GREEN,
            "title": "Strong Independent Signal",
            "description": "High univariate predictive power\nRetains importance in joint models",
            "example": "Example: MedInc",
            "example_detail": "Predicts alone (R²=0.47)\nRemains critical in full model"
        },
        {
            "x": 0.75, "y": 0.25,
            "color": AMBER,
            "title": "Shared Signal (Correlated)",
            "description": "Appears strong in isolation\nRedundant with other features",
            "example": "Example: AveRooms",
            "example_detail": "Univariate R²=0.15\nCorrelated with MedInc"
        },
        {
            "x": 0.25, "y": 0.75,
            "color": BLUE,
            "title": "Jointly Irreplaceable",
            "description": "Weak or non-linear alone\nCrucial in combination",
            "example": "Example: Latitude/Longitude",
            "example_detail": "Low univariate R²\nJointly capture location effects"
        },
        {
            "x": 0.25, "y": 0.25,
            "color": RED,
            "title": "Genuinely Uninformative",
            "description": "Low contribution in all methods\nCandidate for removal",
            "example": "Example: Population",
            "example_detail": "Low univariate R²\nMinimal joint contribution"
        }
    ]
    
    # Draw each quadrant
    for quad in quadrants:
        # Determine quadrant boundaries
        x_start = 0 if quad["x"] < 0.5 else 0.5
        y_start = 0 if quad["y"] < 0.5 else 0.5
        
        # Add background rectangle for quadrant
        rect = mpatches.Rectangle(
            (x_start + 0.02, y_start + 0.02), 0.46, 0.46,
            facecolor=quad["color"], alpha=0.08, zorder=1
        )
        ax.add_patch(rect)
        
        # Add border for quadrant
        border = mpatches.Rectangle(
            (x_start + 0.02, y_start + 0.02), 0.46, 0.46,
            facecolor='none', edgecolor=quad["color"], linewidth=2.5, zorder=2
        )
        ax.add_patch(border)
        
        # Add title
        ax.text(
            quad["x"], y_start + 0.42,
            quad["title"],
            color=quad["color"], fontsize=12, fontweight="bold",
            ha="center", va="center", zorder=3
        )
        
        # Add description
        ax.text(
            quad["x"], y_start + 0.32,
            quad["description"],
            color=LABEL_CLR, fontsize=9.5, ha="center", va="center",
            zorder=3, linespacing=1.5
        )
        
        # Add example label
        ax.text(
            quad["x"], y_start + 0.18,
            quad["example"],
            color=quad["color"], fontsize=10, fontweight="bold",
            ha="center", va="center", zorder=3, style="italic"
        )
        
        # Add example details
        ax.text(
            quad["x"], y_start + 0.08,
            quad["example_detail"],
            color=GRAY, fontsize=8.5, ha="center", va="center",
            zorder=3, linespacing=1.5
        )
    
    # Add interpretation note at the bottom
    note_text = (
        "Interpretation Guide: Use all three lenses (univariate, permutation, model-based) together.\n"
        "Features in different quadrants require different treatment in feature selection and engineering."
    )
    ax.text(
        0.5, -0.15,
        note_text,
        color=GRAY, fontsize=9, ha="center", va="top",
        style="italic", transform=ax.transData
    )
    
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"✓ Saved: {OUT}")
    plt.close()


if __name__ == "__main__":
    main()
