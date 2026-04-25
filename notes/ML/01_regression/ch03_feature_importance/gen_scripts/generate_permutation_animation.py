"""
Gen script: permutation-shuffle-loop.gif
4-frame GIF animation demonstrating permutation importance calculation.
Frame 1: Baseline test set with predictions and MAE = $55k (model weights visible but grayed)
Frame 2: MedInc column highlighted and shuffled — values visibly scrambled
Frame 3: Predictions recalculated with same weights → MAE = $73k shown in red
Frame 4: Δ MAE = +$18k labeled as permutation importance; model weights annotated as "unchanged"
Key annotation on Frame 2: "The model is never retrained — only the test column changes"
Uses California Housing dataset with deterministic random seed.
Output: ../img/permutation-shuffle-loop.gif
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import imageio.v2 as imageio

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "permutation-shuffle-loop.gif"

# Color scheme matching the chapter style
BG        = "#1a1a2e"
PANEL_BG  = "#12122a"
LABEL_CLR = "#e2e8f0"
AMBER     = "#d97706"
GREEN     = "#16a34a"
BLUE      = "#2563eb"
RED       = "#dc2626"
GRAY      = "#6b7280"

# Set random seed for reproducibility
np.random.seed(42)


def load_and_train_model():
    """Load California Housing data and train a Ridge regression model."""
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Ridge model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, feature_names


def draw_mini_data_table(ax, data, feature_name, shuffled=False, highlight=False):
    """Draw a mini data table showing feature values."""
    n_rows = min(6, len(data))
    row_h = 0.12
    
    # Title
    title = f"{feature_name} Column"
    if shuffled:
        title += " (Shuffled)"
    ax.text(0.5, 0.95, title, ha="center", va="top", 
            color=BLUE if highlight else LABEL_CLR,
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    
    # Draw data cells
    for i in range(n_rows):
        y_pos = 0.85 - i * row_h
        bg_color = BLUE if highlight else "#1e2040"
        
        # Background rectangle
        rect = mpatches.FancyBboxPatch(
            (0.15, y_pos - row_h * 0.8), 0.7, row_h * 0.7,
            boxstyle="round,pad=0.01", 
            facecolor=bg_color, edgecolor="#2d2d4e", alpha=0.3,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Value text
        ax.text(0.5, y_pos - row_h * 0.4, f"{data[i]:.2f}",
                ha="center", va="center", color=LABEL_CLR,
                fontsize=9, transform=ax.transAxes)


def draw_predictions_table(ax, y_true, y_pred, mae, highlight_error=False):
    """Draw a table showing true values, predictions, and MAE."""
    n_rows = min(6, len(y_true))
    row_h = 0.10
    
    # Headers
    ax.text(0.25, 0.95, "True", ha="center", va="top",
            color=LABEL_CLR, fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.55, 0.95, "Predicted", ha="center", va="top",
            color=LABEL_CLR, fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.80, 0.95, "Error", ha="center", va="top",
            color=LABEL_CLR, fontsize=9, fontweight="bold", transform=ax.transAxes)
    
    # Data rows
    for i in range(n_rows):
        y_pos = 0.85 - i * row_h
        error = abs(y_true[i] - y_pred[i])
        
        ax.text(0.25, y_pos, f"${y_true[i]*100:.0f}k", ha="center", va="center",
                color=LABEL_CLR, fontsize=8, transform=ax.transAxes)
        ax.text(0.55, y_pos, f"${y_pred[i]*100:.0f}k", ha="center", va="center",
                color=LABEL_CLR, fontsize=8, transform=ax.transAxes)
        ax.text(0.80, y_pos, f"${error*100:.0f}k", ha="center", va="center",
                color=RED if highlight_error else GRAY, fontsize=8, transform=ax.transAxes)
    
    # MAE summary
    mae_color = RED if highlight_error else GREEN
    ax.text(0.5, 0.15, f"MAE = ${mae*100:.0f}k", ha="center", va="center",
            color=mae_color, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG, edgecolor=mae_color, linewidth=2),
            transform=ax.transAxes)


def draw_model_weights(ax, model, feature_names, grayed=False):
    """Draw a simplified representation of model weights."""
    alpha = 0.3 if grayed else 1.0
    
    ax.text(0.5, 0.95, "Model Weights", ha="center", va="top",
            color=LABEL_CLR, fontsize=10, fontweight="bold", 
            alpha=alpha, transform=ax.transAxes)
    
    # Show top 4 feature weights
    weights = model.coef_
    top_indices = np.argsort(np.abs(weights))[-4:][::-1]
    
    y_pos = 0.80
    for idx in top_indices:
        weight = weights[idx]
        name = feature_names[idx][:10]  # Truncate long names
        
        # Draw weight bar
        bar_width = abs(weight) / np.max(np.abs(weights)) * 0.6
        color = GREEN if weight > 0 else RED
        
        rect = mpatches.Rectangle(
            (0.5, y_pos - 0.03), bar_width, 0.06,
            facecolor=color, alpha=alpha * 0.5, transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(0.15, y_pos, name, ha="left", va="center",
                color=LABEL_CLR, fontsize=8, alpha=alpha, transform=ax.transAxes)
        ax.text(0.85, y_pos, f"{weight:.2f}", ha="right", va="center",
                color=LABEL_CLR, fontsize=8, alpha=alpha, transform=ax.transAxes)
        
        y_pos -= 0.15


def create_frame_1(model, X_test, y_test, feature_names):
    """Frame 1: Baseline test set with predictions and MAE."""
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    # Title
    fig.suptitle("Frame 1: Baseline Model Performance", 
                 color=LABEL_CLR, fontsize=16, fontweight="bold", y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    # Left: Model weights (grayed)
    ax_weights = fig.add_subplot(gs[:, 0])
    ax_weights.axis("off")
    ax_weights.set_facecolor(PANEL_BG)
    draw_model_weights(ax_weights, model, feature_names, grayed=True)
    ax_weights.text(0.5, 0.05, "(unchanged throughout)", ha="center", va="top",
                    color=GRAY, fontsize=8, style="italic", transform=ax_weights.transAxes)
    
    # Middle: MedInc column
    ax_data = fig.add_subplot(gs[:, 1])
    ax_data.axis("off")
    ax_data.set_facecolor(PANEL_BG)
    draw_mini_data_table(ax_data, X_test[:, 0], 'MedInc')
    
    # Right: Predictions
    ax_pred = fig.add_subplot(gs[:, 2])
    ax_pred.axis("off")
    ax_pred.set_facecolor(PANEL_BG)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    draw_predictions_table(ax_pred, y_test[:6], y_pred[:6], mae)
    
    # Save frame
    frame_path = HERE / "temp_frame_1.png"
    plt.savefig(frame_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    return frame_path


def create_frame_2(model, X_test, y_test, feature_names):
    """Frame 2: MedInc column highlighted and shuffled."""
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    # Title
    fig.suptitle("Frame 2: Shuffle MedInc Column", 
                 color=LABEL_CLR, fontsize=16, fontweight="bold", y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    # Left: Model weights (grayed)
    ax_weights = fig.add_subplot(gs[:, 0])
    ax_weights.axis("off")
    ax_weights.set_facecolor(PANEL_BG)
    draw_model_weights(ax_weights, model, feature_names, grayed=True)
    ax_weights.text(0.5, 0.05, "(unchanged throughout)", ha="center", va="top",
                    color=GRAY, fontsize=8, style="italic", transform=ax_weights.transAxes)
    
    # Middle: MedInc column (highlighted, shuffled)
    ax_data = fig.add_subplot(gs[:, 1])
    ax_data.axis("off")
    ax_data.set_facecolor(PANEL_BG)
    
    # Shuffle MedInc
    X_test_shuffled = X_test.copy()
    rng = np.random.RandomState(42)
    X_test_shuffled[:, 0] = rng.permutation(X_test[:, 0])
    
    draw_mini_data_table(ax_data, X_test_shuffled[:, 0], 'MedInc', 
                        shuffled=True, highlight=True)
    
    # Key annotation
    ax_data.text(0.5, -0.05, 
                "⚠ The model is never retrained\nOnly the test column changes",
                ha="center", va="top", color=AMBER, fontsize=9,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", 
                facecolor=PANEL_BG, edgecolor=AMBER, linewidth=2),
                transform=ax_data.transAxes)
    
    # Right: Original predictions (not yet updated)
    ax_pred = fig.add_subplot(gs[:, 2])
    ax_pred.axis("off")
    ax_pred.set_facecolor(PANEL_BG)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    draw_predictions_table(ax_pred, y_test[:6], y_pred[:6], mae)
    
    # Save frame
    frame_path = HERE / "temp_frame_2.png"
    plt.savefig(frame_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    return frame_path, X_test_shuffled


def create_frame_3(model, X_test_shuffled, y_test, feature_names):
    """Frame 3: Predictions recalculated with shuffled data, MAE increases."""
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    # Title
    fig.suptitle("Frame 3: Recalculate Predictions with Shuffled Feature", 
                 color=LABEL_CLR, fontsize=16, fontweight="bold", y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    # Left: Model weights (grayed, annotated as unchanged)
    ax_weights = fig.add_subplot(gs[:, 0])
    ax_weights.axis("off")
    ax_weights.set_facecolor(PANEL_BG)
    draw_model_weights(ax_weights, model, feature_names, grayed=True)
    ax_weights.text(0.5, 0.05, "✓ Weights unchanged", ha="center", va="top",
                    color=GREEN, fontsize=9, fontweight="bold", transform=ax_weights.transAxes)
    
    # Middle: MedInc column (shuffled)
    ax_data = fig.add_subplot(gs[:, 1])
    ax_data.axis("off")
    ax_data.set_facecolor(PANEL_BG)
    draw_mini_data_table(ax_data, X_test_shuffled[:, 0], 'MedInc', 
                        shuffled=True, highlight=True)
    
    # Right: Updated predictions with higher error
    ax_pred = fig.add_subplot(gs[:, 2])
    ax_pred.axis("off")
    ax_pred.set_facecolor(PANEL_BG)
    
    y_pred_shuffled = model.predict(X_test_shuffled)
    mae_shuffled = mean_absolute_error(y_test, y_pred_shuffled)
    draw_predictions_table(ax_pred, y_test[:6], y_pred_shuffled[:6], 
                          mae_shuffled, highlight_error=True)
    
    # Save frame
    frame_path = HERE / "temp_frame_3.png"
    plt.savefig(frame_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    return frame_path, mae_shuffled


def create_frame_4(model, X_test, X_test_shuffled, y_test, feature_names):
    """Frame 4: Show permutation importance as delta MAE."""
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    # Title
    fig.suptitle("Frame 4: Permutation Importance = Δ MAE", 
                 color=LABEL_CLR, fontsize=16, fontweight="bold", y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    # Left: Model weights (annotated as unchanged)
    ax_weights = fig.add_subplot(gs[:, 0])
    ax_weights.axis("off")
    ax_weights.set_facecolor(PANEL_BG)
    draw_model_weights(ax_weights, model, feature_names, grayed=True)
    ax_weights.text(0.5, 0.05, "✓ Weights unchanged", ha="center", va="top",
                    color=GREEN, fontsize=9, fontweight="bold", transform=ax_weights.transAxes)
    
    # Middle: Show delta calculation
    ax_delta = fig.add_subplot(gs[:, 1])
    ax_delta.axis("off")
    ax_delta.set_facecolor(PANEL_BG)
    
    y_pred_baseline = model.predict(X_test)
    y_pred_shuffled = model.predict(X_test_shuffled)
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mae_shuffled = mean_absolute_error(y_test, y_pred_shuffled)
    delta_mae = mae_shuffled - mae_baseline
    
    ax_delta.text(0.5, 0.80, "Baseline MAE", ha="center", va="center",
                 color=LABEL_CLR, fontsize=11, fontweight="bold", transform=ax_delta.transAxes)
    ax_delta.text(0.5, 0.70, f"${mae_baseline*100:.0f}k", ha="center", va="center",
                 color=GREEN, fontsize=14, transform=ax_delta.transAxes)
    
    ax_delta.text(0.5, 0.55, "Shuffled MAE", ha="center", va="center",
                 color=LABEL_CLR, fontsize=11, fontweight="bold", transform=ax_delta.transAxes)
    ax_delta.text(0.5, 0.45, f"${mae_shuffled*100:.0f}k", ha="center", va="center",
                 color=RED, fontsize=14, transform=ax_delta.transAxes)
    
    # Arrow
    ax_delta.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.60),
                     arrowprops=dict(arrowstyle="->", lw=2, color=AMBER),
                     transform=ax_delta.transAxes)
    
    # Delta (Permutation Importance)
    ax_delta.text(0.5, 0.25, "Permutation Importance", ha="center", va="center",
                 color=LABEL_CLR, fontsize=12, fontweight="bold", transform=ax_delta.transAxes)
    ax_delta.text(0.5, 0.12, f"Δ MAE = +${delta_mae*100:.0f}k", ha="center", va="center",
                 color=AMBER, fontsize=16, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL_BG, 
                          edgecolor=AMBER, linewidth=3),
                 transform=ax_delta.transAxes)
    
    # Right: Interpretation
    ax_interp = fig.add_subplot(gs[:, 2])
    ax_interp.axis("off")
    ax_interp.set_facecolor(PANEL_BG)
    
    ax_interp.text(0.5, 0.85, "Interpretation", ha="center", va="top",
                  color=LABEL_CLR, fontsize=12, fontweight="bold", transform=ax_interp.transAxes)
    
    interpretation = (
        "MedInc contributes\n"
        f"${delta_mae*100:.0f}k to prediction accuracy.\n\n"
        "When shuffled, the model\n"
        "loses this information,\n"
        "increasing errors.\n\n"
        "✓ Model never retrained\n"
        "✓ Only test data changed\n"
        "✓ Weights stayed constant"
    )
    ax_interp.text(0.5, 0.70, interpretation, ha="center", va="top",
                  color=LABEL_CLR, fontsize=10, linespacing=1.8,
                  transform=ax_interp.transAxes)
    
    # Save frame
    frame_path = HERE / "temp_frame_4.png"
    plt.savefig(frame_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    return frame_path


def main():
    """Generate the permutation importance animation."""
    print("Loading California Housing dataset and training model...")
    model, X_test, y_test, feature_names = load_and_train_model()
    
    # Take a small sample for visualization
    X_test = X_test[:50]
    y_test = y_test[:50]
    
    print("Generating frames...")
    frame_1_path = create_frame_1(model, X_test, y_test, feature_names)
    print("  ✓ Frame 1: Baseline")
    
    frame_2_path, X_test_shuffled = create_frame_2(model, X_test, y_test, feature_names)
    print("  ✓ Frame 2: Shuffle")
    
    frame_3_path, mae_shuffled = create_frame_3(model, X_test_shuffled, y_test, feature_names)
    print("  ✓ Frame 3: Recalculate")
    
    frame_4_path = create_frame_4(model, X_test, X_test_shuffled, y_test, feature_names)
    print("  ✓ Frame 4: Delta")
    
    # Create GIF
    print("Creating GIF animation...")
    frames = []
    target_shape = None
    for path in [frame_1_path, frame_2_path, frame_3_path, frame_4_path]:
        img = imageio.imread(path)
        if target_shape is None:
            target_shape = img.shape
        elif img.shape != target_shape:
            # Resize to match first frame
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
            img = np.array(pil_img)
        frames.append(img)
    
    # Write GIF with appropriate duration for each frame
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT, frames, duration=[2.0, 2.5, 2.0, 3.0], loop=0)
    
    # Clean up temporary frames
    for path in [frame_1_path, frame_2_path, frame_3_path, frame_4_path]:
        path.unlink()
    
    print(f"✓ Saved: {OUT}")


if __name__ == "__main__":
    main()
