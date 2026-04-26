"""
gen_ch05_mse_convergence_comparison.py
Generates: ../img/ch05-mse-convergence-comparison.png

Shows how MSE (baseline), MSE+L1 (Lasso), and MSE+L2 (Ridge) get minimized 
over gradient descent epochs with the SAME regularization parameter (λ=0.001).
This demonstrates the different optimization paths these loss functions take.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-mse-convergence-comparison.png")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)
X_scaled = StandardScaler().fit_transform(X_poly)

X_test_poly = poly.transform(X_test)
X_test_scaled = StandardScaler().fit_transform(X_test_poly)

# ── Track convergence (same pattern as original working script) ───────────────
def track_convergence(penalty, alpha, max_iter=80):
    """Track test MSE at each iteration during gradient descent."""
    mse_history = []
    
    for i in range(1, max_iter + 1):
        if penalty is None:
            model = SGDRegressor(penalty=None, max_iter=i, random_state=42,
                                learning_rate='optimal', early_stopping=False)
        elif penalty == 'l2':
            model = SGDRegressor(penalty='l2', alpha=alpha, max_iter=i, 
                                random_state=42, learning_rate='optimal',
                                early_stopping=False)
        else:  # l1
            model = SGDRegressor(penalty='l1', alpha=alpha, max_iter=i,
                                random_state=42, learning_rate='optimal',
                                early_stopping=False)
        
        model.fit(X_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_test)
        mse_history.append(mse)
    
    return mse_history

print("Training models and tracking MSE convergence...")
ALPHA = 0.001  # Same regularization parameter for fair comparison
max_iter = 80  # Reduced for performance

print("  No regularization...")
ols_mse = track_convergence(None, None, max_iter=max_iter)
print("  Ridge (L2)...")
ridge_mse = track_convergence('l2', ALPHA, max_iter=max_iter)
print("  Lasso (L1)...")
lasso_mse = track_convergence('l1', ALPHA, max_iter=max_iter)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

epochs = np.arange(1, max_iter + 1)

ax.plot(epochs, ols_mse, color="#f87171", linewidth=3.0, 
        label="MSE (no regularization)", alpha=0.95, linestyle='-')
ax.plot(epochs, ridge_mse, color="#60a5fa", linewidth=3.0,
        label=f"MSE + L2 (Ridge, λ={ALPHA})", alpha=0.95, linestyle='-')
ax.plot(epochs, lasso_mse, color="#4ade80", linewidth=3.0,
        label=f"MSE + L1 (Lasso, λ={ALPHA})", alpha=0.95, linestyle='-')

ax.set_xlabel("Gradient Descent Epochs", color="white", fontsize=14, fontweight='bold')
ax.set_ylabel("Test MSE", color="white", fontsize=14, fontweight='bold')
ax.set_title(f"Loss Function Convergence (λ={ALPHA} for all regularized models)",
             color="white", fontsize=15, pad=20, fontweight='bold')

ax.tick_params(colors="white", labelsize=12)
for spine in ax.spines.values():
    spine.set_edgecolor("#4a4a6a")
    spine.set_linewidth(1.5)

ax.legend(loc="upper right", fontsize=12, framealpha=0.4,
          labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

ax.grid(True, alpha=0.2, color="white", linestyle=":", linewidth=0.8)

# Add final value annotations
final_vals = [
    (ols_mse[-1], "#f87171", "No reg"),
    (ridge_mse[-1], "#60a5fa", "Ridge"),
    (lasso_mse[-1], "#4ade80", "Lasso")
]

for final_mse, color, label in final_vals:
    ax.plot(max_iter, final_mse, 'o', color=color, markersize=10, 
            markeredgewidth=2, markeredgecolor='white')
    ax.annotate(f"{label}\n{final_mse:.4f}",
                xy=(max_iter, final_mse), 
                xytext=(max_iter - 20, final_mse),
                color=color, fontsize=10, ha="right", va="center",
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", 
                         edgecolor=color, linewidth=2, alpha=0.8))

# Add insight text box
insight_text = (
    "Key Insight: All three converge but at different rates.\n"
    "Regularization (L1/L2) slows convergence but often achieves\n"
    "better generalization (lower final test MSE)."
)
ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
        fontsize=10, color="white", va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', 
                 edgecolor='#4a4a6a', alpha=0.7, pad=10))

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
plt.close()
print(f"✓ Generated: {OUT_PATH}")
print(f"  - No regularization final MSE: {ols_mse[-1]:.4f}")
print(f"  - Ridge (L2) final MSE: {ridge_mse[-1]:.4f}")
print(f"  - Lasso (L1) final MSE: {lasso_mse[-1]:.4f}")
