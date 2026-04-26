"""
gen_ch05_l1_l2_geometry.py
Generates: ../img/ch05-l1-l2-geometry.png

Shows MSE tapering during gradient descent for three methods:
1. No regularization (OLS) - fast convergence but may overfit
2. L2 Ridge - smooth convergence with penalty
3. L1 Lasso - convergence with sparsity-inducing steps
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-l1-l2-geometry.png")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)
X_scaled = StandardScaler().fit_transform(X_poly)

X_test_poly = poly.transform(X_test)
X_test_scaled = StandardScaler().fit_transform(X_test_poly)

# ── Simulate gradient descent: track MSE over iterations ─────────────────────
# For simplicity, use sklearn's built-in iterative solvers with max_iter tracking
# We'll use SGDRegressor which allows warm_start to track convergence

from sklearn.linear_model import SGDRegressor

def track_convergence(model_name, penalty, alpha, max_iter=200):
    """Track MSE at each iteration during gradient descent."""
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
max_iter = 150

ols_mse = track_convergence("OLS", None, None, max_iter=max_iter)
ridge_mse = track_convergence("Ridge", 'l2', alpha=0.001, max_iter=max_iter)
lasso_mse = track_convergence("Lasso", 'l1', alpha=0.0001, max_iter=max_iter)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

iterations = np.arange(1, max_iter + 1)

ax.plot(iterations, ols_mse, color="#f87171", linewidth=2.5, 
        label="No Regularization (OLS)", alpha=0.9)
ax.plot(iterations, ridge_mse, color="#60a5fa", linewidth=2.5,
        label="Ridge (L2, λ=0.001)", alpha=0.9)
ax.plot(iterations, lasso_mse, color="#4ade80", linewidth=2.5,
        label="Lasso (L1, λ=0.0001)", alpha=0.9)

ax.set_xlabel("Gradient Descent Iterations", color="white", fontsize=13)
ax.set_ylabel("Test MSE", color="white", fontsize=13)
ax.set_title("MSE Convergence: Regularization Slows Descent but Improves Generalization",
             color="white", fontsize=14, pad=15)

ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values():
    spine.set_edgecolor("#4a4a6a")

ax.legend(loc="upper right", fontsize=11, framealpha=0.3,
          labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

ax.grid(True, alpha=0.15, color="white", linestyle=":")

# Add annotations
final_ols = ols_mse[-1]
final_ridge = ridge_mse[-1]
final_lasso = lasso_mse[-1]

ax.annotate(f"Final MSE: {final_ols:.4f}",
            xy=(max_iter, final_ols), xytext=(max_iter - 30, final_ols + 0.03),
            color="#f87171", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", 
                     edgecolor="#f87171", alpha=0.7))

ax.annotate(f"Final MSE: {final_ridge:.4f}",
            xy=(max_iter, final_ridge), xytext=(max_iter - 30, final_ridge - 0.03),
            color="#60a5fa", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                     edgecolor="#60a5fa", alpha=0.7))

ax.annotate(f"Final MSE: {final_lasso:.4f}",
            xy=(max_iter, final_lasso), xytext=(max_iter - 30, final_lasso),
            color="#4ade80", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                     edgecolor="#4ade80", alpha=0.7))

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
plt.close()
print(f"✓ Generated: {OUT_PATH}")

