"""
Model Evaluation Utilities for Regression Tasks

This module provides comprehensive evaluation capabilities for regression models,
including statistical metrics computation and visualization tools to assess
model performance and identify potential issues.

Key features:
- Computation of standard regression metrics (MSE, R²)
- Visual comparison of actual vs predicted values
- Residual analysis for model diagnostics
- Support for both Dask and NumPy data structures
"""

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_regression(y_test, y_pred, plot=True):
    """
    Comprehensive evaluation of regression model predictions.
    
    This function computes standard regression metrics and provides
    detailed visualizations to assess model performance and diagnose
    potential issues such as bias, variance, or systematic errors.
    
    Parameters:
    -----------
    y_test : Dask Series, pandas Series, or NumPy array
        True target values from the test set
    y_pred : NumPy array
        Model predictions corresponding to y_test
    plot : bool, default=True
        Whether to generate diagnostic plots
        
    Metrics Computed:
    -----------------
    - Mean Squared Error (MSE): Average squared differences between predictions and actual values
    - R² Score: Coefficient of determination (proportion of variance explained by model)
    
    Visualizations:
    ---------------
    - Actual vs Predicted plot: Shows how well predictions match true values
    - Residuals plot: Reveals patterns in prediction errors that might indicate model issues
    
    Returns:
    --------
    None
        Prints metrics to console and displays plots if requested
    """
    # Handle different data types - convert Dask Series to NumPy if necessary
    # This ensures compatibility with scikit-learn metrics
    if hasattr(y_test, 'compute'):
        y_test_np = y_test.compute()
    else:
        y_test_np = y_test

    # Compute standard regression evaluation metrics
    mse = mean_squared_error(y_test_np, y_pred)  # Lower values indicate better fit
    r2 = r2_score(y_test_np, y_pred)             # Values closer to 1 indicate better fit
    
    print("=" * 50)
    print("REGRESSION MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print("=" * 50)
    
    # Generate diagnostic plots if requested
    if plot:
        # Create sample indices for x-axis in plots
        idx = np.arange(len(y_test_np))
        
        # Plot 1: Actual vs Predicted Values
        # This plot helps visualize how closely predictions match true values
        # Perfect predictions would show points along a diagonal line
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(idx, y_test_np, 'o-', label='Actual Values', color='blue', alpha=0.7, markersize=4)
        plt.plot(idx, y_pred, 'x-', label='Predicted Values', color='orange', alpha=0.7, markersize=4)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value (Player Potential)")
        plt.title("Model Predictions vs Actual Values")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Residuals Analysis
        # Residuals should be randomly distributed around zero for a good model
        # Patterns in residuals may indicate model bias or missing features
        residuals = y_pred.flatten() - y_test_np.flatten()
        
        plt.subplot(1, 2, 2)
        plt.scatter(idx, residuals, alpha=0.6, color='purple', s=30)
        plt.hlines(0, 0, len(residuals), colors='red', linestyles='dashed', linewidth=2)
        plt.xlabel("Sample Index")
        plt.ylabel("Residuals (Predicted - Actual)")
        plt.title("Residuals Analysis")
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics to residuals plot
        plt.text(0.02, 0.95, f'Mean Residual: {np.mean(residuals):.3f}\nStd Residual: {np.std(residuals):.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Additional diagnostic: Scatter plot of predicted vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_np, y_pred, alpha=0.6, color='green', s=30)
        
        # Add perfect prediction line (diagonal)
        min_val = min(np.min(y_test_np), np.min(y_pred))
        max_val = max(np.max(y_test_np), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values\n(Points closer to red line indicate better predictions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
