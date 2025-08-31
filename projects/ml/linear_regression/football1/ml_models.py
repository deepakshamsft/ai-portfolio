"""
Machine Learning Models for Linear Regression

This module implements custom gradient descent algorithms for linear regression.
It provides two different implementations:
1. A ChatGPT-inspired implementation with comprehensive tracking and visualization
2. A manual implementation focusing on algorithmic details and convergence analysis

Both implementations use gradient descent optimization to find optimal model parameters
that minimize the mean squared error between predictions and actual values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def chatgpt_gradient_descent(X_train, y_train, X_test, y_test,
                            alpha=0.01, n_epochs=1000, print_every=50, plot_loss=True):
    """
    Multiple linear regression using gradient descent with comprehensive tracking.
    
    This implementation provides detailed monitoring of the training process,
    including loss tracking, epoch recording, and optional visualization.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training target values
    X_test : array-like
        Test feature matrix  
    y_test : array-like
        Test target values
    alpha : float, default=0.01
        Learning rate - controls step size in gradient descent
    n_epochs : int, default=1000
        Maximum number of training iterations
    print_every : int, default=50
        Frequency of loss printing (every N epochs)
    plot_loss : bool, default=True
        Whether to plot the loss convergence curve
        
    Returns:
    --------
    tuple
        (theta, y_pred_test, epochs_recorded, losses)
        - theta: learned model parameters
        - y_pred_test: predictions on test set
        - epochs_recorded: epochs where loss was recorded
        - losses: loss values at recorded epochs
    """
    # Convert Dask arrays to NumPy for computation compatibility
    X_train_np = X_train.compute()
    y_train_np = y_train.compute().values.reshape(-1,1)
    X_test_np = X_test.compute()
    y_test_np = y_test.compute().values.reshape(-1,1)
    
    # Add bias term (intercept) to feature matrix
    # This allows the model to learn a y-intercept in addition to feature weights
    X_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    
    # Initialize model parameters (weights) to zero
    # theta[0] will be the bias term, theta[1:] will be feature weights
    theta = np.zeros((X_b.shape[1], 1))
    
    # Initialize tracking variables for monitoring training progress
    losses = []
    epochs_recorded = []
    
    # Gradient descent optimization loop
    for epoch in range(n_epochs):
        # Forward pass: compute predictions using current parameters
        predictions = X_b @ theta
        
        # Compute prediction errors (residuals)
        errors = predictions - y_train_np
        
        # Compute gradient of loss function with respect to parameters
        # This tells us the direction and magnitude of parameter updates
        gradient = (1 / X_b.shape[0]) * (X_b.T @ errors)
        
        # Update parameters in the direction that reduces loss
        theta -= alpha * gradient
        
        # Record loss periodically for monitoring convergence
        if epoch % print_every == 0:
            loss = np.mean(errors**2) / 2  # Mean squared error / 2
            losses.append(loss)
            epochs_recorded.append(epoch)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Visualize training progress if requested
    if plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_recorded, losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Convergence")
        plt.grid(True)
        plt.show()
    
    # Generate predictions on test set using learned parameters
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    y_pred_test = X_test_b @ theta
    
    return theta, y_pred_test, epochs_recorded, losses


def manual_gradient_descent(X_train_scaled, y_train, alpha=0.01, n_epochs=1000):
    """
    Manual implementation of multiple linear regression using gradient descent.
    
    This implementation provides a step-by-step approach to gradient descent
    with detailed convergence monitoring and automatic stopping criteria.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Scaled training feature matrix (should be preprocessed)
    y_train : array-like
        Training target values
    alpha : float, default=0.01
        Learning rate - controls how much to update parameters each iteration
    n_epochs : int, default=1000
        Maximum number of training iterations
        
    Returns:
    --------
    None
        This function prints training progress and displays convergence plot
        
    Key Features:
    -------------
    - Automatic convergence detection based on error change
    - Real-time error monitoring and printing
    - Visualization of error reduction over epochs
    - Robust handling of numerical data types
    """
    # Convert target values to numpy array with proper shape and type
    # Reshape to column vector for matrix operations
    y = y_train.to_numpy().reshape(-1, 1).astype(float)
    
    # Convert feature matrix to float for numerical stability
    X = X_train_scaled.astype(float)
    
    # Add bias column (intercept term) to feature matrix
    # This allows the model to learn a y-intercept
    ones = np.ones((X.shape[0], 1)).astype(float)
    X = np.hstack([ones, X])
    
    # Initialize model parameters (weights) to zero
    # w[0] will be the bias term, w[1:] will be feature weights
    w = np.zeros((X.shape[1], 1)).astype(float)
    
    # Initialize tracking variables
    errors = np.array([])  # Track sum of squared errors over epochs
    epochs = 0             # Count of completed epochs
    num_of_rows = X.shape[0]  # Number of training samples
    
    # Gradient descent optimization loop with automatic stopping
    while errors.size == 0 or (errors[-1] > 0 and epochs < n_epochs):
        epochs += 1
        
        # Forward pass: compute predictions using current weights
        y_pred = (X @ w)
        
        # Compute prediction errors (residuals)
        e = y - y_pred
        
        # Compute sum of squared errors (SSE) for this epoch
        # This is our loss function that we want to minimize
        e_sum = (e.T @ e).item()
        errors = np.append(errors, e_sum)
        print(f"Epoch {epochs}: SSE = {e_sum:.4f}")
        
        # Compute gradient of loss function with respect to weights
        # The gradient points in the direction of steepest increase
        dw = (X.T @ e) / num_of_rows
        
        # Update weights in the direction that reduces error
        # Multiply by learning rate to control step size
        w = w + (alpha * dw)
        
        # Check for convergence: stop if error change is very small
        # This prevents unnecessary computation when model has converged
        if errors.size > 1 and abs(errors[-1] - errors[-2]) < 0.1:
            print(f"Converged after {epochs} epochs!")
            break
    
    # Visualize the training progress
    # Plot shows how the error decreases over training epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch Number")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title("Training Progress: Error Reduction Over Time")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Final model parameters:")
    print(f"Bias (intercept): {w[0, 0]:.4f}")
    print(f"Feature weights: {w[1:].flatten()}")
    
    return w
