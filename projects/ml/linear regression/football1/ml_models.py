# ml_models.py (enhanced visualization)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def chatgpt_gradient_descent(X_train, y_train, X_test, y_test,
                            alpha=0.01, n_epochs=1000, print_every=50, plot_loss=True):
    """
    Multiple linear regression using gradient descent.
    Tracks loss over epochs and plots if desired.
    """
    X_train_np = X_train.compute()
    y_train_np = y_train.compute().values.reshape(-1,1)
    X_test_np = X_test.compute()
    y_test_np = y_test.compute().values.reshape(-1,1)
    
    # Add bias term
    X_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    theta = np.zeros((X_b.shape[1], 1))
    
    losses = []
    epochs_recorded = []
    
    for epoch in range(n_epochs):
        predictions = X_b @ theta
        errors = predictions - y_train_np
        gradient = (1 / X_b.shape[0]) * (X_b.T @ errors)
        theta -= alpha * gradient
        
        if epoch % print_every == 0:
            loss = np.mean(errors**2) / 2
            losses.append(loss)
            epochs_recorded.append(epoch)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Plot loss curve
    if plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_recorded, losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Convergence")
        plt.grid(True)
        plt.show()
    
    # Predict on test set
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    y_pred_test = X_test_b @ theta
    
    return theta, y_pred_test, epochs_recorded, losses


def manual_gradient_descent(X_train_scaled, y_train, alpha=0.01, n_epochs=1000):
    """
    Multiple linear regression using gradient descent with Dask arrays.
    """
    # Convert y to a Dask array
    y =  y_train.to_numpy().reshape(-1,1).astype(float)
    
    # Convert X to a Dask array
    X = X_train_scaled.astype(float)
    
    # Add bias column
    ones = np.ones((X.shape[0], 1)).astype(float)
    X = np.hstack([ones, X])
    
    # Initialize weights
    w = np.zeros((X.shape[1], 1)).astype(float)
    
    errors = np.array([])
    epochs = 0
    num_of_rows = X.shape[0]

    while errors.size == 0 or (errors[-1] > 0 and epochs < n_epochs):
        epochs += 1
        
        # Compute predictions
        y_pred = (X @ w)
        
        # Compute error
        e = y - y_pred
        
        # Compute loss (SSE)
        e_sum = (e.T @ e).item()
        errors = np.append(errors, e_sum)
        print(e_sum)
        
        # Gradient
        dw = (X.T @ e)/ num_of_rows
        
        # Update weights
        w = w + (alpha * dw)
        
        # Stop if change is small
        if errors.size > 1 and abs(errors[-1]-errors[-2]) < 0.1:
            break
        
    plt.plot(errors)  # x-axis = indices 0,1,2,...
    plt.xlabel("Epoch Number")
    plt.ylabel("Standard Error")
    plt.title("Epoch number vs SSE")
    plt.show()
    
    """
    X_train_np = X_train.compute()
    y_train_np = y_train.compute().values.reshape(-1,1)
    X_test_np = X_test.compute()
    y_test_np = y_test.compute().values.reshape(-1,1)
    
    # Add bias term
    X_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    theta = np.zeros((X_b.shape[1], 1))
    
    losses = []
    epochs_recorded = []
    
    for epoch in range(n_epochs):
        predictions = X_b @ theta
        errors = predictions - y_train_np
        gradient = (1 / X_b.shape[0]) * (X_b.T @ errors)
        theta -= alpha * gradient
        
        if epoch % print_every == 0:
            loss = np.mean(errors**2) / 2
            losses.append(loss)
            epochs_recorded.append(epoch)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Plot loss curve
    if plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_recorded, losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Convergence")
        plt.grid(True)
        plt.show()
    
    # Predict on test set
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    y_pred_test = X_test_b @ theta
    
    return theta, y_pred_test, epochs_recorded, losses
    """