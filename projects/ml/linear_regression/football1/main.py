"""
Football Player Potential Prediction using Linear Regression

This script implements a machine learning pipeline to predict football player potential
using various player attributes. The pipeline includes data loading, preprocessing,
feature engineering, and training using custom gradient descent implementations.

Key features:
- Handles large datasets using Dask for efficient processing
- Robust data preprocessing including height/weight parsing
- Custom gradient descent implementations for linear regression
- Comprehensive evaluation and visualization
"""

import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ml_models import manual_gradient_descent, chatgpt_gradient_descent
from ml_evaluation import evaluate_regression

# --------------------------
# 1. Load the dataset
# --------------------------
# Load football player data using Dask for efficient handling of large datasets
# The dataset contains various player attributes that will be used to predict potential
df = dd.read_csv(
    r"C:\repos\testprojects\football1\data\kl.csv",
    assume_missing=True,     # make int columns nullable for dirty data
    blocksize="16MB",        # chunk size; adjust if memory issues
    dtype=str,               # load as string first to avoid parsing errors
    on_bad_lines="skip",     # skip malformed rows
    encoding_errors="ignore" # skip weird encodings
)

# --------------------------
# 2. Clean and convert specific columns
# --------------------------
# The dataset contains height and weight in string formats that need conversion
# to numeric values for machine learning processing

# Convert height strings like "5'11" → total inches
# This handles the common format where height is given as feet'inches
df['Height'] = df.map_partitions(
    lambda pdf: pdf['Height'].apply(
        lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1])
        if isinstance(x, str) else None
    ),
    meta=('Height', 'float32')
)

# Convert weight strings to numeric
# Assumes format like "187lbs" - extracts the numeric part from weight strings
# Note: This parsing might need enhancement for inconsistent formats
df['Weight'] = df.map_partitions(
    lambda pdf: pdf['Weight'].apply(
        lambda x: pd.to_numeric(x[-3:], errors='coerce') if isinstance(x, str) else None
    ),
    meta=('Weight', 'float64')
)

# Convert all remaining columns to numeric (bad values → NaN)
# This ensures all data is in numeric format suitable for machine learning
df = df.map_partitions(lambda pdf: pdf.apply(pd.to_numeric, errors='coerce'))

# Force all numeric columns to float64 to avoid Dask array dtype issues
# Consistent data types prevent computation errors in later stages
df = df.map_partitions(lambda pdf: pdf.astype('float64'))

# --------------------------
# 3. Feature selection and data cleaning
# --------------------------
# Remove features that don't contribute meaningful information to the model

# Find columns where all values are NaN across all partitions
# These columns provide no useful information and should be removed
all_nan_cols = df.columns[df.isna().all().compute()].tolist()

# Drop those columns to reduce dimensionality and improve performance
df = df.drop(columns=all_nan_cols)

# Keep only numeric columns for machine learning
# Non-numeric columns cannot be used directly in regression models
valid_cols = df.select_dtypes(include=['number']).columns.tolist()

# Drop non-informative ID-like columns that don't contribute to prediction
# These columns typically contain unique identifiers that don't help predict potential
for col in ['ID', 'Unnamed: 0', 'Jersey Number']:
    if col in valid_cols:
        valid_cols.remove(col)

# Filter dataset to include only selected features
df = df[valid_cols]

# --------------------------
# 4. Train/test split
# --------------------------
# Separate features (X) and target variable (y) for supervised learning
# 'Potential' is our target variable - what we want to predict

y = df['Potential']              # target column - player potential rating
X = df.drop(columns=['Potential'])  # feature columns - all other player attributes

# Convert Dask DataFrames to pandas for compatibility with scikit-learn
# This brings the distributed data into memory for processing
y = y.compute()
X = X.compute()

# Split dataset into training and testing sets (80/20 split)
# This ensures we can evaluate model performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 5. Handle missing values
# --------------------------
# Real-world datasets often contain missing values that need to be addressed
# We use mean imputation to fill missing values with the average from training data

# Impute missing values with mean from training data
# Using mean strategy provides reasonable estimates for missing numeric values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)  # fit and transform training data
X_test = imputer.transform(X_test)        # only transform test data to avoid data leakage

# --------------------------
# 6. Scale features (z-score normalization)
# --------------------------
# Feature scaling is crucial for gradient descent optimization
# It ensures all features contribute equally and improves convergence speed

# Apply standard scaling (mean=0, std=1) to normalize feature ranges
scaler = StandardScaler()
scaler.fit(X_train)  # fit only on training set to prevent data leakage

# Transform both training and test sets using the same scaling parameters
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 7. Train the model using gradient descent
# --------------------------
# Train the linear regression model using our custom gradient descent implementation
# The learning rate (alpha) controls how quickly the model learns
manual_gradient_descent(X_train_scaled, y_train, alpha=0.001)

# Alternative: Use the ChatGPT-inspired gradient descent implementation
# This version includes more detailed tracking and visualization capabilities
"""
theta, y_pred_manual, epochs, losses = chatgpt_gradient_descent(
    X_train_scaled, y_train, X_test_scaled, y_test,
    alpha=0.07, n_epochs=500, print_every=1, plot_loss=True
)
"""

# Evaluate model performance on test set
# Uncomment to see detailed evaluation metrics and plots
# evaluate_regression(y_test, y_pred_manual, plot=True)
