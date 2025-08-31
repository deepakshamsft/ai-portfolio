# Football Player Potential Prediction using Linear Regression

A machine learning project that predicts football player potential ratings using various player attributes through custom gradient descent implementations.

## 📋 Project Overview

This project implements a complete machine learning pipeline for predicting football player potential ratings. It uses linear regression with custom gradient descent algorithms to learn relationships between player attributes (height, weight, skill ratings, etc.) and their overall potential.

### Key Features

- **Robust Data Processing**: Handles real-world messy data with missing values and inconsistent formats
- **Custom Gradient Descent**: Two different implementations of gradient descent optimization
- **Comprehensive Evaluation**: Detailed metrics and visualizations for model assessment
- **Scalable Architecture**: Uses Dask for efficient handling of large datasets

## 🎯 Project Goals

- Predict football player potential ratings based on current attributes
- Demonstrate custom implementation of gradient descent algorithms
- Showcase data preprocessing techniques for real-world datasets
- Provide comprehensive model evaluation and visualization

## 📁 Project Structure

```
football1/
├── README.md                   # Project documentation (this file)
├── main.py                     # Main execution script
├── ml_models.py               # Custom gradient descent implementations
├── ml_evaluation.py           # Model evaluation and visualization utilities
├── requirements.txt           # Python dependencies
└── data/
    └── kl.csv                 # Football player dataset
```

## 📊 Dataset

The project uses a football player dataset (`kl.csv`) containing:

- **Player Attributes**: Height, weight, age, position-specific skills
- **Target Variable**: Player potential rating (what we want to predict)
- **Data Challenges**: Missing values, inconsistent formats (e.g., "5'11" for height, "187lbs" for weight)

### Data Preprocessing Steps

1. **Format Conversion**: Height strings → inches, Weight strings → numeric values
2. **Missing Value Handling**: Mean imputation for numeric features
3. **Feature Selection**: Remove non-informative columns (IDs, jersey numbers)
4. **Normalization**: Z-score standardization for gradient descent optimization

## 🧠 Machine Learning Approach

### Model: Multiple Linear Regression

The project implements linear regression to predict player potential using the equation:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `y` = Player potential (target)
- `β₀` = Bias term (intercept)
- `βᵢ` = Feature weights
- `xᵢ` = Player attributes (features)

### Optimization: Gradient Descent

Two custom implementations:

1. **ChatGPT-Inspired Implementation** (`chatgpt_gradient_descent`)
   - Comprehensive loss tracking and visualization
   - Epoch-based monitoring with customizable reporting frequency
   - Returns detailed training history

2. **Manual Implementation** (`manual_gradient_descent`)
   - Step-by-step algorithmic approach
   - Automatic convergence detection
   - Real-time error monitoring

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository** (if part of larger project):
   ```bash
   git clone <repository-url>
   cd projects/ml/linear_regression/football1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Ensure dataset is available**:
   - Place `kl.csv` in the `data/` directory
   - Update the file path in `main.py` if necessary

2. **Execute the main script**:
   ```bash
   python main.py
   ```

3. **Expected Output**:
   - Training progress with epoch-by-epoch loss values
   - Convergence visualization plots
   - Model parameter values
   - Performance metrics (MSE, R²)

## 📈 Model Evaluation

The project provides comprehensive evaluation through:

### Metrics
- **Mean Squared Error (MSE)**: Average squared differences between predictions and actual values
- **R² Score**: Proportion of variance in the target variable explained by the model
- **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretability

### Visualizations
- **Training Progress**: Loss reduction over epochs
- **Actual vs Predicted**: How well the model predicts compared to true values
- **Residuals Analysis**: Distribution of prediction errors
- **Scatter Plot**: Direct comparison of predicted vs actual values

## 🔧 Configuration Options

### Hyperparameters

Modify these in `main.py` to experiment with different settings:

```python
# Learning rate - controls step size in gradient descent
alpha = 0.001

# Maximum training iterations
n_epochs = 1000

# Train/test split ratio
test_size = 0.2

# Loss reporting frequency
print_every = 50
```

### Model Selection

Choose between implementations by commenting/uncommenting in `main.py`:

```python
# Use manual implementation
manual_gradient_descent(X_train_scaled, y_train, alpha=0.001)

# Use ChatGPT-inspired implementation (commented by default)
# theta, y_pred_manual, epochs, losses = chatgpt_gradient_descent(
#     X_train_scaled, y_train, X_test_scaled, y_test,
#     alpha=0.07, n_epochs=500, print_every=1, plot_loss=True
# )
```

## 📋 File Descriptions

### `main.py`
Main execution script that orchestrates the entire machine learning pipeline:
- Data loading and preprocessing
- Feature engineering and selection
- Train/test splitting
- Model training and evaluation

### `ml_models.py`
Contains two gradient descent implementations:
- `chatgpt_gradient_descent()`: Feature-rich implementation with tracking
- `manual_gradient_descent()`: Educational implementation with detailed steps

### `ml_evaluation.py`
Model evaluation utilities:
- `evaluate_regression()`: Comprehensive regression model assessment
- Statistical metrics computation
- Diagnostic plot generation

### `requirements.txt`
Python package dependencies including:
- Data processing: `pandas`, `dask`, `numpy`
- Machine learning: `scikit-learn`
- Visualization: `matplotlib`

## 🔬 Technical Implementation Details

### Data Processing Pipeline

1. **Dask Integration**: Efficient handling of large datasets with lazy evaluation
2. **Type Safety**: Consistent float64 conversion to prevent numerical issues
3. **Memory Efficiency**: Chunked processing with configurable block sizes

### Gradient Descent Algorithm

The core optimization uses the update rule:
```
θ = θ - α × ∇J(θ)
```

Where:
- `θ` = model parameters
- `α` = learning rate
- `∇J(θ)` = gradient of cost function

### Convergence Criteria

- Maximum epoch limit
- Automatic stopping when error change < threshold (0.1)
- Real-time monitoring of training progress

## 🎓 Learning Outcomes

This project demonstrates:

1. **Data Preprocessing**: Handling real-world messy data
2. **Feature Engineering**: Converting categorical/string data to numeric
3. **Algorithm Implementation**: Building gradient descent from scratch
4. **Model Evaluation**: Comprehensive assessment techniques
5. **Visualization**: Effective communication of results

## 🚀 Future Enhancements

Potential improvements and extensions:

1. **Advanced Algorithms**: Implement Ridge/Lasso regression
2. **Feature Engineering**: Add polynomial features, interaction terms
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Cross-Validation**: More robust model evaluation
5. **Model Comparison**: Compare with scikit-learn implementations
6. **Production Pipeline**: Model serialization and deployment

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## 📝 License

[Add license information if applicable]

## 👥 Authors

[Add author information]

---

**Note**: This project is designed for educational purposes to demonstrate machine learning concepts and custom algorithm implementation. For production use, consider using established libraries like scikit-learn.