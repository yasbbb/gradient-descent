import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Load dataset
subprocess.run(["git", "clone", "https://github.com/yasbbb/gradient-descent.git"])
data = pd.read_csv("gradient-descent/ObesityDataSet_raw_and_data_sinthetic.csv")

# Check the number of rows before and after cleaning
initial_row_count = data.shape[0]  # Rows before
data.drop_duplicates(inplace=True)
final_row_count = data.shape[0]  # Rows after

# Display how many rows were removed
print(f"\nRows removed: {initial_row_count - final_row_count}\n")

# Convert categorical columns to numerical
binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC'] # columns with two variables 
multi_class_cols = ['CAEC', 'CALC', 'MTRANS'] #columns with three or more variables
target_col = 'NObeyesdad' #target variable 

for col in binary_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Apply One-Hot Encoding to multi-class columns
data = pd.get_dummies(data, columns=multi_class_cols, drop_first=True)

# Encode target variable if it's categorical
le_target = LabelEncoder()
data[target_col] = le_target.fit_transform(data[target_col])

# Step 1: Preprocessing
# Separate features and target variable
X = data.drop(columns=['NObeyesdad']) 
y = data['NObeyesdad']

# Transform skewed features using PowerTransformer 
power_transformer = PowerTransformer(method='yeo-johnson')
X_transformed = power_transformer.fit_transform(X)

# Polynomial Features for Interaction terms and non-linearity capture
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_transformed)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Step 2: Split the dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Hyperparameter Optimization and Model Evaluation
learning_rates = [0.005, 0.01, 0.05]
n_iterations_list = [1000, 2000, 5000]
best_params = None
best_mse = float('inf')
log_file = open("gradient_descent_log.txt", "w")

# Create a dictionary to store error histories for all models
error_histories = {}

# Loop over hyperparameters
for lr in learning_rates:
    for n_iterations in n_iterations_list:
        m, n = X_train.shape
        weights = np.zeros(n)
        bias = 0
        error_history = []

        # Gradient Descent Loop
        for i in range(n_iterations):
            y_pred = np.dot(X_train, weights) + bias
            error = y_pred - y_train
            mse = mean_squared_error(y_train, y_pred)
            error_history.append(mse)

            # Calculate gradients
            dW = (1/m) * np.dot(X_train.T, error)
            dB = (1/m) * np.sum(error)

            # Update weights and bias
            weights -= lr * dW
            bias -= lr * dB

        # Store error history for this model
        error_histories[(lr, n_iterations)] = error_history

        # Evaluate on test set
        y_test_pred = np.dot(X_test, weights) + bias
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Log the parameters and results
        log_file.write(f"Learning Rate: {lr}, Iterations: {n_iterations}, Train MSE: {mse}, Test MSE: {test_mse}\n")
        print(f"Learning Rate: {lr}, Iterations: {n_iterations}, Train MSE: {mse}, Test MSE: {test_mse}")

        # Check if this is the best model so far
        if test_mse < best_mse:
            best_mse = test_mse
            best_params = (lr, n_iterations, weights, bias, error_history)

log_file.close()

# Extract the best model parameters and error history
best_lr, best_n_iterations, best_weights, best_bias, best_error_history = best_params

# Step 4: Final Evaluation Metrics and Plots
# Final predictions using best model
y_test_pred_best = np.dot(X_test, best_weights) + best_bias

# Evaluation Metrics
final_mse = mean_squared_error(y_test, y_test_pred_best)
final_r2 = r2_score(y_test, y_test_pred_best)
explained_var = explained_variance_score(y_test, y_test_pred_best)

print(f"\nBest Model Evaluation Metrics:")
print(f"Best Learning Rate: {best_lr}")
print(f"Best Iterations: {best_n_iterations}")
print(f"Final Test MSE: {final_mse}")
print(f"R² Value: {final_r2}")
print(f"Explained Variance: {explained_var}")
print(f"Weight Coefficients: {best_weights}")
print(f"Bias: {best_bias}")

# Calculate predictions on the test set using the best model
y_test_pred_best = np.dot(X_test, best_weights) + best_bias

# Calculate the error metrics
test_mse = mean_squared_error(y_test, y_test_pred_best)
test_r2 = r2_score(y_test, y_test_pred_best)
test_explained_var = explained_variance_score(y_test, y_test_pred_best)

# Print the results
print(f"\nTest MSE: {test_mse}")
print(f"Test R²: {test_r2}")
print(f"Test Explained Variance: {test_explained_var}")

# # Plotting MSE vs. Iterations for Best Model
plt.figure(figsize=(12, 6))
plt.plot(range(best_n_iterations), best_error_history, label='Training Error')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE vs. Iterations for Best Model')
plt.grid(True)
plt.legend()
plt.show()

# Plot the MSE vs Iterations for all models
plt.figure(figsize=(12, 8))
for (lr, n_iterations), error_history in error_histories.items():
    plt.plot(error_history, label=f"LR: {lr}, Iterations: {n_iterations}")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("MSE vs Iterations for Different Models")
plt.legend()
plt.show()

# Plotting Predictions vs. Actual for a Key Feature (e.g., Feature 0)
plt.figure(figsize=(12, 6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], y_test_pred_best, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Feature 0')
plt.ylabel('Target Variable')
plt.title('Actual vs. Predicted Values for Feature 0')
plt.grid(True)
plt.legend()
plt.show()

# Plotting Weight Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Weight'], bins=20, kde=True)
plt.title('Weight Distribution')
plt.xlabel('Weight (kg)')
plt.grid(True)
plt.show()

# Plotting residuals 
residuals = y_test - y_test_pred_best
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
