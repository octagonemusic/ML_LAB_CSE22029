import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
file_path = 'groundwater_data.xlsx'  # Update this with your actual file path
df = pd.read_excel(file_path, header=5)

# Select the relevant features: Chloride (mg/L) and Sodium (mg/L)
selected_columns = df[['Chloride (mg/L)', 'Sodium (mg/L)']].copy()

# Convert non-numeric values to NaN and remove rows with NaN values
selected_columns['Chloride (mg/L)'] = pd.to_numeric(selected_columns['Chloride (mg/L)'], errors='coerce')
selected_columns['Sodium (mg/L)'] = pd.to_numeric(selected_columns['Sodium (mg/L)'], errors='coerce')
selected_columns = selected_columns.dropna()

# Calculate Z-scores
z_scores = np.abs(stats.zscore(selected_columns))

# Define a threshold for Z-score
threshold = 3

# Filter out the outliers
selected_columns = selected_columns[(z_scores < threshold).all(axis=1)]

# Define the feature vectors (X) and target variable (y)
X = selected_columns[['Chloride (mg/L)']].values  # Chloride is the feature
y = selected_columns['Sodium (mg/L)'].values      # Sodium is the target variable

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict the target variable for the training set
y_train_pred = reg.predict(X_train)

# Predict the target variable for the test set
y_test_pred = reg.predict(X_test)

# Define a small epsilon value to avoid division by zero
epsilon = 1e-10

# Evaluate the model on the training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / (y_train + epsilon))) * 100
r2_train = r2_score(y_train, y_train_pred)

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / (y_test + epsilon))) * 100
r2_test = r2_score(y_test, y_test_pred)

print("Training Metrics:")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R-squared:", r2_train)

print("\nTest Metrics:")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R-squared:", r2_test)

# Plotting the results for the training set
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.scatter(X_train, y_train_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Chloride (mg/L)')
plt.ylabel('Sodium (mg/L)')
plt.title('Training Set: Chloride vs Sodium')
plt.legend()

# Plotting the results for the test set
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_test_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Chloride (mg/L)')
plt.ylabel('Sodium (mg/L)')
plt.title('Test Set: Chloride vs Sodium')
plt.legend()

plt.tight_layout()
plt.show()