import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to load data
def load_data(file_path):
    return pd.read_excel(file_path, header=5)

# Function to preprocess data
def preprocess_data(df):
    # Select the relevant features
    selected_columns = df[['Chloride (mg/L)', 'Sodium (mg/L)']].copy()
    
    # Convert non-numeric values to NaN and remove rows with NaN values
    selected_columns['Chloride (mg/L)'] = pd.to_numeric(selected_columns['Chloride (mg/L)'], errors='coerce')
    selected_columns['Sodium (mg/L)'] = pd.to_numeric(selected_columns['Sodium (mg/L)'], errors='coerce')
    selected_columns = selected_columns.dropna()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(selected_columns))
    
    # Filter out the outliers
    threshold = 3
    selected_columns = selected_columns[(z_scores < threshold).all(axis=1)]
    
    return selected_columns

# Function to split data into training and testing sets
def split_data(selected_columns):
    X = selected_columns[['Chloride (mg/L)']].values  # Feature
    y = selected_columns['Sodium (mg/L)'].values      # Target variable
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train the Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    epsilon = 1e-10
    mape = np.mean(np.abs((y - y_pred) / (y + epsilon))) * 100
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2

# Function to print evaluation metrics
def print_metrics(mse, rmse, mape, r2, data_type='Training'):
    print(f"{data_type} Metrics:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
    print(f"R-squared: {r2}\n")

# Function to plot results
def plot_results(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', label='Actual')
    plt.scatter(X_train, y_train_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Chloride (mg/L)')
    plt.ylabel('Sodium (mg/L)')
    plt.title('Training Set: Chloride vs Sodium')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_test_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Chloride (mg/L)')
    plt.ylabel('Sodium (mg/L)')
    plt.title('Test Set: Chloride vs Sodium')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to orchestrate the workflow
def main():
    file_path = 'groundwater_data.xlsx'  # Update this with your actual file path
    df = load_data(file_path)
    selected_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(selected_columns)

    model = train_model(X_train, y_train)

    # Evaluate on training set
    mse_train, rmse_train, mape_train, r2_train = evaluate_model(model, X_train, y_train)
    print_metrics(mse_train, rmse_train, mape_train, r2_train, data_type='Training')

    # Evaluate on test set
    mse_test, rmse_test, mape_test, r2_test = evaluate_model(model, X_test, y_test)
    print_metrics(mse_test, rmse_test, mape_test, r2_test, data_type='Test')

    # Plotting results
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plot_results(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)

if __name__ == "__main__":
    main()
