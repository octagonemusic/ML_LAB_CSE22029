import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    # Convert relevant columns to numeric and drop NaN values in Sodium and Chloride
    df['Sodium (mg/L)'] = pd.to_numeric(df['Sodium (mg/L)'], errors='coerce')
    df['Chloride (mg/L)'] = pd.to_numeric(df['Chloride (mg/L)'], errors='coerce')
    
    # Drop rows with NaN values in selected columns
    df = df.dropna(subset=['Sodium (mg/L)', 'Chloride (mg/L)'])
    
    return df

def split_data(df):
    X = df[['Sodium (mg/L)']]  # Feature: Sodium
    y = df['Chloride (mg/L)']   # Target: Chloride
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    return y_test, y_pred

def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot for actual vs predicted values
    plt.scatter(y_test.index, y_test, label='Actual Chloride (mg/L)', color='blue', alpha=0.6, marker='o')
    plt.scatter(y_test.index, y_pred, label='Predicted Chloride (mg/L)', color='green', alpha=0.6, marker='x')

    # Draw lines between actual and predicted values
    for i in range(len(y_test)):
        plt.plot([y_test.index[i], y_test.index[i]], [y_test.iloc[i], y_pred[i]], color='gray', linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Chloride (mg/L)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    file_path = 'labeled_data.xlsx'  # Update this with your actual file path
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Training Decision Tree Regressor...")
    dt_model = train_decision_tree(X_train, y_train)
    y_test_dt, y_pred_dt = evaluate_model(dt_model, X_test, y_test)
    plot_actual_vs_predicted(y_test_dt, y_pred_dt, title="Decision Tree Regressor: Actual vs Predicted")

    print("\nTraining K-NN Regressor...")
    knn_model = train_knn(X_train, y_train)
    y_test_knn, y_pred_knn = evaluate_model(knn_model, X_test, y_test)
    plot_actual_vs_predicted(y_test_knn, y_pred_knn, title="K-NN Regressor: Actual vs Predicted")

if __name__ == "__main__":
    main()
