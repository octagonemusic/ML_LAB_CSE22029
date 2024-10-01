import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    # Convert relevant columns to numeric and drop NaN values in Sodium, Bicarbonate, Fluoride, and Chloride
    df['Sodium (mg/L)'] = pd.to_numeric(df['Sodium (mg/L)'], errors='coerce')
    df['Bicarbonate (mg/L)'] = pd.to_numeric(df['Bicarbonate (mg/L)'], errors='coerce')
    df['Fluoride (mg/L)'] = pd.to_numeric(df['Fluoride (mg/L)'], errors='coerce')
    df['Chloride (mg/L)'] = pd.to_numeric(df['Chloride (mg/L)'], errors='coerce')

    # Drop rows with NaN values in selected columns
    df = df.dropna(subset=['Sodium (mg/L)', 'Bicarbonate (mg/L)', 'Fluoride (mg/L)', 'Chloride (mg/L)'])
    
    # Encode the Quality labels to numeric
    le = LabelEncoder()
    df['Quality'] = le.fit_transform(df['Quality'])  # Convert string labels to numeric

    return df

def split_data(df):
    X = df[['Sodium (mg/L)', 'Bicarbonate (mg/L)', 'Fluoride (mg/L)', 'Chloride (mg/L)']]
    y = df['Quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

def main():
    file_path = 'labeled_data.xlsx'  # Update this with your actual file path
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
