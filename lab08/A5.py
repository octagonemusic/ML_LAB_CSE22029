import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data function
def load_data(file_path, columns_to_import):
    return pd.read_excel(file_path, usecols=columns_to_import)

# Preprocess data function
def preprocess_data(data):
    # Convert Quality labels to numeric values
    data['Quality'] = data['Quality'].map({'Safe': 1, 'Unsafe': 0})
    
    # Separate features and target variable
    X = data.drop('Quality', axis=1)
    y = data['Quality']
    return X, y

# Train Naive Bayes model function
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Evaluate model function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

# Main function to run the entire process
def main():
    # Define file path and columns to import
    file_path = 'labeled_data.xlsx'
    columns_to_import = ['Sodium (mg/L)', 'Bicarbonate (mg/L)', 'Fluoride (mg/L)', 'Chloride (mg/L)', 'Quality']
    
    # Load the data
    data = load_data(file_path, columns_to_import)
    
    # Drop rows with missing values
    data.dropna(inplace=True)  # Removes rows with any NaN values
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Naive Bayes model
    model = train_naive_bayes(X_train, y_train)
    
    # Evaluate the model and get predictions
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Optionally, display the original predictions (Safe/Unsafe)
    y_pred_original = pd.Series(y_pred).map({1: 'Safe', 0: 'Unsafe'})
    print("Predicted Quality:", y_pred_original.values)

# Run the main function
if __name__ == "__main__":
    main()
