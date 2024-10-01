import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def create_sample_data():
    """Create sample data for linear regression."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 3, 5, 7, 11])
    return X, y

def initialize_model(fit_intercept=True):
    """Initialize the Linear Regression model."""
    return LinearRegression(fit_intercept=fit_intercept)

def train_model(model, X, y):
    """Fit the model to the training data."""
    model.fit(X, y)
    return model

def make_predictions(model, X):
    """Make predictions using the fitted model."""
    return model.predict(X)

def evaluate_model(model, X, y):
    """Evaluate the model's performance."""
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r_squared = r2_score(y, predictions)
    return mse, r_squared

def print_model_attributes(model):
    """Print model attributes."""
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def main():
    """Main function to run the linear regression workflow."""
    # Create sample data
    X, y = create_sample_data()
    
    # Initialize and train the model
    model = initialize_model()
    model = train_model(model, X, y)
    
    # Make predictions
    predictions = make_predictions(model, X)

    # Evaluate the model
    mse, r_squared = evaluate_model(model, X, y)
    
    # Print model attributes and evaluation results
    print_model_attributes(model)
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)

if __name__ == "__main__":
    main()
