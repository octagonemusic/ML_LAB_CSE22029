import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prepare the customer data (features and labels)
def prepare_data():
    # Features: Candies, Mangoes, Milk Packets, Payment
    X = np.array([[20, 6, 2, 386],
                  [16, 3, 6, 289],
                  [27, 6, 2, 393],
                  [19, 1, 2, 110],
                  [24, 4, 2, 280],
                  [22, 1, 5, 167],
                  [15, 4, 2, 271],
                  [18, 4, 2, 274],
                  [21, 1, 4, 148],
                  [16, 2, 4, 198]])

    # Labels: 1 for "Yes" (High value), 0 for "No"
    y = np.array([[1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [1],
                  [0],
                  [0]])

    # Scale the features using MinMaxScaler to normalize the values between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Perceptron learning (gradient descent)
def train_perceptron(X, y, learning_rate, epochs):
    # Initialize weights randomly and bias to 0
    weights = np.random.rand(X.shape[1], 1)
    bias = 0
    
    errors = []  # To store sum of squared errors at each epoch
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(X.shape[0]):
            # Linear combination
            linear_output = np.dot(X[i], weights) + bias
            # Apply sigmoid activation
            prediction = sigmoid(linear_output)
            # Calculate error
            error = y[i] - prediction
            # Update weights and bias using gradient descent
            weights += learning_rate * error * X[i].reshape(-1, 1)
            bias += learning_rate * error
            # Sum of squared errors
            total_error += error ** 2
            
        errors.append(total_error)
    
    return weights, bias, errors

# Function to make predictions using perceptron weights
def predict_perceptron(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    predicted_output = sigmoid(linear_output)
    return np.where(predicted_output >= 0.5, 1, 0)

# Pseudo-inverse method to calculate weights
def pseudo_inverse_method(X, y):
    # Adding bias term to input features
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calculate pseudo-inverse of X
    pseudo_inverse = np.linalg.pinv(X_with_bias)
    
    # Compute weights using the pseudo-inverse formula
    weights_pseudo = np.dot(pseudo_inverse, y)
    
    return weights_pseudo

# Function to make predictions using weights obtained from pseudo-inverse
def predict_pseudo(X, weights):
    # Adding bias term to input features
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Linear combination of inputs and weights
    linear_output = np.dot(X_with_bias, weights)
    
    # Apply sigmoid to get output in the range [0, 1]
    predicted_output = sigmoid(linear_output)
    
    # Convert predictions to binary values (0 or 1)
    return np.where(predicted_output >= 0.5, 1, 0)

# Main function to compare perceptron learning and pseudo-inverse methods
def compare_methods():
    # Prepare the data
    X, y = prepare_data()

    # 1. Perceptron learning method
    learning_rate = 0.05
    epochs = 10000
    weights_perceptron, bias_perceptron, errors = train_perceptron(X, y, learning_rate, epochs)
    
    # Predict with perceptron weights
    predictions_perceptron = predict_perceptron(X, weights_perceptron, bias_perceptron)
    print("Predictions from Perceptron Learning:\n", predictions_perceptron)
    
    # 2. Pseudo-inverse method
    weights_pseudo = pseudo_inverse_method(X, y)
    print("\nWeights from Pseudo-Inverse Method:\n", weights_pseudo)
    
    # Predict with pseudo-inverse weights
    predictions_pseudo = predict_pseudo(X, weights_pseudo)
    print("Predictions from Pseudo-Inverse Method:\n", predictions_pseudo)

    # Compare predictions
    print("\n--- Comparison ---")
    print("Perceptron Predictions: ", predictions_perceptron.ravel())
    print("Pseudo-Inverse Predictions: ", predictions_pseudo.ravel())
    
    # Calculate errors for both methods
    error_perceptron = np.sum(np.abs(predictions_perceptron - y))
    error_pseudo = np.sum(np.abs(predictions_pseudo - y))
    print(f"Perceptron Learning Error: {error_perceptron}")
    print(f"Pseudo-Inverse Error: {error_pseudo}")
    
    # Plot the sum of squared errors vs epochs for the perceptron learning
    plt.plot(range(epochs), errors, label="Sum of Squared Errors")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of Squared Errors")
    plt.title("Epoch vs Sum of Squared Errors (Perceptron Learning)")
    plt.show()

# Run the comparison
compare_methods()
