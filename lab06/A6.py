import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (used for weight update)
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to train the perceptron and track sum squared error
def train_perceptron(X, y, learning_rate=0.01, epochs=10000):
    # Initialize weights with smaller random values
    np.random.seed(1)
    weights = np.random.rand(X.shape[1], 1) * 0.01  # Smaller weights
    bias = np.random.rand(1) * 0.01  # Smaller bias

    sse_list = []  # To store sum squared errors at each epoch

    for epoch in range(epochs):
        # Forward propagation (simple perceptron)
        linear_output = np.dot(X, weights) + bias
        predicted_output = sigmoid(linear_output)

        # Calculate error (difference between expected and predicted output)
        error = y - predicted_output

        # Sum of squared error (SSE)
        sse = np.sum(np.square(error))
        sse_list.append(sse)

        # Update weights and bias using the error
        adjustments = error * sigmoid_derivative(predicted_output)
        weights += np.dot(X.T, adjustments) * learning_rate
        bias += np.sum(adjustments) * learning_rate

        # Print the SSE at every 1000 epochs to track learning progress
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: SSE = {sse}')
    
    return weights, bias, sse_list

# Function to predict with trained perceptron
def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    predicted_output = sigmoid(linear_output)
    return np.where(predicted_output >= 0.5, 1, 0)  # Thresholding at 0.5

# Prepare the customer data
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

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Main function to train the perceptron, make predictions, and plot error graph
def main():
    # Prepare the data
    X, y = prepare_data()

    # Train the perceptron and get sum squared error (SSE) list
    learning_rate = 0.01  # Continue with a smaller learning rate
    epochs = 10000
    weights, bias, sse_list = train_perceptron(X, y, learning_rate, epochs)

    # Predict on the training data
    predictions = predict(X, weights, bias)
    print("Predictions:\n", predictions)

    # Plot the graph of epoch vs sum squared error (SSE)
    plt.plot(range(epochs), sse_list)
    plt.title('Epoch vs Sum Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('Sum Squared Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
