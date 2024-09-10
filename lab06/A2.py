import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
alpha = 0.05

# Define the AND gate input and expected output
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 0, 0, 1])

def step_function(x):
    return 1 if x >= 0 else 0

# Function to calculate the weighted sum
def weighted_sum(inputs, weights):
    return np.dot(inputs, weights[1:]) + weights[0]

# Function to update the weights based on the error
def update_weights(weights, inputs, error, alpha):
    weights[1:] += alpha * error * inputs
    weights[0] += alpha * error

# Training process
def train_perceptron(epochs):
    global W0, W1, W2
    weights = np.array([W0, W1, W2])
    errors = []
    
    for epoch in range(epochs):
        total_error = 0
        
        for inputs, target in zip(training_inputs, expected_output):
            y = step_function(weighted_sum(inputs, weights))
            error = target - y
            total_error += error ** 2
            update_weights(weights, inputs, error, alpha)
        
        errors.append(total_error)

        # If total error is 0, weights have converged
        if total_error == 0:
            print(f"Weights converged after {epoch+1} epochs.")
            break
            
    return weights, errors, epoch+1

# Train the perceptron
final_weights, errors, epochs = train_perceptron(500)  # Max 100 epochs

# Plot epochs against sum-square-error
plt.plot(range(1, epochs+1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Epochs vs Sum-Square Error')
plt.show()

print(f"Final weights: {final_weights}")

