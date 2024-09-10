import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
initial_W0 = 10
initial_W1 = 0.2
initial_W2 = -0.75

# Define the AND gate input and expected output
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 0, 0, 1])

# Activation function (Step function)
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
def train_perceptron(alpha, epochs=100):
    weights = np.array([initial_W0, initial_W1, initial_W2])
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
            print(f"Weights converged after {epoch+1} epochs with learning rate {alpha}.")
            break
            
    return weights, errors, epoch+1

# Learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
iterations_to_converge = []

# Train the perceptron with different learning rates
for alpha in learning_rates:
    _, _, epochs = train_perceptron(alpha)
    iterations_to_converge.append(epochs)

# Plot learning rates against iterations to converge
plt.plot(learning_rates, iterations_to_converge, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations to Converge')
plt.title('Learning Rate vs Iterations to Converge')
plt.show()

# Print the results
for alpha, epochs in zip(learning_rates, iterations_to_converge):
    print(f"Learning Rate: {alpha}, Iterations to Converge: {epochs}")