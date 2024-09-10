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

# Activation functions
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

# Function to calculate the weighted sum
def weighted_sum(inputs, weights):
    return np.dot(inputs, weights[1:]) + weights[0]

# Function to update the weights based on the error
def update_weights(weights, inputs, error, alpha):
    weights[1:] += alpha * error * inputs
    weights[0] += alpha * error

# Training process
def train_perceptron(activation_function, epochs=1000):
    global W0, W1, W2
    weights = np.array([W0, W1, W2])
    errors = []
    
    for epoch in range(epochs):
        total_error = 0
        
        for inputs, target in zip(training_inputs, expected_output):
            y = activation_function(weighted_sum(inputs, weights))
            error = target - y
            total_error += error ** 2
            update_weights(weights, inputs, error, alpha)
        
        errors.append(total_error)
        
        # If total error is 0, weights have converged
        if total_error == 0:
            print(f"Weights converged after {epoch+1} epochs with {activation_function.__name__}.")
            break
            
    return weights, errors, epoch+1

# Train the perceptron with different activation functions
activation_functions = [bipolar_step_function, sigmoid_function, relu_function]
results = {}

for activation_function in activation_functions:
    final_weights, errors, epochs = train_perceptron(activation_function)
    results[activation_function.__name__] = (final_weights, errors, epochs)

    # Plot epochs against sum-square-error
    plt.plot(range(1, epochs+1), errors, marker='o', label=activation_function.__name__)

plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Epochs vs Sum-Square Error for Different Activation Functions')
plt.legend()
plt.show()

# Print final weights and epochs for each activation function
for activation_function in activation_functions:
    name = activation_function.__name__
    final_weights, errors, epochs = results[name]
    print(f"Activation Function: {name}")
    print(f"Final weights: {final_weights}")
    print(f"Epochs to converge: {epochs}\n")    