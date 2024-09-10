import numpy as np
import matplotlib.pyplot as plt

# Initialize weights
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# XOR inputs and expected outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 1, 1, 0])

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to calculate the output of perceptron
def perceptron_output(input_data, W0, W1, W2):
    return step_activation(W0 + W1 * input_data[0] + W2 * input_data[1])

# Training the perceptron
def train_perceptron(inputs, expected_output, W0, W1, W2, learning_rate, epochs=1000):
    errors_per_epoch = []
    
    for epoch in range(epochs):
        sum_squared_error = 0
        for i in range(len(inputs)):
            # Calculate the perceptron output
            output = perceptron_output(inputs[i], W0, W1, W2)
            
            # Calculate the error
            error = expected_output[i] - output
            sum_squared_error += error ** 2
            
            # Update the weights
            W0 += learning_rate * error
            W1 += learning_rate * error * inputs[i][0]
            W2 += learning_rate * error * inputs[i][1]
        
        errors_per_epoch.append(sum_squared_error)
        
    return W0, W1, W2, errors_per_epoch

# Train the perceptron
W0_final, W1_final, W2_final, errors_per_epoch = train_perceptron(
    inputs, expected_output, W0, W1, W2, learning_rate)

# Plotting the sum-squared errors over epochs
plt.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Epochs vs Error for XOR Gate SLP')
plt.show()

print(f"Final Weights: W0 = {W0_final}, W1 = {W1_final}, W2 = {W2_final}")
print("Final outputs after training:")
for inp in inputs:
    print(f"Input: {inp}, Output: {perceptron_output(inp, W0_final, W1_final, W2_final)}")
