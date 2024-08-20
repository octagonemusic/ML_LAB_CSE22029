import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate 20 random data points for X and Y between 1 and 10
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20) 

# Assign classes based on a simple rule: if X + Y > 10, class1 (Red), else class0 (Blue)
classes = np.where(X + Y > 10, 1, 0)

print(X)
print(Y)
print(classes)

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot class 0 points
plt.scatter(X[classes == 0], Y[classes == 0], color='blue', label='Class 0')

# Plot class 1 points
plt.scatter(X[classes == 1], Y[classes == 1], color='red', label='Class 1')

# Add labels and title
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot of Training Data')
plt.legend()
plt.grid(True)
plt.show()