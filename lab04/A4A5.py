import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Set the random seed for reproducibility
np.random.seed(42)

# Generate 20 random data points for X and Y between 1 and 10
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)

# Assign classes based on a simple rule: if X + Y > 10, class1 (Red), else class0 (Blue)
classes_train = np.where(X_train + Y_train > 10, 1, 0)

# Create the kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
X_train_reshaped = np.column_stack((X_train, Y_train))
knn.fit(X_train_reshaped, classes_train)

# Define the range of k values to test
k_values = [1, 3, 5, 7, 9]

# Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
X_test_flat = X_test.ravel()
Y_test_flat = Y_test.ravel()
X_test_reshaped = np.column_stack((X_test_flat, Y_test_flat))

# Loop through each k value
for k in k_values:
    # Create the kNN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier
    X_train_reshaped = np.column_stack((X_train, Y_train))
    knn.fit(X_train_reshaped, classes_train)
    
    # Classify the test set data
    predicted_classes = knn.predict(X_test_reshaped)
    
    # Create a scatter plot of the test data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_flat[predicted_classes == 0], Y_test_flat[predicted_classes == 0], color='blue', label='Class 0', alpha=0.5)
    plt.scatter(X_test_flat[predicted_classes == 1], Y_test_flat[predicted_classes == 1], color='red', label='Class 1', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title(f'Scatter Plot of Test Data with kNN Classification (k={k})')
    plt.legend()
    plt.grid(True)
    plt.show()