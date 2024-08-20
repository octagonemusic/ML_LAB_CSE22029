import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
file_path = 'groundwater_data.xlsx'  # Update this with your actual file path
df = pd.read_excel(file_path, header=5)

# Select the relevant features: Chloride (mg/L) and pH
selected_columns = df[['Chloride (mg/L)', 'pH']]

# Remove rows with missing values in the selected columns
selected_columns = selected_columns.dropna()

# Define the class labels based on the criteria:
# Safe if Chloride <= 250 mg/L and 6.5 <= pH <= 8.5
safe_condition = (selected_columns['Chloride (mg/L)'] <= 250) & (selected_columns['pH'] >= 6.5) & (selected_columns['pH'] <= 8.5)
y = safe_condition.astype(int).values  # 1 for safe, 0 for unsafe

# Define the feature vectors (X)
X = selected_columns.values  # Chloride and pH are the features

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for 'k'
param_grid = {'n_neighbors': list(range(1, 31))}

# Initialize the kNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best 'k' value
best_k = grid_search.best_params_['n_neighbors']
print(f"The best 'k' value is: {best_k}")

# Train the kNN classifier with the best 'k' value
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_test = best_knn.predict(X_test)
y_pred_train = best_knn.predict(X_train)

# Evaluate the classifier
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
report_test = classification_report(y_test, y_pred_test)
report_train = classification_report(y_train, y_pred_train)

# Calculate confusion matrix for test and train data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

print("Test Accuracy:", accuracy_test)
print("Train Accuracy:", accuracy_train)
print("Test Classification Report:\n", report_test)
print("Train Classification Report:\n", report_train)
print("Test Confusion Matrix:\n", conf_matrix_test)
print("Train Confusion Matrix:\n", conf_matrix_train)

# Plotting the scatter plots for training and test data side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Train data plot
axes[0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='green', label='Safe (Train)')
axes[0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Unsafe (Train)')
axes[0].set_xlabel('Chloride (mg/L)')
axes[0].set_ylabel('pH')
axes[0].set_title('Train Data')
axes[0].legend()
axes[0].set_xlim(0, 8000)  # Set x-axis limits for train data
axes[0].set_ylim(3, 10)    # Set y-axis limits for train data

# Test data plot
axes[1].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='green', label='Safe (Test)')
axes[1].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Unsafe (Test)')
axes[1].set_xlabel('Chloride (mg/L)')
axes[1].set_ylabel('pH')
axes[1].set_title('Test Data')
axes[1].legend()
axes[1].set_xlim(0, 8000)  # Set x-axis limits for test data
axes[1].set_ylim(3, 10)    # Set y-axis limits for test data

plt.tight_layout()
plt.show()