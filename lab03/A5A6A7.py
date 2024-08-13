import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the Excel sheet, specifying that the column names are in the 6th row (index 5)
file_path = 'lab03/groundwater_data.xlsx'
df = pd.read_excel(file_path, header=5)

# Select only the "District Name" and "Chloride (mg/L)" columns
selected_columns = df[['District Name', 'Chloride (mg/L)']]

# Group by district and calculate the mean chloride level for each district
grouped_df = selected_columns.groupby('District Name').mean().reset_index()

# Remove NaN values from the 'Chloride (mg/L)' column
chloride_values = grouped_df['Chloride (mg/L)'].dropna()

# Define feature vectors (X) and class labels (y)
X = chloride_values.values.reshape(-1, 1)  # Reshape for sklearn compatibility
threshold = 250  # Example threshold for binary classification
y = (chloride_values > threshold).astype(int).values  # Binary labels based on threshold

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Study the prediction behavior
print("Predicted labels for the test set:")
print(y_pred)

print("Actual labels for the test set:")
print(y_test)

# Perform classification for a given vector
test_vect = X_test[0].reshape(1, -1)  # Select the first test vector and reshape it for prediction
predicted_class = knn.predict(test_vect)

# Print the predicted class for the selected test vector
print(f"Test vector: {test_vect[0][0]}, Predicted class: {predicted_class[0]}, Actual class: {y_test[0]}")   
