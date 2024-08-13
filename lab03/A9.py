import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

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
knn = KNeighborsClassifier(n_neighbors=2)

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_test = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

# Evaluate the classifier
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
report_test = classification_report(y_test, y_pred_test)
report_train = classification_report(y_train, y_pred_train)

# Calculate confusion matrix for test and train data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

# Calculate precision, recall, and F1-score for test and train data
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)

# Print the evaluation results
print(f"Test Accuracy: {accuracy_test}")
print("Test Classification Report:")
print(report_test)
print("Test Confusion Matrix:")
print(conf_matrix_test)

print(f"Train Accuracy: {accuracy_train}")
print("Train Classification Report:")
print(report_train)
print("Train Confusion Matrix:")
print(conf_matrix_train)

print(f"Test Precision: {precision_test}")
print(f"Test Recall: {recall_test}")
print(f"Test F1-Score: {f1_test}")

print(f"Train Precision: {precision_train}")
print(f"Train Recall: {recall_train}")
print(f"Train F1-Score: {f1_train}")

# Infer the model's learning outcome
if accuracy_train > accuracy_test:
    print("The model might be overfitting.")
elif accuracy_train < accuracy_test:
    print("The model might be underfitting.")
else:
    print("The model seems to be well-fitted.")