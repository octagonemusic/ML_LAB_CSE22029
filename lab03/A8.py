import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the data from the Excel sheet, specifying that the column names are in the 6th row (index 5)
file_path = 'groundwater_data.xlsx'
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

# Function to train and evaluate the kNN classifier for a given k
def evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Evaluate kNN for k=1 and k=3
accuracy_k1 = evaluate_knn(1)
accuracy_k3 = evaluate_knn(3)

print(f"Accuracy for k=1: {accuracy_k1}")
print(f"Accuracy for k=3: {accuracy_k3}")

# Vary k from 1 to 11 and store the accuracies
k_values = range(1, 12)
accuracies = [evaluate_knn(k) for k in k_values]

# Plot the accuracy against the values of k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. k in kNN Classifier')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()