import pandas as pd
from sklearn.model_selection import train_test_split

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

# Print the shapes of the train and test sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")  
