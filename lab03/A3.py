import pandas as pd
import numpy as np
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

# Select two feature vectors (for simplicity, we take the first two non-NaN values)
feature_vector_1 = chloride_values.iloc[0]
feature_vector_2 = chloride_values.iloc[1]

# Function to calculate Minkowski distance
def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Calculate Minkowski distances for r from 1 to 10
r_values = range(1, 11)
distances = [minkowski_distance(feature_vector_1, feature_vector_2, r) for r in r_values]

# Plot the distances
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.xlabel('r')
plt.ylabel('Distance')
plt.xticks(r_values)
plt.grid(True)
plt.show()