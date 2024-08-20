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

# Define acceptable and unacceptable classes based on chloride level
acceptable_class = grouped_df[grouped_df['Chloride (mg/L)'] <= 250]
unacceptable_class = grouped_df[grouped_df['Chloride (mg/L)'] > 250]

# Calculate the mean (centroid) for each class
acceptable_mean = acceptable_class['Chloride (mg/L)'].mean(axis=0)
unacceptable_mean = unacceptable_class['Chloride (mg/L)'].mean(axis=0)

# Calculate the spread (standard deviation) for each class
acceptable_std = acceptable_class['Chloride (mg/L)'].std(axis=0)
unacceptable_std = unacceptable_class['Chloride (mg/L)'].std(axis=0)

# Calculate the distance between the mean vectors of the classes
interclass_distance = np.linalg.norm(acceptable_mean - unacceptable_mean)

# Print the results
print(f"Acceptable Class Mean: {acceptable_mean}")
print(f"Unacceptable Class Mean: {unacceptable_mean}")
print(f"Acceptable Class Spread (Std Dev): {acceptable_std}")
print(f"Unacceptable Class Spread (Std Dev): {unacceptable_std}")
print(f"Interclass Distance: {interclass_distance}")

# Display the grouped DataFrame
print(grouped_df)

# Calculate the mean and variance for the chloride levels
chloride_mean = grouped_df['Chloride (mg/L)'].mean()
chloride_variance = grouped_df['Chloride (mg/L)'].var()

# Print the mean and variance
print(f"Chloride Mean: {chloride_mean}")
print(f"Chloride Variance: {chloride_variance}")

# Remove NaN values from the 'Chloride (mg/L)' column
chloride_values = grouped_df['Chloride (mg/L)'].dropna()

# Calculate the histogram data using numpy.histogram
hist, bin_edges = np.histogram(chloride_values, bins=10)

# Plot the histogram using matplotlib.pyplot.bar
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
plt.title('Histogram of Chloride Levels')
plt.xlabel('Chloride (mg/L)')
plt.ylabel('Frequency')
plt.show()

#MINKOWSKI DISTANCE
feature_vector_1 = chloride_values.iloc[0]
feature_vector_2 = chloride_values.iloc[25]

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