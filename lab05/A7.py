import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = 'groundwater_data.xlsx'  # Update this with your actual file path
df = pd.read_excel(file_path, header=5)

# Select the "Chloride (mg/L)" column
chloride_data = df[['Chloride (mg/L)']].copy()

# Convert non-numeric values to NaN and remove rows with NaN values
chloride_data['Chloride (mg/L)'] = pd.to_numeric(chloride_data['Chloride (mg/L)'], errors='coerce')
chloride_data = chloride_data.dropna()

# Normalize the data
scaler = StandardScaler()
chloride_normalized = scaler.fit_transform(chloride_data)

# Initialize list to store the distortions
distortions = []
k_values = range(2, 10)  # You can adjust the range of k values

# Perform K-means clustering for different values of k and calculate distortions
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(chloride_normalized)
    distortions.append(kmeans.inertia_)

# Plot the distortions against the k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, distortions, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.show()