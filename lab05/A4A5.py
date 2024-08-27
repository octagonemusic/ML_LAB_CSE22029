import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can change the number of clusters
chloride_data['Cluster'] = kmeans.fit_predict(chloride_normalized)

# Calculate clustering metrics
silhouette_avg = silhouette_score(chloride_normalized, kmeans.labels_)
ch_score = calinski_harabasz_score(chloride_normalized, kmeans.labels_)
db_index = davies_bouldin_score(chloride_normalized, kmeans.labels_)

print(f'Silhouette Score: {silhouette_avg}')
print(f'Calinski-Harabasz Score: {ch_score}')
print(f'Davies-Bouldin Index: {db_index}')

# Visualize the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(chloride_data['Chloride (mg/L)'], np.zeros_like(chloride_data['Chloride (mg/L)']), 
                      c=chloride_data['Cluster'], cmap='viridis', marker='o')

# Add legend
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.xlabel('Chloride (mg/L)')
plt.title('K-means Clustering of Chloride (mg/L)')
plt.show()