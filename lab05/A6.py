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

# Initialize lists to store the scores
silhouette_scores = []
ch_scores = []
db_indices = []
k_values = range(2, 20)  # You can adjust the range of k values

# Perform K-means clustering for different values of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(chloride_normalized)
    
    silhouette_avg = silhouette_score(chloride_normalized, labels)
    ch_score = calinski_harabasz_score(chloride_normalized, labels)
    db_index = davies_bouldin_score(chloride_normalized, labels)
    
    silhouette_scores.append(silhouette_avg)    
    ch_scores.append(ch_score)
    db_indices.append(db_index)

# Plot the scores against the k values
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title('Calinski-Harabasz Score vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Calinski-Harabasz Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, db_indices, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()