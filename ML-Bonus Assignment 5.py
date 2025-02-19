
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
dataset.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Preprocessing the data: Scaling the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset[['Annual Income (k$)', 'Spending Score (1-100)']])

# Plot Elbow Graph for K-Means and K-Distance Graph for DBSCAN
# Changing the column indices to 3 and 4 to select the last two columns.
x = dataset.iloc[:, [3, 4]].values
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    # Indented block for the 'for' loop starts here
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    # The variable 'x' was defined earlier and holds the data for clustering.
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    # Indented block ends here

# K-Distance Graph for DBSCAN (To find optimal eps)**
# Nearest neighbors for DBSCAN (k=4 or 5 is common)
neigh = NearestNeighbors(n_neighbors=4)
neighbors = neigh.fit(scaled_data)
distances, indices = neighbors.kneighbors(scaled_data)
# Sorting distances (K-Distances) for plotting
distances = np.sort(distances[:, 3], axis=0)  # Take the distance of the 4th nearest neighbor (index 3)

# Plot side by side
plt.figure(figsize=(14, 6))

# Plotting the results
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss,marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")

# Plotting the K-Distance graph
plt.subplot(1, 2, 2)
plt.plot(distances)
plt.title('K-Distance Graph (4th Nearest Neighbor)')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to 4th Nearest Neighbor')
plt.show()

# Step 3: Apply Clustering and Plot Results
import matplotlib.colors as mcolors
# K-Means Clustering
optimal_k = 4  # Based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

# DBSCAN Clustering (Selecting eps based on K-Distance Graph)
# Adjust eps and min_samples values
dbscan = DBSCAN(eps=0.3, min_samples=4)  # Adjusted parameters
dbscan_labels = dbscan.fit_predict(scaled_data)

# Compute silhouette scores
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
# Check if more than one cluster is found before calculating silhouette score
dbscan_silhouette = silhouette_score(scaled_data[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if len(set(dbscan_labels)) > 1 else -1

# Define distinct colors
colors = list(mcolors.TABLEAU_COLORS.values())

plt.figure(figsize=(12, 5))

# K-Means clustering plot with distinct colors
plt.subplot(1, 2, 1)
for i in range(optimal_k):
    plt.scatter(scaled_data[kmeans_labels == i, 0], scaled_data[kmeans_labels == i, 1],
                color=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids')  # Keep centroids in black
plt.title(f'K-Means Clustering (Silhouette Score: {kmeans_silhouette:.2f})')
plt.legend()

# DBSCAN clustering plot with distinct colors
plt.subplot(1, 2, 2)
unique_labels = set(dbscan_labels)
for label in unique_labels:
    if label == -1:
        color = 'black'  # Black for noise
        label_name = 'Noise'
    else:
        color = colors[label % len(colors)]  # Assign distinct colors
        label_name = f'Cluster {label}'
    plt.scatter(scaled_data[dbscan_labels == label, 0], scaled_data[dbscan_labels == label, 1],
                color=color, label=label_name, alpha=0.7)  # No edge color

plt.title(f'DBSCAN Clustering (Silhouette Score: {dbscan_silhouette:.2f})')
plt.legend()
plt.show()

# Step 4: Provide Recommendations
print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
print("
Analysis:")
print("- K-Means performs well when clusters are compact and well-separated.")
print("- DBSCAN is useful when clusters have irregular shapes and there is noise (outliers).")
print("
Comparison:")
if kmeans_silhouette > dbscan_silhouette:
    print("- K-Means forms clearer and more defined clusters, making it better for customer segmentation.")
    print("- DBSCAN identifies noise but may struggle with well-separated, compact clusters.")
else:
    print("- DBSCAN performs better when the data has varying densities or significant noise.")
    print("- K-Means may not perform well if the clusters are not well-separated.")

print("
Recommendation for Retail Marketing:")
if kmeans_silhouette > dbscan_silhouette:
    print("- K-Means is recommended if the business needs well-defined customer groups (e.g., budget vs. premium customers).")
else:
    print("- DBSCAN is better suited for identifying unusual shopping behaviors and outliers, which can help detect niche customer segments.")
