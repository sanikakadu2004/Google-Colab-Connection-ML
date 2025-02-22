
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
dataset.head()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
# Selecting the last two columns (Ensure dataset has at least 4 columns)
x = dataset.iloc[:, [3, 4]].values
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results
plt.plot(range(1, 11), wcss, marker='o')  # Added marker for better visualization
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for K-Means")
plt.show()

# Fit the KMeans model and predict cluster labels
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(x)
# Plotting the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=60, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=60, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=60, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=60, c='violet', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=60, c='yellow', label='Cluster 5')
# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label =
'Centroids')
plt.xlabel('Annual Income (k$)') # Labels and Title
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters (K-Means)')
plt.legend()
plt.show()

# Visualizing clusters on a 2D plot (using 'Annual_Income_(k$)' vs. 'Spending_Score')
plt.figure(figsize=(8, 6))
plt.scatter(dataset['Annual Income (k$)'], dataset['Spending Score (1-100)'], c=y_kmeans, cmap='viridis')
plt.title('Customer Segments Based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.colorbar(label='Cluster')
plt.show()
