import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Loading the dataset into a dataframe
df = pd.read_csv('wine_dataset.csv' , sep=';')

# Dropping all columns apart from the 'alcohol' & 'color_intensity' columns
df = df[['alcohol', 'color_intensity']]  

# Visualizing the dataframe by viewing the first few rows of the data
print(df.head()) 

# Determining the optimal number of clusters (k) using the Elbow method
sse = []
K_range = range(1, 11)  # Testing k from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)  # SSE for each k

# Plotting the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(K_range, sse, marker='o')
plt.title('Optimal k Elbow Method ')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors ')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# Performing the K-means clustering with the optimal k
optimal_k = 3      # From the Elbow method, determine the optimal k (you can choose any number, I just went for 3 in this instance)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df)
df['cluster'] = kmeans.labels_

# Visualizing the clusters and centroids
plt.figure(figsize=(10, 8))

# Scatter plot for each cluster
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['alcohol'], cluster_data['color_intensity'], label=f'Cluster {cluster + 1}')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', color='black', s=150, label='Centroids')

plt.title('Clusters of Wine Data')
plt.xlabel('Alcohol')
plt.ylabel('Color Intensity')
plt.legend()
plt.grid(True)
plt.show()

