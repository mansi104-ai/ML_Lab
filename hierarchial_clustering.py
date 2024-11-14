# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generating sample data
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

# Plotting dendrogram
plt.figure(figsize=(10, 7))
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

# Applying Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')  # Removed 'affinity' argument
y_hc = hc.fit_predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='rainbow')
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
