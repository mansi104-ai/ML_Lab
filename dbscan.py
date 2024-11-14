import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs

class DBSCANClustering:
    def __init__(self, X, epsilon=0.5, min_samples=5):
        self.epsilon = epsilon  # Maximum distance between two samples to be considered neighbors
        self.min_samples = min_samples  # Minimum number of samples in a neighborhood to form a core point
        self.num_examples = X.shape[0]
        self.plot_figure = True  # Option to plot the clusters
        self.labels = -np.ones(self.num_examples, dtype=int)  # Initial label assignments (-1 for noise)

    def region_query(self, X, point_idx):
        # Find neighbors within epsilon distance of point
        neighbors = []
        for idx, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) < self.epsilon:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(self, X, point_idx, neighbors, cluster_id):
        # Label initial point
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels[neighbor_idx] == -1:  # Label previously considered noise as part of cluster
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == 0:  # Expand to unvisited point
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self.region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1

    def fit(self, X):
        cluster_id = 0
        for point_idx in range(self.num_examples):
            if self.labels[point_idx] != -1:  # Skip if already assigned
                continue
            neighbors = self.region_query(X, point_idx)
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.expand_cluster(X, point_idx, neighbors, cluster_id)
        
        if self.plot_figure:
            self.plot_clusters(X)
        
        return self.labels

    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        unique_labels = set(self.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Black color for noise points
                color = "k"
            cluster_points = X[self.labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=color, label=f'Cluster {label}')
        
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.title("DBSCAN Clustering")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generating sample data with make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=0.5, random_state=42)
    
    # Initializing and fitting the DBSCAN model
    epsilon = 0.5
    min_samples = 5
    dbscan = DBSCANClustering(X, epsilon=epsilon, min_samples=min_samples)
    labels = dbscan.fit(X)
