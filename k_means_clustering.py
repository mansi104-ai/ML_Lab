import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters  # Number of clusters
        self.max_iterations = 100  # Maximum number of iterations
        self.num_examples, self.num_features = X.shape
        self.plot_figure = True  # Option to plot the clusters

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids
    
    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
            clusters[closest_centroid].append(point_idx)
        return clusters
    
    def update_centroids(self, X, clusters):
        centroids = np.zeros((self.K, self.num_features))
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:  # Avoid division by zero
                new_centroid = np.mean(X[cluster], axis=0)
                centroids[cluster_idx] = new_centroid
        return centroids
    
    def predict_cluster(self, X, centroids):
        predictions = np.zeros(X.shape[0])
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
            predictions[point_idx] = closest_centroid
        return predictions
    
    def fit(self, X):
        centroids = self.initialize_random_centroids(X)
        for _ in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self.update_centroids(X, clusters)
            
            if np.all(previous_centroids == centroids):
                break
        
        if self.plot_figure:
            self.plot_clusters(X, clusters, centroids)
        
        return centroids, clusters
    
    def plot_clusters(self, X, clusters, centroids):
        plt.figure(figsize=(8, 6))
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        
        for cluster_idx, cluster in enumerate(clusters):
            cluster_points = X[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx+1}')
        
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.title("K-Means Clustering")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generating sample data
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    # Initializing and fitting the KMeans model
    num_clusters = 4
    kmeans = KMeansClustering(X, num_clusters)
    centroids, clusters = kmeans.fit(X)
