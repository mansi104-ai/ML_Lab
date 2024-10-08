import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Euclidean distance calculation
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# KNN class
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    # Fit method (simply memorizes the training data)
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Predict method
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]  # Corrected method call to _predict
        return np.array(predictions)

    # Predict a single instance
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote, most common class label among the k neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage with a dataset
if __name__ == "__main__":
    # Example dataset (you can replace it with any dataset)
    from sklearn.datasets import load_wine

    # Load dataset (Wine dataset)
    data = load_wine()
    X, y = data.data, data.target

    # Split the dataset into 70:30 train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train KNN model
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
