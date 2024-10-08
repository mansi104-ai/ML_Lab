import numpy as np
from scipy.spatial.distance import pdist, squareform
import cvxopt
import matplotlib.pyplot as plt


class Dual_SVM:
    def __init__(self, kernel="rbf", degree=2, gamma=10.0, C=1.0, coeff=0.0):
        # Initialize hyperparameters
        self.C = C  # Regularization parameter
        self.degree = degree  # Degree for polynomial kernel
        self.gamma = gamma  # Gamma for RBF kernel
        self.kernel = kernel  # Kernel type: "linear", "poly", "rbf"
        self.coeff = coeff  # Coefficient for polynomial kernel
        self.tolerance = 1e-5  # Tolerance for identifying support vectors

        # Variables to be initialized later
        self.sv_X = None  # Support vector feature set
        self.sv_y = None  # Support vector labels
        self.sv_idx = None  # Indices of support vectors
        self.W = None  # Weight vector
        self.b = None  # Intercept (bias term)
        self.alphas = None  # Lagrange multipliers (dual coefficients)

    def linear(self, X1, X2):
        return X1 @ X2 + 0.0
    
    def poly(self, X1, X2):
        return (X1 @ X2 + self.coeff) ** self.degree
    
    def rbf(self, X1, X2):
        distances = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2) ** 2
        return np.exp(-self.gamma * distances)
    
    def extract_SVs(self, X, y, alphas):
        sv = alphas > self.tolerance
        ind = np.arange(len(alphas))[sv]
        alphas = alphas[sv]
        sv_X = X[sv]
        sv_y = y[sv]
        return alphas, sv, sv_X, sv_y, ind

    def fit(self, X, y):
        n, m = X.shape

        # Compute the kernel matrix based on the chosen kernel
        if self.kernel == "linear":
            K = self.linear(X, X.T)
        elif self.kernel == "poly":
            K = self.poly(X, X.T)
        elif self.kernel == "rbf":
            K = self.rbf(X, X)
        else:
            raise ValueError("Kernel not found")
        
        # Setting up the parameters for the quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(y, (1, n), 'd')
        b = cvxopt.matrix(0.0)

        # Constraints for Lagrange multipliers (alphas)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        # Solve the quadratic programming problem using cvxopt
        cvxopt.solvers.options['show_progress'] = False  # Disable solver output
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Extract the support vectors and alphas
        self.alphas, self.sv_idx, self.sv_X, self.sv_y, ind = self.extract_SVs(X, y, alphas)
        
        # Calculate the weight vector (only relevant for linear kernel)
        self.W = np.zeros(m)
        for i in range(len(self.alphas)):
            self.W += self.alphas[i] * self.sv_y[i] * self.sv_X[i]

        # Calculate the intercept (bias term)
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[i], self.sv_idx])
        self.b /= len(self.alphas)

    def predict(self, X):
        if self.kernel == "linear":
            K = self.linear(X, self.sv_X.T)
            decision = np.dot(K, self.alphas * self.sv_y) + self.b
            return np.sign(decision)
        elif self.kernel == "poly":
            K = self.poly(X, self.sv_X.T)
            decision = np.dot(K, self.alphas * self.sv_y) + self.b
            return np.sign(decision)
        elif self.kernel == "rbf":
            K = self.rbf(X, self.sv_X)
            decision = np.dot(K, self.alphas * self.sv_y) + self.b
            return np.sign(decision)
        else:
            return None


# Function to plot decision boundary
def plot_decision_boundary(svm, X, y):
    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict on the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
    plt.title(f"SVM Decision Boundary ({svm.kernel} kernel)")
    plt.show()

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create a synthetic 2D dataset for classification
    X, y = make_classification(
        n_samples=200, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0,  # No redundant features
        n_clusters_per_class=1, 
        random_state=42
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the SVM model with RBF kernel
    svm = Dual_SVM(kernel="rbf", C=1.0, gamma=1.0)
    svm.fit(X_train, y_train)

    # Plot the decision boundary
    plot_decision_boundary(svm, X_train, y_train)

    # Predict accuracy on test set
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
