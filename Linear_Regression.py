import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a linear relationship with 2 features
np.random.seed(42)
X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)
y = 4 + 3 * X1 + 5 * X2 + np.random.randn(100, 1)  # Linear relationship with some noise

# Stack X1 and X2 horizontally to form a 2D dataset
X = np.hstack([X1, X2])

# Function to calculate the predictions
def predict(X, theta):
    return X.dot(theta)

# Function to compute the Mean Squared Error (MSE) cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        gradients = (1 / m) * X.T.dot(predict(X, theta) - y)
        theta = theta - learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Print cost every 10 iterations
        if i % 10 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return theta, cost_history

# Add a bias term (intercept) to X
X_b = np.c_[np.ones((100, 1)), X]  # X_b contains the intercept and 2 features

# Initialize theta (weights) to random values
theta = np.random.randn(3, 1)

# Set learning rate and number of iterations
learning_rate = 0.1
num_iterations = 100

# Perform Gradient Descent
theta_final, cost_history = gradient_descent(X_b, y, theta, learning_rate, num_iterations)

# Print the final theta values
print(f"Final theta values: \n{theta_final}")

# Plot the cost function history
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), cost_history, 'b-', linewidth=2, label='Cost Function')
plt.title('Cost Function History')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.show()  # Make sure this is called to display the plot
