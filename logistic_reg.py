#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#Cost function (Binary Cross entropy)
def compute_cost(X,y,weights):
    m = len(y)
    h = sigmoid(np.dot(X,weights))
    cost = (-1/m)*np.sum(y * np.log(h) + (1-y) *np.log(1-h))
    return cost

#Gradient descent
def gradient_descent(X,y,weights,learning_rate,num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(np.dot(X,weights))
        gradient = (1/m)*np.dot(X.T,(h-y))
        weights -= learning_rate*gradient
        cost = compute_cost(X,y,weights)
        cost_history.append(cost)

        if i%1000 == 0:
            print(f"iteration{i}: Cost{cost}")

    return weights, cost_history

#Logiatic regression model class
class LogisticRegressionScratch:
    def __init__(self, learning_rate = 0.01,num_iterations = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None

    def fit(self,X,y):
        X = np.insert(X,0,1,axis = 1)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.weights,self.cost_history = gradient_descent(
            X,y,self.weights,self.learning_rate,self.num_iterations
        )

    def predict_proba(self,X):
        X = np.insert(X,0,1,axis = 1) #bias term
        return sigmoid(np.dot(X,self.weights))
    
    def predict(self,X,threshold = 0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
#Function to predict the decision boundary(only for 2D data)
def plot_decision_boundary(model,X,y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

#Example
if __name__ == "__main__":

    #Load the wine dataset from sklearn
    wine= load_wine()
    X= wine.data[:,:2] #Use only two features for visualization
    y = (wine.target != 0)*1 #Convert to binary classification(1 is not class 0 , otherwise 0)

    #Split the training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    #Feature scaling for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Initialize and train logistic regression model
    model = LogisticRegressionScratch(learning_rate=0.01, num_iterations=100)
    model.fit(X_train,y_train)

    #Predict on the test set
    y_pred = model.predict(X_test)

    #Compute accuracy
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Test accuracy : {accuracy *100:.2f}%")

    #Plot the decision boundary
    plot_decision_boundary(model,X_train,y_train)
