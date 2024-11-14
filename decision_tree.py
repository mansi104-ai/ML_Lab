#import the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Loading the dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
# Loading data directly from the URL, specifying no header and ',' as separator
balance_data = pd.read_csv(url, header=None, sep=',')

#Assigning the feature and target column
X = balance_data.values[:,1:5]   # Features (Left-Weight, Left-Distance, Right-Weight, Right-Distance)
Y = balance_data.values[:, 0]    # Target (Class Name)

#Splitting the dataset into training and testing sets ( 70% training, 30% testing)
X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

#Building and training the decision tree classifier
clf_gini = DecisionTreeClassifier(criterion= "gini",random_state=100,max_depth = 3,min_samples_leaf=5)
clf_gini.fit(X_train,y_train)

#Making predictions on the test set
y_pred = clf_gini.predict(X_test)

#Calculating accuracy
accuracy = accuracy_score(y_test,y_pred)
print("Acccuracy score using gini index : ",accuracy)

#Building and training the decision tree classifier with entropy classification
clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=100, max_depth=3,min_samples_leaf=5)
clf_entropy.fit(X_train,y_train)

#Making predictions on the test set
y_pred_entropy = clf_entropy.predict(X_test)

#Calculating accuracy
accuracy_entropy = accuracy_score(y_test,y_pred_entropy)
print("Accuracy using entropy : ",accuracy_entropy)

#Visualizing the decision tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini,filled = True,feature_names=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'], class_names=['L', 'B', 'R'])
plt.show()