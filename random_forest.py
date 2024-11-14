# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
balance_data = pd.read_csv(url, header=None, sep=',')

# Assigning feature columns and target column
X = balance_data.values[:, 1:5]  # Features (Left-Weight, Left-Distance, Right-Weight, Right-Distance)
Y = balance_data.values[:, 0]    # Target (Class Name)

# Splitting the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Building and training the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
rf_classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Calculating accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
