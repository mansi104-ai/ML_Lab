import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Load the breast cancer dataset
data = load_breast_cancer()
X= data.data
y= data.target

#Convert the data to a pandas dataframe for a better understanding 
df = pd.DataFrame(X, columns= data.feature_names)
df['target'] = y

#Display the basic information about the data
print("Data shape : ", df.shape)
print("Print the first 5 rows")
print(df.head())

#Step 1 : Preprocessing  - Standardize the features
scaler = StandardScaler()
X_scales = scaler.fit_transform(X)

#Step 2 : Train and test data splitting in the ratio of 70:30
X_train, X_test, y_train, y_test = train_test_split(X_scales,y,test_size = 0.3, random_state = 42)

#Step 3 : Initialize the SVM classifier
svm_model = SVC(kernel = 'rbf', random_state = 42)

#Step 4 : Train the SVM model on the training data
svm_model.fit(X_train,y_train)

#Step 5 : Make predictions on the testing data
y_pred = svm_model.predict(X_test)

#Evaluate the model performance
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of the SVM model : {accuracy:.4f}")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


