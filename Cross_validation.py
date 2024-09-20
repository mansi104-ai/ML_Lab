#Import the required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Load the breast cancer dataset
data = load_breast_cancer()

#Convert the data to pandas dataframe 
df= pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Original data(first 5 rows):")
print(df.head())

#Simulate the missing values(optional step)
df.iloc[0:10,0] = np.nan #Set the first 10 values in the first column to NaN

#Step2 :Handle the missing values
df.fillna(df.mean(), inplace=True) 
print("\nData after filling Missing Values (first 5 rows) :")
print(df.head())

##Step 3 : Splitting features and target
X = df.drop('target',axis = 1)#Features
y = df['target']

##Step4 : Standardization ( Z-SScore Normalization)
scaler= StandardScaler()
x_scaled= scaler.fit_transform(X)

##--------------Step 5 : Cross-Validation----------##
#Initialize the model
model = LogisticRegression(max_iter = 10000)

#Apply 5-fold cross-validation
cv_scores = cross_val_score(model,x_scaled,y, cv = 5)

#print the cross validation scores and mean accuracy
print("\n Cross-Validation scores : ",cv_scores)
print("Mean accuracy :",np.mean(cv_scores))


