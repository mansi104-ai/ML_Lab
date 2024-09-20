# Import the relevant libraries and the iris dataset
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# module for Z-score
from sklearn.preprocessing import StandardScaler

#module for min max scaling
from sklearn.preprocessing import MinMaxScaler

#Load the iris dataset
data = load_iris()

#Convert the dataset into Pandas Dataframe 
df = pd.DataFrame(data.data, columns= data.feature_names)


# #Simulate missing values by replacing some values with Nan
# df.iloc[0:10,0] = np.nan  #Introduce Nan in the first column for the first 10 rows

# print("Original data with missing values : ")
# print(df.head())

#Step 1: Handling missing values
#We'll fill the missing values with the mean of the column

df.fillna(df.mean(), inplace= True)
print("\nData after filling the missing values:")
print(df.head())

##-------------Z-score-----------------##
scaler = StandardScaler()
z_score_normalized = scaler.fit_transform(df)
df_z_score = pd.DataFrame(z_score_normalized, columns=df.columns)
print("\nZ-score normalized Data (first 5 rows):")
print(df_z_score.head())

##------------Min-Max Scaling------------##
min_max_scaler = MinMaxScaler()
min_max_normalized = min_max_scaler.fit_transform(df)
df_min_max = pd.DataFrame(min_max_normalized,columns=df.columns)
print("\nMin-Max Normalized data (first 5 rows )")
print(df_min_max.head())
