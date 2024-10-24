import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
import plotly.express as px
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self,X,num_clusters):
        self.K = num_clusters #cluster number
        self.max_iterations = 100 #max iteration

        self.num_examples,self.num_features = X.shape

        self.plot_figure= True

    def initialize_random_centroids(self,X):
        centroids = np.zeros((self.K,self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids
    
    #Create cluster function

    def create_cluster(self,X,centroids):
        clusters= [[] for _ in range(self.K)]
        for point_idx,point in enumerate(X):
            