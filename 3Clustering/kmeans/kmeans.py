# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:28:01 2020

@author: encry973r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3, 4]].values

# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init=10, max_iter=300, random_state =0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# from the WCSS plot above, the optimal number of clusters(K) = 5

# apply kmeans to mall.csv
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
# y_kmeans returns an array of cluster numbers of each X observation
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# visualize the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1],  c='red', label='Standard')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1],  c='blue', label='Careless')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1],  c='green', label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c='cyan', label='Sensible')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], c='magenta', label='Careful')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, 
            c='black', label='Centroids')
plt.xlim(0, 200)
plt.ylim(0, 150)
plt.legend()
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.savefig('kmeans.png')
plt.show()








