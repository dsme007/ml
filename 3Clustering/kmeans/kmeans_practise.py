# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 02:57:59 2020

@author: encry973r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3, 4]].values

# implement WCSS to know the optimal K value
from sklearn.cluster import KMeans
wcss = []

# experiment with a range of 1-10 for k-value
for k in range(1, 11):
    # create the k-mean class
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    # fit the dataset
    kmeans.fit(X)
    # append to wcss
    wcss.append(kmeans.inertia_)
    

# plot wcss against k values
plt.plot(range(1, 11), wcss)
plt.title('The elbow Method')
plt.xlabel('K values')
plt.ylabel('WCSS')
plt.show()

# from the plot above, the optimal k-value is 5
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
# fit and predict observations' clusters
y_kmeans = kmeans.fit_predict(X)

# plot the scatter
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=200, c='red', label='Cluster1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=200, c='blue', label='Cluster2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=200, c='green', label='Cluster3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=200, c='cyan', label='Cluster4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=200, c='magenta', label='Cluster5')
plt.legend(loc='upper right')
plt.title('Clients\' clusters [k-means]')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income(k$)')
plt.show()










