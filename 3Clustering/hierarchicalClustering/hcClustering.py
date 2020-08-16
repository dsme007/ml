# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:44:53 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('mall.csv')
X = data.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# the optimal number of 5 clusters was obtained from the dendrogram
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# visualize the cluster
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], c='red', s=200, label='Sensible')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], c='blue', s=200, label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], c='green', s=200, label='Target')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], c='cyan', s=200, label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], c='magenta', s=200, label='Careful')
plt.xlim(0, 200)
plt.ylim(0, 150)
plt.legend(loc='upper right')
plt.title('Customers Clusters [Hierarchical Clustering]')
plt.xlabel('Annual Income (k$')
plt.ylabel('Spending Score (1 - 100')
plt.savefig('hierarchical_clustering.png')
plt.show()