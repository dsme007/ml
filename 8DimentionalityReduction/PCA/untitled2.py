# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 05:33:02 2020

@author: encry973r
"""

# import libraries
import numpy as np
import pandas as pd

# import dataset
data = pd.read_csv('Wine.csv')

# matrix of features and independent variable
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance_ratio = pca.explained_variance_ratio_





















