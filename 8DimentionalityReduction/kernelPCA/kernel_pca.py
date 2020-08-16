# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:49:42 2020

@author: encry973r
"""

import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply kernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel='rbf', random_state=0)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
containded_variance_ratio = kpca.

# prediction
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



















