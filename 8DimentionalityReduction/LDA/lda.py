# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:41:28 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv('Wine.csv')

# matrix of features
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=None)
# both X_train and y_train are fitted: SUPERVISED LEARNING
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# predicter
y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)














