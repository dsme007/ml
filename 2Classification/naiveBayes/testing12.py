# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:30:53 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')

# matrices of feature
X = data.iloc[:, 2:4].values
y = data.iloc[:, 4].values

# train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fit to naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predicting the Test set result
y_pred = classifier.predict(X_test)

# confusion metric
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
truth = cm[0, 0] + cm[1, 1]

correct_values = cm[0,0] + cm[1,1]
wrong_values = cm[0,1] + cm[0,1]
accuracy = np.round((correct_values/(correct_values + wrong_values))*100, 2)

print("Correct values : {0}\nWrong values : {1}\nAccuracy : {2}%"
      .format(correct_values, wrong_values, accuracy))
