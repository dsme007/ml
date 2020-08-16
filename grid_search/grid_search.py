# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:39:07 2020

@author: encry973r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# train and split datatset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply svm
from sklearn.svm import SVC
classifier = SVC(C =1, kernel='rbf', gamma =1.15, random_state=0 )
classifier.fit(X_train, y_train)

# predict X_test
y_pred = classifier.predict(X_test)

# confusion metric
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
truth = cm[0, 0] + cm[1, 1]
print('truth = ' + str(truth) + '%')

# k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv=10)
accuracies.mean() #mean accuracy = 93%
accuracies.std() #STD = +-(6.57%)

# parameter tuning
from sklearn.model_selection import GridSearchCV
parameters = [
                {'C': [1, 10, 100], 'kernel': ['linear']},
                {'C': [1, 1, 100], 'kernel': ['rbf'], 'gamma': [1.1, 1.15, 1.2]},
            ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


















