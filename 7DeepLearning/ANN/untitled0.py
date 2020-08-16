# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:24:48 2020

@author: encry973r
"""

import numpy as np
import pandas as pd

# import dataset
data = pd.read_csv('Churn_Modelling.csv')

# matrix of features and dependent variable
X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

# convert country column to dummies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ct for the country dimension
ct = ColumnTransformer(transformers=[('one_hot_encoder', 
                                     OneHotEncoder(categories='auto'),
                                     [1])],
                       remainder='passthrough')
X = ct.fit_transform(X)
# drop first dummie column
X = X[:, 1:]

# ct for the gender dimension
ct = ColumnTransformer(transformers=[('one_hot_encoder',
                                      OneHotEncoder(categories='auto'),
                                      [3])],
                        remainder='passthrough')
X = ct.fit_transform(X)
# drop first dummie column
X = X[:, 1:]

# split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# copies for reference
X_train2 = X_train.copy()
X_test2 = X_test.copy()

# dimensional reduction : PCA , n_components = 6 ; gave 62.81%
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance_ratio = pca.explained_variance_ratio_

# build ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# add input and first hidden layer
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu', input_dim=6))
# add second hidden layer
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
# add out put layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit ANN to train dataset
classifier.fit(X_train, y_train, batch_size=10, epochs=100)



























# Build ANN architecture
# import keras
# from keras.models import Sequential
# from keras.layers import Dense

# initialize the classifier
# classifier = Sequential()

# accuracy
# units = 6, accuracy = 83.41%
# units = 11, accuracy = 86.68%

# add input and first hidden
# classifier.add(Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=11))
# # add second hidden layer
# classifier.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))
# # output layer
# classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# # compile network
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # fit data to classifier
# classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# # predict for X_test
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)

# decision tree classifier
# 80.25% accuracy
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# Random Forest Classifier
# 86.85% accuracy
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# kernel svm
# 86.85% accuracy
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)

# poly svm
# 85.7% accuracy
# from sklearn.svm import SVC
# classifier = SVC(kernel='poly', degree=3, random_state=0)
# classifier.fit(X_train, y_train)

# kmeans
# 82.95% accuracy
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# KNN
# 82.7% accuracy
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# classifier.fit(X_train, y_train)


# # predict for X_test
# y_pred = classifier.predict(X_test)


# show confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 

# truth = ((cm[0, 0] + cm[1, 1])/cm.sum())*100

# print(str(truth) + '%')



































