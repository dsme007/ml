# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:22:26 2020

@author: encry973r
"""

import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# matrix of features and dependent variables
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# encode gender column (because it contains )
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

"""encode gender column
# ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [2])], remainder='passthrough')
fit to X
X = ct.fit_transform(X)
drop one encoded column to avoid the dummie variable trap
X = X[:, 1:]
"""

# encode country column
ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')
# fit to X
X = ct.fit_transform(X)
# drop one encoded column to avoid the dummie variable trap
X = X[:, 1:]
# convert X to floating point array
X = np.array(X, dtype=np.float)

# split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize model
classifier = Sequential()
# adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=11))
# adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# adding the output layer
# use activation = 'softmax' when dealing with an output of more than 2 categories
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# compile the ANN
# when dealing with an output of more than 2 categories, use loss = 'categorical_crossentropy'
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# predict y for our test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

correct_values = cm[0,0] + cm[1,1]
wrong_values = cm[0,1] + cm[1,0]
accuracy = np.round((correct_values/(correct_values + wrong_values))*100, 2)

print("Correct values : {0}\nWrong values : {1}\nAccuracy : {2}%"
      .format(correct_values, wrong_values, accuracy))






















