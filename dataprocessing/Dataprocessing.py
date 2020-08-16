# -*- coding: utf-8 -*-
"""
Created on Sun May 17 04:32:08 2020

@author: encry973r
"""
# import necessary libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read-in dataset
data = pd.read_csv('Data.csv')

# independent and dependent variables
X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

# work with missing fields
from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
X[:, 1:3] = Imputer.fit_transform(X[:, 1:3])

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('one-hot-encoder', OneHotEncoder(categories='auto'), [0])], 
                                       remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

#   split data into train and test parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

sc_y = StandardScaler()
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

# apply regression algorithm to train dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)





















