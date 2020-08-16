# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:25:31 2020

@author: encry973r
"""
# import necessary files
import numpy as np
import pandas as pd

# read-in dataset
data = pd.read_csv('Data.csv')

# independent and dependent variables
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

# work with missing data 
from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = Imputer.fit_transform(X[:, 1:3])

# categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Y = label.fit_transform(Y)

# split data into train and test parts
from sklearn.model_selection import train_test_split

# split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)











