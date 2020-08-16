# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:25:31 2020

@author: encry973r
"""
# import necessary files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read-in dataset
data = pd.read_csv('Salaries.csv')

# independent and dependent variables
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# feature scalling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# Y_train = sc.fit_transform(Y_train)
# Y_test = sc.fit_transform(Y_test)

# regression training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predict Y_test
Y_pred = regressor.predict(X_test)


# visualizing result for train set
# plt.scatter(X_train, Y_train)
# plt.plot(X_train, regressor.predict(X_train), c='red')
# plt.title('Salary vs. Experience (Train set result)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# # visualizing result for test set
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, regressor.predict(X_test), c='black')
# plt.title('Salary vs. Experience (Test set result')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()



















