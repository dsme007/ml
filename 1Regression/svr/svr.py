# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:29:48 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Gaming_data.csv')

# MATRIX OF FEATURES
X = data.iloc[:, 0:1].values
Y = data.iloc[:, 1:2].values

# view the scatter
# plt.plot(X,Y, c='black')

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# fit svr to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y.ravel())

plt.scatter(X, Y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Gaming data (SVR)')
plt.xlabel('Gaming steps')
plt.ylabel('Points')
plt.show()

# predict Y for different values of 'value'
value = 7.5


# transform X
Y_pred = regressor.predict(sc_X.transform(np.array([[value]])))
# revert the predicted value to non-transformed value
Y_pred = sc_Y.inverse_transform(Y_pred)




