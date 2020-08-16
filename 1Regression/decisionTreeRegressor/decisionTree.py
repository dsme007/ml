# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:29:07 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Gaming_data.csv')
X = data.iloc[:, 0:1].values
Y = data.iloc[:, 1:2].values

"""
no need for feature scaling because DT does does not utilize euclidean distances
no need!!!!
"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)

Y_pred = regressor.predict([[7.5]])

# to make the curve smooth!!!
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)


plt.scatter(X, Y)
plt.plot(X_grid, regressor.predict(X_grid), color='red')
