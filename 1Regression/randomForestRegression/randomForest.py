# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:24:00 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Gaming_data.csv')
X = data.iloc[:, 0:1].values
Y = data.iloc[:, 1:2].values

# fit model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, Y.ravel())

# the steps cannot be seen, so use higher resolution values for X as X_grid
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y)
plt.plot(X_grid, regressor.predict(X_grid), color='red')
plt.title('Gaming steps (Random Forest Regression)')
plt.xlabel('Steps')
plt.ylabel('Points')
plt.show()

y_pred = regressor.predict([[6.5]]) # 160333.3333333333 close to the anticipated value of 160,000
