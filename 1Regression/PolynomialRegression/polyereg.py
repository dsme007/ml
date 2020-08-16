# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 04:40:21 2020

@author: encry973r
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Gaming_data.csv')

# MATRIX OF FEATURES
X= dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values


# always plot the scatter plot to hav an idea of the problem's nature
# plot.scatter(X,Y)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# using polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
# fit the powers of the variables of required degree
X_poly = poly_reg.fit_transform(X)
# fit to linear regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# visualizing the linear regresion result
plt.scatter(X, Y)
plt.plot(X, lin_reg.predict(X), color='red')
plt.title('Linear regession plot')
plt.xlabel('Steps')
plt.ylabel('Points')
plt.show()



# Visualizing the Polynomial Regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y)
plt.plot(X, lin_reg2.predict(X_poly), color='black')
plt.title('Polynomial regession plot')
plt.xlabel('Steps')
plt.ylabel('Points')
plt.show()

# THE COMPANY NOW HAS A MODEL TO PREDICT THE POINTS FOR PRESENT 
# AND ADDITIONAL LEVEL (SHOULD THEY DESIRE)

# predict Y for some value of X using linear regression
print(lin_reg.predict([[7.5]]))

# predict Y for some value of X using polynomial regression (LEVEL 11 FOR INSTANCE)
print(lin_reg2.predict(poly_reg.fit_transform([[7.5]])))













