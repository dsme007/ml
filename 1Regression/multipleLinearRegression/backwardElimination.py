# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:10:56 2020

@author: encry973r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Org_data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 4].values

# encode categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
                                     remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# avoid dummie variable trap
X = X[:, 1:]

# split dataset into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# multiple Linear regression
# append unity column to X
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# 3rd variable has highest 'P' value
# R2 = .951, Adj.R2 = 0.945, 
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
# print(regressor_OLS.summary())

# # second variable has higher 'p' value
# R2 = .951, Adj.R2 = 0.946, 
# X_opt = X[:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
# print(regressor_OLS.summary())

# 3rd variable has highest 'P' value
# R2 = .951, Adj.R2 = 0.948, 
# X_opt = X[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
# print(regressor_OLS.summary())

# 3rd variable has highest 'P' value
# this makes a better fit as it has the highest Adj. R2 value = 0.95 R2 = 0.950
# X_opt = X[:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
# print(regressor_OLS.summary())

"""the optimal Columns that can predict the profit with the highest statistical 
significance/effec(when compared with the rest columns)"""
# are the first and 4th columns
# R2 = .947, Adj.R2 = 0.945, 
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())























