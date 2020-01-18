#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:36:06 2020

@author: siddhartha
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
#X = X.reshape(30,1)  # making X into matrix not vector
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')
plt.title('Sal vs Exp(train set)')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()


plt.scatter(X_test,y_test,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')
plt.title('Sal vs Exp (test set)')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()