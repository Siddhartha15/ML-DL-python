#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:32:14 2020

@author: siddhartha
"""

# Artificial Neural Networks

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])

#Categorical features deprecated in newer versions of OneHotEncoder class
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
        transformers=[(
            "Country",
            OneHotEncoder(),
            [1]
            )],
        remainder='passthrough'
    )
X = np.array(ct.fit_transform(X),dtype=int)
X = X[:,1:] #Removing dummy variables after oneHotEncoder

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Make the ANN 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation='relu',input_dim=11))
#adding the Second hidden layer
classifier.add(Dense(units=6,activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1,activation='sigmoid')) #sigmoid is used for output layer with 1 category,
                                                    # Softmax is used for more than one categories

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fiting the ANN to training set 
classifier.fit(X_train, y_train, batch_size=10,epochs=100)




























