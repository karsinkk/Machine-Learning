__author__ = 'Karsin'

import numpy as np
import pandas as pd
import math
import os

# Getting the Data
path = os.getcwd() + "\Logistic Regression\Data\Training.csv"
ColNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
train = pd.read_csv(path, names=ColNames)
Y = train.Class.tolist()
X = np.genfromtxt(path, delimiter=',')

# Cleaning the Data and generating params
X = X[:, :-1]
m = X.shape[0]
n = X.shape[1] + 1
X = np.c_[np.ones(m), X]
X = np.matrix(X)
y = np.hstack((np.matrix(list(map(lambda x: int(x == "Iris-setosa"), Y))).T, np.matrix(list(map(lambda x: int(x == "Iris-versicolor"), Y))).T,np.matrix(list(map(lambda x: int(x == "Iris-virginica"), Y))).T))
theta = np.zeros((n, 3))
lamb = 1


# Regularization Functions

sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

def cost(X, y, theta, lamb):
     j = -(1 / m) * ((y.dot(math.log10(sigmoid(X.dot(theta))))) + ((1 - y).dot(math.log10(sigmoid(1 - (X.dot(theta))))))) + (lamb / 2 * m) * (theta.dot(theta.T))
     return j

print(cost(X,y[:,0],theta[:,0],lamb))
