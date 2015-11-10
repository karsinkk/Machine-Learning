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
alpha = 1
iter = 100


#  Logistic Regression Functions
sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

log10 = np.vectorize(math.log10)

def cost(_x, _y, _theta, _lamb):
     j = -(1 / m) * (_y.T.dot(log10(sigmoid(_x.dot(_theta.T))).T) + ((1 - _y).T.dot(log10(sigmoid(1 - (_x.dot(_theta.T)))).T))) + (_lamb / 2 * m) * (_theta.dot(_theta.T))
     return j

def descent(_x, _y, _theta, _lamb, _alpha, _iter):

     _theta = np.matrix(_theta).T

     for i in range(1,_iter + 1):
          g = _theta
          g[0,:] = g[0,:] - (_alpha / m) * (sigmoid(_x.dot(g)) - _y[:,0]).T.dot(_x[:,0])
          g[1:,] = g[1:,] * (1 - (_alpha * _lamb / m)) - (_alpha / m) * (_x[:,1:]).T.dot(sigmoid(_x.dot(g)) - _y)
          _theta = g
     return _theta


# Training(Calculating Theta using Gradient Descent)
for k in range(0,y.shape[1]):

     theta[:,k] = descent(X,y[:,k],theta[:,k],lamb,alpha,iter).T




