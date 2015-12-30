__author__ = 'Karsin'

import numpy as np
import pandas as pd
import math
import os

# Getting the Data
CWD = os.getcwd()
path = CWD + "\Logistic Regression\Data\Training.csv"
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
lamb = .0001
alpha = .15
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


# Testing

path = CWD + "\Logistic Regression\Data\Iris.csv"
ColNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
test = pd.read_csv(path, names=ColNames)
Y = test.Class.tolist()
X = np.genfromtxt(path, delimiter=',')

X = X[:, :-1]
m = X.shape[0]
n = X.shape[1] + 1
X = np.c_[np.ones(m), X]
X = np.matrix(X)
Solution = np.hstack((np.matrix(list(map(lambda x: int(x == "Iris-setosa"), Y))).T, np.matrix(list(map(lambda x: int(x == "Iris-versicolor"), Y))).T,np.matrix(list(map(lambda x: int(x == "Iris-virginica"), Y))).T))
Solution = np.argmax(Solution,axis=1)

Y = sigmoid(X.dot(theta))
Result = (np.argmax(Y,axis=1))

# Result Analysis
Comparison = np.concatenate((Result,Solution),axis=1)

print("The Outcome is :\n",Comparison,"\n",Result.size)

Accuracy = np.sum(np.matrix(list(map(lambda x,x1: int(x == x1), Solution, Result)))) / Result.size * 100

print("\n Accuracy is : ",Accuracy,"%")



