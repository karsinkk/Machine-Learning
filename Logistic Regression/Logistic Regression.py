__author__ = 'Karsin'

import numpy as np
import pandas as pd
import math
import os

CWD = os.getcwd()

# Getting the Data
path = CWD + "\Logistic Regression\Data\Training.csv"
ColNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
train = pd.read_csv(path, names=ColNames)
Y = train.Class.tolist()
X = np.genfromtxt(path, delimiter=',')
X = X[:, :-1]

theta = np.zeros(X.shape[1] + 1)
y1 = list(map(lambda x: int(x == "Iris-setosa"), Y))
y1 = np.array(y1)
y2 = list(map(lambda x: int(x == "Iris-versicolor"), Y))
y2 = np.array(y2)
y3 = list(map(lambda x: int(x == "Iris-virginica"), Y))
y3 = np.array(y3)


def signum(x):
    return 1 / (1 + math.exp(-x))
