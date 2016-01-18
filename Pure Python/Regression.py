# -*- coding: utf-8 -*-
"""
Created on Mon Jan  18 21:04:42 2016

@author: Karsin Kamakotti
"""


import numpy as np


class Regression(object):

    def __init__(self, aplha=.15, Lambda=.0001, epoch=100, X, Y, labels):
        """ Initializing all the parameters"""
        X = np.c_[np.ones(m), X]
        X = np.matrix(X)
        self.X, self.Y = X, Y
        self.m, self.n, self.k = X.shape[0], X.shape[1], y.shape[1]
        self.alpha, self.Lambda, self.epoch = alpha, Lambda, epoch
        self.theta = np.zeros((n, self.k))
        self.labels = labels
        self.cost_epoch = []

    def cost(self):
        """ Cost Function """
        j = -(1 / self.m) * (self.Y.T.dot(np.log10(sigmoid(self.X.dot(self.theta.T))).T)+
                     (
                     (1 - self.Y).T.dot(np.log10(sigmoid(1 - (self.X.dot(self.theta.T)))).T))
                     ) + (self.Lambda / 2 * self.m) * (self.theta.dot(self.theta.T))

        return j

    def grad_descent(self, _Y, _theta):
        """ Gradient Descent with Regularization"""
        _theta = np.matrix(_theta).T

        for i in range(1, self.epoch + 1):
            g = _theta

            g[0, :] = g[0, :] - (self.alpha / self.m) * (
                sigmoid(self.X.dot(g)) - _Y[:, 0]).T.dot(self.X[:, 0])

            g[1:, ] = g[1:, ] * (1 - (self.alpha * self.Lambda / self.m)) - (
                self.alpha / self.m) * (
                self.X[:, 1:]).T.dot(sigmoid(self.X.dot(g)) - _Y)

            _theta = g
          
        return _theta

    def fit(self):
        """Training (Calculating Theta using Gradient Descent) """
        for k in range(0, self.k):
            theta[:, k] = self.grad_descent(
                          X,
                          Y[:, k],
                          self.theta[:, k],
                          ).T

    def predict(self, X_test, Y_act):
        """ Predicting the class of the Test Data"""
        Y_test = sigmoid(X_test.dot(self.theta))
        Y_test = (np.argmax(Y_test, axis=1))
        Y_class = list(map(lambda x: labels[x], Y_test))
        Accuracy = np.sum(
            np.matrix(list(map(lambda x, y: int(x == y), Y_test, Y_act)))
            ) / Y_act.size * 100

        return(Y_test, Y_class, Accuracy)


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))
