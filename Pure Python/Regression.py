# -*- coding: utf-8 -*-
"""
Created on Mon Jan  18 21:04:42 2016

@author: Karsin Kamakotti
"""


import numpy as np


class Regression(object):

    def __init__(self, aplha=.15, Lambda=.0001, epoch=100, X, Y, labels):
        """ Initialize all the parameters"""
        X = np.c_[np.ones(m), X]
        X = np.matrix(X)
        self.X, self.Y = X, Y
        self.m, self.n, self.k = X.shape[0], X.shape[1], y.shape[1]
        self.alpha, self.Lambda, self.epoch = alpha, Lambda, epoch
        self.theta = np.zeros((n, self.k))
        self.labels = labels

    def cost(self):
        """ Cost Function """
        j = -(1 / self.m) * (Y.T.dot(np.log10(sigmoid(X.dot(Theta.T))).T) +
                    (
                     (1 - Y).T.dot(np.log10(sigmoid(1 - (X.dot(Theta.T)))).T))
                      ) + (Lambda / 2 * self.m) * (Theta.dot(Theta.T))

        return j

    def grad_descent(self):
        """ Gradient Descent with Regularization"""
        theta = np.matrix(self.theta).T

        for i in range(1, _iter + 1):
            g = self.theta

            g[0, :] = g[0, :] - (self.alpha / m) * (
                sigmoid(self.X.dot(g)) - self.Y[:, 0]).T.dot(self.X[:, 0])

            g[1:, ] = g[1:, ] * (1 - (self.alpha * self.Lambda / self.m)) - (
                self.alpha / self.m) * (
                self.X[:, 1:]).T.dot(sigmoid(self.X.dot(g)) - self.Y)

            self.theta = g

        return self.theta

    def fit(self):
        # Training(Calculating Theta using Gradient Descent)
        for k in range(0, self.k):
            theta[:, k] = descent(X, Y[:, k], self.theta[:, k], self.Lamb, self.alpha, self.epoch).T

    def predict(self, X_test):
        """ Predicting the class of the Test Data"""
        Y_test = sigmoid(X_test.dot(self.theta))
        Y_test = (np.argmax(Y_test, axis=1))
        Y_class = list(map(lambda x: labels[x], Y_test))


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))
