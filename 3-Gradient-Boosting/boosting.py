import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree.tree import DecisionTreeClassifier
from cart_tree import CartTree

class ConstrantTree():
    """
        Simple predictor, which always returns certain constant.
        It is required for commonality.
    """
    def __init__(self, constant):
        self.constant = constant

    def fit(self, X, y):
        return self

    def predict(self, X):
        res = np.empty(X.shape[0])
        res.fill(self.constant)
        return res

class GradientBoosting(BaseEstimator, RegressorMixin):
    """
        Gradient Boostring with mean squared error.
        It uses CART as basic algorithm.
    """

    def __init__(self, iterationsNumber):
        self.iterationsNumber = iterationsNumber

    def particalPredict(self, X, predictorNumber):
        res = np.zeros((X.shape[0], ))
        for i in xrange(predictorNumber):
            res += self.treeWeights[i] * self.trees[i].predict(X)
        return res

    def fit(self, X, y):
        self.treeWeights = np.empty((self.iterationsNumber, ))
        self.trees = np.empty((self.iterationsNumber, ), dtype=object)

        self.treeWeights[0] = 1
        self.trees[0] = ConstrantTree(np.mean(y))

        for i in xrange(1, self.iterationsNumber):
            newY = 2 * (y - self.particalPredict(X, i))
            self.trees[i] = CartTree()
            self.trees[i].fit(X, newY)
            newYPredict = self.trees[i].predict(X)
            self.treeWeights[i] = sum((newY / 2) * newYPredict) / sum(newYPredict ** 2)

        return self

    def predict(self, X):
        return self.particalPredict(X, self.iterationsNumber)



