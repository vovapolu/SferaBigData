from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class CartTreeNode:
    def __init__(self, splitVal, featureNum):
        self.splitVal = splitVal
        self.featureNum = featureNum
        self.left = None
        self.right = None

    def next(self, x):
        if x[self.featureNum] < self.splitVal:
            return self.left
        else:
            return self.right

    def isLeaf(self):
        return False

class CartTreeLeaf:
    def __init__(self, predictVal):
        self.predictVal = predictVal

    def predict(self, x):
        return self.predictVal

    def isLeaf(self):
        return True

class CartTree(BaseEstimator, RegressorMixin):
    """
       Decision tree with CART algorithm
    """

    def __init__(self, maxDepth=2):
        self.maxDepth = maxDepth

    def calcSplit(self, feature, y, splitVal):
        splitLeftMean = np.mean(y[feature < splitVal])
        splitRightMean = np.mean(y[feature >= splitVal])
        if not np.isnan(splitLeftMean) and not np.isnan(splitRightMean):
            splitLeftMse = np.mean((y[feature < splitVal] - splitLeftMean) ** 2)
            splitRightMse = np.mean((y[feature >= splitVal] - splitRightMean) ** 2)
            return splitLeftMse + splitRightMse
        else:
            return np.mean((y - np.mean(y)) ** 2)

    def findBestSplit(self, feature, y):
        sortedInds = np.argsort(feature)
        sortedFeature = feature[sortedInds]
        sortedY = y[sortedInds]

        bestSplitVal = np.mean(sortedFeature)
        bestSplitCalc = self.calcSplit(sortedFeature, sortedY, bestSplitVal)
        for i in xrange(sortedFeature.shape[0] - 1):
            if sortedFeature[i] != sortedFeature[i + 1]:
                nowSplitVal = (sortedFeature[i] + sortedFeature[i + 1]) / 2.0
                nowSplitCalc = self.calcSplit(sortedFeature, sortedY, nowSplitVal)
                if nowSplitCalc < bestSplitCalc:
                    bestSplitCalc = nowSplitCalc
                    bestSplitVal = nowSplitVal

        return bestSplitVal, bestSplitCalc


    def createNode(self, X, y, depth):
        if X.shape[0] == 0:
            node = CartTreeLeaf(0)
        elif depth >= self.maxDepth or X.shape[0] == 1:
            node = CartTreeLeaf(np.mean(y))
        else:
            bestFeatureSplit = self.findBestSplit(X[:, 0], y)
            bestFeature = 0
            for f in xrange(1, X.shape[1]):
                nowFeatureSplit = self.findBestSplit(X[:, f], y)
                if nowFeatureSplit[1] < bestFeatureSplit[1]:
                    bestFeatureSplit = nowFeatureSplit
                    bestFeature = f

            node = CartTreeNode(bestFeatureSplit[0], bestFeature)
            node.left = self.createNode(X[X[:, bestFeature] < bestFeatureSplit[0]],
                                        y[X[:, bestFeature] < bestFeatureSplit[0]], depth + 1)
            node.right = self.createNode(X[X[:, bestFeature] >= bestFeatureSplit[0]],
                                         y[X[:, bestFeature] >= bestFeatureSplit[0]], depth + 1)

        return node

    def fit(self, X, y):
        self.root = self.createNode(X, y, 0)
        return self

    def predict(self, X):
        res = np.empty(X.shape[0])
        for i in xrange(X.shape[0]):
            nowNode = self.root
            while not nowNode.isLeaf():
                nowNode = nowNode.next(X[i])
            res[i] = nowNode.predictVal

        return res
