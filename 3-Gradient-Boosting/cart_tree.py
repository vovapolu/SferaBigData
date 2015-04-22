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

    def findBestSplit(self, feature, y):
        sortedInds = np.argsort(feature)
        feature = feature[sortedInds]
        y = y[sortedInds]

        leftSum = y[0]
        leftCnt = 1
        rightCnt = y.shape[0] - 1
        while leftCnt < feature.shape[0] - 1 and feature[leftCnt - 1] == feature[leftCnt]:
            leftCnt += 1
            rightCnt -= 1
            leftSum += y[leftCnt - 1]

        rightSum = np.sum(y) - leftSum
        leftMse = np.sum((y[:leftCnt] - leftSum / leftCnt) ** 2)
        rightMse = np.sum((y[leftCnt:] - rightSum / rightCnt) ** 2) if rightCnt > 0 else 0

        bestSplit = leftCnt - 1
        bestMse = leftMse / (leftCnt + 1) + ((rightMse / rightCnt) if rightCnt > 0 else 0)

        for leftCnt in xrange(leftCnt + 1, feature.shape[0]):
            rightCnt -= 1

            leftDeltaMse = (leftSum - (leftCnt - 1) * y[leftCnt - 1]) / (leftCnt * (leftCnt - 1))
            leftSum += y[leftCnt - 1]
            leftMse += leftDeltaMse * leftDeltaMse * (leftCnt - 1) + (y[leftCnt - 1] - leftSum / leftCnt) ** 2

            rightDeltaMse = (rightSum - (rightCnt + 1) * y[leftCnt - 1]) / (rightCnt * (rightCnt + 1))
            rightMse += -(y[leftCnt - 1] - (rightSum / (rightCnt + 1))) ** 2 - rightDeltaMse * rightDeltaMse * rightCnt
            rightSum -= y[leftCnt - 1]

            allMse = leftMse / leftCnt + ((rightMse / rightCnt) if rightCnt > 0 else 0)

            if feature[leftCnt - 1] != feature[leftCnt]:
                if allMse < bestMse:
                    bestSplit = leftCnt - 1
                    bestMse = allMse

        if bestSplit < feature.shape[0] - 1:
            bestSplitVal = (feature[bestSplit] + feature[bestSplit + 1]) / 2.0
        else:
            bestSplitVal = feature[bestSplit] + 1
        return bestSplitVal, bestMse

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
