import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeClassifier

class SelectedFeaturesDecisionTree(BaseEstimator, RegressorMixin):
    """
        Decision Tree, which uses certain features.
    """

    def __init__(self, selectedFeatures):
        self.selectedFeatures = selectedFeatures
        self.tree = DecisionTreeClassifier()

    def fit(self, X, y):
        self.tree.fit(X[:, self.selectedFeatures], y)
        return self

    def predict(self, X):
        return self.tree.predict(X[:, self.selectedFeatures])


class RandomForest(BaseEstimator, RegressorMixin):
    """
        Random Forest, which uses CART as base algorithm.
    """

    def __init__(self, numberOfTrees = 50, bestFeatureNumber = None, bestSamplesNumber = None):
        self.bestFeatureNumber = bestFeatureNumber
        self.bestSamplesNumber = bestSamplesNumber
        self.numberOfTrees = numberOfTrees
        self.trees = np.empty(numberOfTrees, dtype=object)

    def fit(self, X, y):
        bestFeatureNumber = self.bestFeatureNumber
        bestSamplesNumber = self.bestSamplesNumber
        if bestFeatureNumber is None:
            bestFeatureNumber = X.shape[1]
        if bestSamplesNumber is None:
            bestSamplesNumber = X.shape[0] / 10

        for i in xrange(self.numberOfTrees):
            selectedFeatures = np.random.choice(X.shape[1], bestFeatureNumber, replace=False)
            selectedSamples = np.random.choice(X.shape[0], bestSamplesNumber)
            self.trees[i] = SelectedFeaturesDecisionTree(selectedFeatures)
            self.trees[i].fit(X[selectedSamples], y[selectedSamples])

        return self

    def predict(self, X):
        res = np.zeros(X.shape[0])
        for i in xrange(self.numberOfTrees):
            res += self.trees[i].predict(X)

        return res / self.numberOfTrees
