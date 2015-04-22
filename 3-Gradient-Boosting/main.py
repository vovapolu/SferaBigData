import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from boosting import GradientBoosting
from random_forest import RandomForest
import matplotlib.pyplot as plt


def load_data():
    with file("data.txt", "r") as dataFile:
        X = np.array([map(float, line.split()[:8]) for line in dataFile])
        return X[:, 1:], X[:, 0]

def CV(clf, X, y):
    return cross_validation.cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10), scoring="mean_squared_error")

def testBoosting():
    scores = []
    treesCount = 10
    for i in xrange(1, treesCount):
        boost = GradientBoosting(i)
        cvScore = CV(boost, X, y)
        scores.append(-np.mean(cvScore))

    plt.plot(np.arange(1, treesCount), scores)
    plt.show()

def testRandomForest():
    scores = []
    treesCount = 10
    for i in xrange(1, treesCount):
        forest = RandomForest(i)
        cvScore = CV(forest, X, y)
        scores.append(-np.mean(cvScore))

    plt.plot(np.arange(1, treesCount), scores)
    plt.show()

X, y = load_data()

testBoosting()
testRandomForest()