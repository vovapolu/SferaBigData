# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA, RandomizedPCA, FastICA
import matplotlib.pyplot as plt

sys.stdout = file("results.txt", "w")

def load_data():
    with file("data.txt", "r") as dataFile:
        X = np.array([map(float, line.split()[:8]) for line in dataFile])
        return (X[:, 1:], X[:, 0])

def CV(clf, X, y):
    return cross_validation.cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10), scoring="mean_squared_error")


class TreeFeatureSelector():
    def __init__(self, tree, numberOfFeatures):
        self.tree = tree
        self.numberOfFeatures = numberOfFeatures

    def fit(self, X, y=None):
        self.tree.fit(X, y)
        return self

    def transform(self, X):
        importances = self.tree.feature_importances_
        inds = importances.argsort()[::-1][:self.numberOfFeatures]
        return X[:, inds]


featureSelectorsCount = 7
def getFeatureSelectors(bestFeaturesNumber):
    featureSelectors = [("Random Forest", TreeFeatureSelector(RandomForestClassifier(), bestFeaturesNumber)),
                        ("PCA", PCA(n_components=bestFeaturesNumber, whiten=True)),
                        ("Decision Tree", TreeFeatureSelector(DecisionTreeClassifier(), bestFeaturesNumber)),
                        ("KBest", SelectKBest(k=bestFeaturesNumber)),
                        ("Randomized PCA", RandomizedPCA(n_components=bestFeaturesNumber, whiten=True)),
                        ("Fast ICA", FastICA(n_components=bestFeaturesNumber, whiten=True)),
                        ("Extra Trees", TreeFeatureSelector(ExtraTreesClassifier(), bestFeaturesNumber))]

    return featureSelectors

regressions = [("Linear Regression", LinearRegression()),
               ("Decision Tree", DecisionTreeRegressor()),
               ("SVR", SVR()),
               ("Random Forest", RandomForestRegressor()),
               ("Extra Trees", ExtraTreesRegressor())]

nowPlot = 0

def updatePlots():
    plotStep = 7
    if (len(results) >= plotStep):
        global nowPlot

        nowPlot += 1
        fig = plt.figure(nowPlot)
        ax = fig.add_subplot(111)

        for i in xrange(0, plotStep):
            nowres = results.pop()
            ax.plot(nowres[0], nowres[1], label=nowres[2])

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
        fig.savefig('plots/' + labels[0], bbox_extra_artists=(lgd,), bbox_inches='tight')

results = []
X, y = load_data()

for regression in regressions:
    points = np.empty((featureSelectorsCount, X.shape[1]))
    nowPoint = 0
    for number in xrange(1, X.shape[1] + 1):
        selectors = getFeatureSelectors(number)
        selectorNum = 0
        for selector in selectors:
            clf = Pipeline([("feature selector", selector[1]), ("regression", regression[1])])
            points[selectorNum][number - 1] = -np.mean(CV(clf, X, y))
            selectorNum += 1

    for selectorNum in xrange(featureSelectorsCount):
        results.append((np.arange(1, X.shape[1] + 1), points[selectorNum], getFeatureSelectors(1)[selectorNum][0] + " + " + regression[0]))
        updatePlots()