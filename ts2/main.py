# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA, RandomizedPCA, FastICA

def load_data():
    with file("data.txt", "r") as dataFile:
        X = np.array([map(float, line.split()[:8]) for line in dataFile])
        return (X[:, 1:], X[:, 0])
        
def CV(clf, X, y):
    return cross_validation.cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10), scoring="mean_squared_error")
    
X, y = load_data();

featureSelectors = [("Random Forest", RandomForestClassifier()), ("PCA", PCA(n_components="mle", whiten=True)), 
                    ("Decision Tree", DecisionTreeClassifier()), ("KBest", SelectKBest(k=5)),
                    ("Linear SVC", svm.LinearSVC()), ("Randomized PCA", RandomizedPCA(whiten=True)), 
                    ("Fast ICA", FastICA(whiten=True)), ("Extra Trees", ExtraTreesClassifier())]
                    
regressions = [("Linear Regression", LinearRegression()), ("Decision Tree", DecisionTreeRegressor()),
               ("SVR", svm.SVR()), ("Random Forest", RandomForestRegressor()), ("Extra Trees", ExtraTreesRegressor())]

results = []

for selector in featureSelectors:
    for regression in regressions:
        clf = Pipeline([("feature selector", selector[1]), ("regression", regression[1])])
        results.append((-np.mean(CV(clf, X, y)), selector[0] + " + " + regression[0]))
        
for regression in regressions:
    clf = regression[1]
    results.append((-np.mean(CV(clf, X, y)), "Only " + regression[0]))

results.sort()

for result in results:
    print result[1]
    print "CV score:", result[0]
