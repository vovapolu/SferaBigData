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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.decomposition import PCA, RandomizedPCA, FastICA, KernelPCA

def load_data():
    with file("data.txt", "r") as dataFile:
        X = np.array([map(float, line.split()[:8]) for line in dataFile])
        return (X[:, 1:], X[:, 0])
        
def CV(clf, X, y):
    return cross_validation.cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10), scoring="mean_squared_error")
    
X, y = load_data();

featureSelectors = [("Random Forest", RandomForestClassifier()), ("PCA", PCA()), 
                    ("Decision Tree", DecisionTreeClassifier()), ("KBest", SelectKBest(k=5)),
                    ("Linear SVC", svm.LinearSVC()), ("Randomized PCA", RandomizedPCA()), 
                    ("Fast ICA", FastICA()), ("Kernel ICA", KernelPCA())]
regressions = [("Linear Regression", LinearRegression()), ("Decision Tree", DecisionTreeRegressor()),
               ("SVR", svm.SVR())]

for selector in featureSelectors:
    for regression in regressions:
        clf = Pipeline([("feature selector", selector[1]), ("regression", regression[1])])
        print selector[0], "+", regression[0]
        print "CV score", -np.mean(CV(clf, X, y))
        
for regression in regressions:
    clf = regression[1]
    print "Only", regression[0]
    print "CV score", -np.mean(CV(clf, X, y))
