# enables import from other directories

import os, sys 
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


# other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# scipy
from scipy.stats import sem

# sklearn

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix # accuracy
from sklearn import metrics

# shared imports
from shared.config import *
from shared.utility import *


def main(): 
    df = getDataSet()
    decision_tree(df)

def loo_cv(X, y): # not sure how to use this yes
    loo = LeaveOneOut()
 
    print(loo.get_n_splits(X))
    print(loo)

    for train_index, test_index in loo.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       # print(X_train, X_test, y_train, y_test)

def decision_tree(df):
    columns = len(df.columns)
    X = df.iloc[:, [0, columns - 2]].values #parameters
    y = df.iloc[:, columns - 1].values # answers

    loo_cv(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


    # preprocess
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # fit method is used to train data
    classifier = DecisionTreeClassifier(criterion= 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)


    # predict
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot(X_train, y_train, classifier, X_test, y_test)
    

def getDataSet(): 
    df1 = getFile(WHITE_WINE_FILENAME)
    df1 = normalize(df1)
    df2 = getFile(RED_WINE_FILENAME)
    df2 = normalize(df2)
    df = concat(df1, 0, df2, 1)
    return df


def plot(X_train, y_train, classifier, X_test, y_test):
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.2, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree (Training set)')
    # plt.xlabel(ALCOHOL)
    # plt.ylabel(QUALITY)
    plt.legend()
    plt.show()

    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.2, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree (Test set)')
    # plt.xlabel(ALCOHOL)
    # plt.ylabel(QUALITY)
    plt.legend()
    plt.show()
main()