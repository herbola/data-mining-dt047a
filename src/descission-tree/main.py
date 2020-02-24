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
    descision_tree(df)

def loo_cv(X, y):
    loo = LeaveOneOut()
 
    print(loo.get_n_splits(X))
    print(loo)

    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train, X_test, y_train, y_test)

def descision_tree(df):
    # Here the X variable contains all the columns from the dataset,
    # except the "Class" column, which is the label. 
    # The y variable contains the values from the "Class" column.
    # The X variable is our attribute set and y variable contains corresponding labels.
    X = df.drop('Class', axis=1)
    y = df['Class']

   # loo_cv(X, y) # not working


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    # fit method is used to train data
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)


    # predict
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def getDataSet(): 
    df1 = getFile(WHITE_WINE_FILENAME)
    df1 = normalize(df1)
    df2 = getFile(RED_WINE_FILENAME)
    df2 = normalize(df2)
    df = concat(df1, 0, df2, 1)
    return df

main()