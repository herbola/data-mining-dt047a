# enables import from other directories

import os, sys 
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# shared imports
from shared.config import *
from shared.utility import *
from shared.plot import *

# config
test_size = 0.25

def main(): 
    df = getDataSet()

    X, y = getXandY(df)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)


    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting K-NN to the Training set

    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Testsize: ' + str(test_size * 100) + '%')
    print('Success rate: ' + str(successRate(cm)))
    print('Kappa success rate: ' + str(kappaSuccesRate(cm)))
    roc(classifier, X_test, y_test)

def freqBinning(df, targetClass):
    entries = len(df)
    sum = df[targetClass].sum()
    divider = sum / entries
    df = divide(df, divider, 1, 0, targetClass)
    return df

def widthBinning(df, targetClass):
    max = df[targetClass].max()
    min = df[targetClass].min();
    divider = (max + min) / 2
    df = divide(df, divider, 1, 0, targetClass)
    return df

def divide(df, divider, c1, c2, targetClass):
    print('Dividing: ' + str(divider))
    df1 = df[df[targetClass] >= divider]
    df2 = df[df[targetClass] < divider]
    df = concat(df1, c1, df2, c2)
    return df

def getDataSet(): 
    df = getFile(WHITE_WINE_FILENAME)
    df.drop_duplicates()

    df = freqBinning(df, QUALITY)
   # df = widthBinning(df, QUALITY)

    df = normalize(df)
    df.drop(len(df.columns) - 2, 1)
    df[len(df.columns) - 1] = df[len(df.columns) - 1].apply(np.int64)
    return df;

main()