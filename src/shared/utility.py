

import pandas as pd
from sklearn import preprocessing
import os
dirname = os.path.dirname(__file__)


import numpy as np


# get file from dataset folder
def getFile(filename):
    filename = os.path.join(dirname, "./../datasets/" + filename)
    df = pd.read_csv(filename, delimiter=',')
    return df

# normalize values in datafrme, 0-1
def normalize(df): 
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

# concat two dataframes and add type key to it
def concat(df1, name1, df2, name2): 
    df1 = df1.assign(Class = name1);
    df2 = df2.assign(Class = name2);
    # return pd.concat([df1, df2], keys=[name1, name2])
    return pd.concat([df1, df2])

def getXandY(df):
    return  [df.iloc[:, list(range(0, len(df.columns) - 2))].values, df.iloc[:, len(df.columns) - 1].values]

def successRate(cm):
    return (cm[0][0] + cm[1][1]) / cmSum(cm)

def cmSum(cm):
    return sum(map(sum, cm))

def kappaSuccesRate(cm):
    a = cm[0][0]
    b = cm[0][1]
    c = cm[1][0]
    d = cm[1][1]
    p0 = (a + d) / cmSum(cm)
    py = ((a + b) / cmSum(cm)) * ((a + c) / cmSum(cm))
    pn = ((c + d) / cmSum(cm)) * ((b + d) / cmSum(cm))
    pe = py + pn
    k = (p0 - pe) / ( 1 - pe)
    return k



