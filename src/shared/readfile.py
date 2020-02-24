

import pandas as pd
from sklearn import preprocessing
import os
dirname = os.path.dirname(__file__)


def getFile(filename):
    filename = os.path.join(dirname, "./../datasets/" + filename)
    df = pd.read_csv(filename, delimiter=',')
    return df

def normalize(df): 
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df