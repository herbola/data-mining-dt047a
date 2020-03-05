

import pandas as pd
from sklearn import preprocessing
import os
dirname = os.path.dirname(__file__)

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

