# enables import from other directories

import os, sys 
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))



# scipy
from scipy.stats import sem

# sklearn

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix # accuracy
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
# shared imports
from shared.config import *
from shared.utility import *
from shared.plot import plot

# config
test_size = 0.25
random_state = 42
def main(): 
    df = getDataSet()
    decision_tree(df)


def decision_tree(df):
    X, y = getXandY(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 0)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # fit method is used to train data
    classifier = DecisionTreeClassifier(criterion= 'entropy', random_state = random_state)
    classifier.fit(X_train, y_train)


    # predict
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Testsize: ' + str(test_size * 100) + '%')
    print('Random state: ' + str(random_state) + '%')
    print('Success rate: ' + str(successRate(cm)))
    print('Kappa success rate: ' + str(kappaSuccesRate(cm)))
    

def getDataSet(): 
    df1 = getFile(WHITE_WINE_FILENAME)
    df1 = normalize(df1)
    df2 = getFile(RED_WINE_FILENAME)
    df2 = normalize(df2)
    df = concat(df1, 0, df2, 1)
    return df

main()