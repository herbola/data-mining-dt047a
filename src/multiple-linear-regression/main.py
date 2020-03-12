# enables import from other directories

import os, sys 
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error

# statsmodel
import statsmodels.api as sm
# shared imports
from shared.config import *
from shared.utility import *
 

def main():
    df = getFile(WHITE_WINE_FILENAME)
    
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    X, y = getXandY(df)

    # Splitting the df into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    # Fitting Multiple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    print("max error: " + str(max_error(y_test, y_pred) * 100) + "%")
    print("mean absolute error: " + str(mean_absolute_error(y_test, y_pred) * 100 ) + "%")

    # Building the optimal model using Backward Elimination
    X = np.append(arr = np.ones((len(df), 1)).astype(int), values = X, axis = 1)
    X_opt = X[:, list(range(0, len(df.columns) - 2))]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()

    #Delete citric acid
    X_opt = X[:, list(range(0, len(df.columns) - 2))]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()

    #Delete chlorides
    X_opt = X[:, list(range(0, len(df.columns) - 2))]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()

    #Delete total sulfur
    X_opt = X[:, list(range(0, len(df.columns) - 2))]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()


main()