from pathlib import Path
from re import I
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor as RFR 

#* set path
dataRoot = Path.cwd().parent/'data'

#*#########
#! Load and Prep Data
#*#########
#load data
df = pd.read_csv(dataRoot/'data.csv')
#turn into numpy matrix
X = df.to_numpy()

#*#########
#! Imputation Pipeline
#*#########
#set up learner
rfr = RFR()
#set up sklearn iterative imputer
it_imp = IterativeImputer(estimator = rfr, 
                        max_iter = 10,
                        #n_nearest_features = X, could be helpful to reduce computation time and we could use number of features used by a Lasso on a subsample
                        #however, not necessarily helpful because Lasso performed for each column and here only one parameter
                        random_state = 2022)
#fit iterative imputer to X and return X with imputed values - ignore first column as this is only ID
imputed_X = it_imp.fit_transform(X[:, 1:])