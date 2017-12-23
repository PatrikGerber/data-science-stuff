import sys
sys.path.append("/home/patrik/Programming/data-science-stuff")

# General useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for preprocessing
from sklearn import preprocessing
from Titanic.TitanicDataCleaner import TitanicDataCleaner
from Titanic.TitanicFeatureEngineer import TitanicFeatureEngineer
from Titanic.TitanicDataManager import TitanicDataManager
from ModelEvaluation import ModelEvaluation

# Imports for Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Loading data
# =============================================================================

data_train, y, data_predict = TitanicDataManager.loadData()

# =============================================================================
# Combining the two datasets for ease of manipulation
# =============================================================================

combined = pd.concat([data_train, data_predict])
combined.reset_index(drop = True, inplace = True)

# =============================================================================
# Cleaning data
# =============================================================================

combined = TitanicDataCleaner.clean(combined)

# =============================================================================
# Engineering new features
# =============================================================================

titanicFeatureEngineer = TitanicFeatureEngineer()
combined = titanicFeatureEngineer.engineer(combined)

# =============================================================================
# Scaling data
# =============================================================================

scaler = preprocessing.StandardScaler()
combined = pd.DataFrame(scaler.fit_transform(combined), columns = combined.columns)


#remove = ["SibSp", "Title_Master. ", "Parch", "Embarked_S", "IsAlone", "CabinLetter_E", "CabinLetter_B", 
#          "CabinLetter_D", "CabinLetter_C", "Embarked_Q", "Title_Rev. ", "CabinLetter_F", "Title_Dr. ", 
#          "Title_None", "CabinLetter_G", "CabinLetter_T", "CabinLetter_n"]
#combined.drop(labels = remove, axis = 1, inplace = True)

# =============================================================================
# Unwrapping the two datasets
# =============================================================================

data_train = combined.loc[0:data_train.shape[0] - 1, :]

data_predict = combined.loc[data_train.shape[0]:, :]
data_predict.reset_index(drop = True, inplace = True)

# =============================================================================
# Selecting a model
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(data_train, y, 
                test_size = 0.2, random_state = 1, stratify = y)

param_range	= [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3, 10]
forest_param_grid = {"n_estimators": [64, 128, 256],
                     "max_features": [3, 5, 7],
                     "max_depth": [3, 4, 5]}

estimator = GridSearchCV(estimator = RandomForestClassifier(),
                   param_grid = forest_param_grid,
                   scoring = "accuracy", 
                   cv = 5, 
                   n_jobs = -1)

estimator.fit(X_train, y_train)
#ModelEvaluation.displayCVScores(estimator, X_train, y_train)

# =============================================================================
# Producing predictions
# =============================================================================

e = RandomForestClassifier(**estimator.best_params_)
ModelEvaluation.plotLearningCurve(e, data_train, y, cv = 3, n_jobs = -1)
e.fit(data_train, y)

feat_labels = data_train.columns
importances = e.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(data_train.shape[1]):
    print("%2d)	%-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
# =============================================================================
# Saving predictions
# =============================================================================

TitanicDataManager.savePredictions(e, data_predict)