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

# Imports for Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/train.csv")
data_predict = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/test.csv")

y = data_train.loc[:, "Survived"]
data_train.drop("Survived", axis = 1, inplace = True)

# =============================================================================
# Combining the two datasets for ease of manipulation
# =============================================================================

combined = pd.concat([data_train, data_predict])
combined.reset_index(drop = True, inplace = True)

# =============================================================================
# Cleaning data
# =============================================================================

titanicDataCleaner = TitanicDataCleaner()
combined = titanicDataCleaner.clean(combined)

# =============================================================================
# Engineering new features
# =============================================================================

titanicFeatureEngineer = TitanicFeatureEngineer()
combined = titanicFeatureEngineer.engineer(combined)

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

param_range	= [0.01, 0.03, 0.1, 0.3, 1.0]

svc_param_grid = {'C': param_range,
                  "gamma": param_range,
                  'kernel': 'rbf' }

forest_param_grid = {"n_estimators": [1000],
                     "max_features": [5],
                     "max_depth": [4]}

estimator = GridSearchCV(estimator = RandomForestClassifier(),
                   param_grid = forest_param_grid,
                   scoring = "accuracy", 
                   cv = 2, 
                   n_jobs = -1)

scores = cross_val_score(estimator, X_train, y_train, scoring = "accuracy", cv = 4)
print("CV accuracy:	%.3f	+/-	%.3f" % (np.mean(scores), np.std(scores)))

estimator.fit(X_train, y_train)
print("Accuracy on test set: %.3f" % estimator.score(X_test, y_test))

# =============================================================================
# Producing predictions
# =============================================================================

e = RandomForestClassifier(**estimator.best_params_)
e.fit(data_train, y)

Ids = pd.DataFrame(list(range(892, 892 + data_predict.shape[0])), columns = ["PassengerId"])
survived = pd.DataFrame(e.predict(data_predict).astype(int), columns = ["Survived"])
answer = pd.concat([Ids, survived], axis = 1)

answer.to_csv("answer.txt", encoding='utf-8', index=False)
