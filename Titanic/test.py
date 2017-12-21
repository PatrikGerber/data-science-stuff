import sys
sys.path.append("/home/patrik/Programming/data-science-stuff")

# General useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from _Imputer import _Imputer
from AgeImputer import AgeImputer

# Imports for Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data_train = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/train.csv")
data_predict = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/test.csv")

y = data_train.loc[:, "Survived"]
data_train.drop("Survived", axis = 1, inplace = True)

# =============================================================================
# Imputing missing values
# =============================================================================

# Fare

imp = _Imputer(strategy = "median", features = ["Fare"])
imp.fit(data_train)
data_train = imp.transform(data_train)
data_predict = imp.transform(data_predict)

# Age

ageImputer = AgeImputer()
ageImputer.fit(data_train)
data_train = ageImputer.transform(data_train)
data_predict = ageImputer.transform(data_predict)

# Embarked

most_common = data_train["Embarked"].value_counts().index[0]
data_train.loc[data_train.isnull()["Embarked"], "Embarked"] = most_common
data_predict.loc[data_predict.isnull()["Embarked"], "Embarked"] = most_common

# =============================================================================
# Encoding categorical variables
# =============================================================================

# Sex and Embarked

temp = pd.concat([data_train, data_predict])
temp = pd.get_dummies(data = temp, columns = ["Sex", "Embarked"], drop_first = True)
temp.reset_index(drop = True, inplace = True)

# =============================================================================
# Engineering new features
# =============================================================================

# Creating #Passengers feature: number of passengers with given ticket

nPassengers = pd.DataFrame(temp["Ticket"].value_counts()[temp["Ticket"].values].values, 
                           columns = ["#Passengers"])
temp = pd.concat([temp, nPassengers], axis = 1)

# Dividing Fare by #Passengers

temp["Fare"] /= temp["#Passengers"]

# Creating IsAlone feature

temp["IsAlone"] = (temp["Parch"] + temp["SibSp"] == 0) + 0

# Dropping unnecessary features, and unwrapping into test and training sets
temp.drop(["Cabin", "Name", "Ticket"], axis = 1, inplace = True)

data_train = temp.loc[0:data_train.shape[0] - 1, :]

data_predict = temp.loc[data_train.shape[0]:, :]
data_predict.reset_index(drop = True, inplace = True)

# =============================================================================
# Selecting a model
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(data_train, y, 
                test_size = 0.2, random_state = 0, stratify = y)

param_range	=	[0.001, 0.003, 0.01, 0.03, 	0.1, 0.3, 1.0]

param_grid	=	{'C':	param_range,
				 'gamma':	param_range,
                 'kernel':	['rbf']}

estimator = GridSearchCV(estimator = SVC(),
                   param_grid = param_grid,
                   scoring = "accuracy", 
                   cv = 5, 
                   n_jobs = -1
                   )

scores = cross_val_score(estimator, X_train, y_train, scoring = "accuracy", cv = 5)
print("CV accuracy:	%.3f	+/-	%.3f" % (np.mean(scores), np.std(scores)))

#svc.fit(X_train, y_train)
#svc.score(X_test, y_test)

answer = pd.DataFrame([estimator.predict(data_predict).astype(int),
                       range(892, 892 + data_predict.shape[0])],
                      columns = ["PassengerId", "Survived"])

answer.to_csv("answer.txt", encoding='utf-8', index=False)
