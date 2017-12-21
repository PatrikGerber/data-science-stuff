import sys
sys.path.append("/home/patrik/Programming/data-science-stuff/Titanic")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from _Imputer import _Imputer
from AgeImputer import AgeImputer

from sklearn.linear_model import LogisticRegression

data_train = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/train.csv")
data_test = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/test.csv")

y_train = data_train.loc[:, "Survived"]
data_train = data_train.loc[:, ["Sex", "Age", "Fare"]]
data_test = data_test.loc[:, ["Sex", "Age", "Fare"]]

# =============================================================================
# Imputing Fare 
# =============================================================================

imp = _Imputer(strategy = "median", features = ["Fare"])
imp.fit(data_train)
data_train = imp.transform(data_train)

# =============================================================================
# Imputing Age
# =============================================================================

ageImputer = AgeImputer()
ageImputer.fit(data_train)
data_train = ageImputer.transform(data_train)

# =============================================================================
# Encoding Sex
# =============================================================================

sexEnc = preprocessing.LabelEncoder()
sexEnc.fit(data_train["Sex"])
data_train = pd.concat([pd.DataFrame(sexEnc.transform(data_train["Sex"]), columns = ["Sex"]), 
                        data_train.drop("Sex", axis = 1)], 
                       axis = 1)

# =============================================================================
# Encoding Embarked
# =============================================================================

embEnc = preprocessing.OneHotEncoder(sparse = False)

#pipe = make_pipeline(_Imputer(strategy = "median", features = ["Fare"]),
#                     AgeImputer(),
#                     preprocessing.LabelEncoder(),
#                     LogisticRegression())
#
#pipe.fit(data_train, y_train)


# =============================================================================
# trainData = pd.read_csv("train.csv")
# testData = pd.read_csv("test.csv")
# 
# # Imputing missing values for Age and Fare
# imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# trainData.loc[:,["Age", "Fare"]] = imp.fit_transform(trainData.loc[:,["Age", "Fare"]].values.reshape(-1, 2))
# testData.loc[:,["Age", "Fare"]] = imp.fit_transform(testData.loc[:,["Age", "Fare"]].values.reshape(-1, 2))
# 
# # Encoding categorical variables
# ### Sex
# trainData.loc[:, "Sex"] = (trainData.loc[:, "Sex"] == "male").astype(int)
# testData.loc[:, "Sex"] = (testData.loc[:, "Sex"] == "male").astype(int)
# 
# enc = preprocessing.OneHotEncoder(sparse = False)
# trainSexes = pd.DataFrame(enc.fit_transform(trainData.loc[:, "Sex"].values.reshape(-1,1)), columns = ["Female", "Male"])
# testSexes = pd.DataFrame(enc.fit_transform(testData.loc[:, "Sex"].values.reshape(-1,1)), columns = ["Female", "Male"])
# 
# trainData = pd.concat([trainData, trainSexes], axis = 1)
# testData = pd.concat([testData, testSexes], axis = 1)
# trainData.drop("Sex", axis=1, inplace=True)
# testData.drop("Sex", axis=1, inplace=True)
# 
# ### Embarked
# labelEnc = preprocessing.LabelEncoder()
# 
# labelEnc.fit(pd.concat([trainData.loc[:, "Embarked"], testData.loc[:, "Embarked"]], axis = 0).astype(str))
# 
# trainData.loc[:, "Embarked"] = labelEnc.transform(trainData.loc[:, "Embarked"].astype(str))
# testData.loc[:, "Embarked"] = labelEnc.transform(testData.loc[:, "Embarked"].astype(str))
# 
# enc.fit(pd.concat([trainData.loc[:, "Embarked"], testData.loc[:, "Embarked"]], axis = 0).values.reshape(-1,1))
# 
# trainData = pd.concat([trainData, pd.DataFrame(enc.transform(trainData.loc[:, "Embarked"].values.reshape(-1,1)), columns = ["A", "B", "C", "D"])], axis = 1)
# testData = pd.concat([testData, pd.DataFrame(enc.transform(testData.loc[:, "Embarked"].values.reshape(-1,1)), columns = ["A", "B", "C", "D"])], axis = 1)
# 
# trainData.drop("Embarked", axis=1, inplace=True)
# testData.drop("Embarked", axis=1, inplace=True)
# 
# # Fitting the model
# X = trainData.loc[:, ["Age", "Fare", "Male", "Female", "A", "B", "C", "D"]]
# XTest = testData.loc[:, ["Age", "Fare", "Male", "Female", "A", "B", "C", "D"]]
# y = trainData.loc[:, "Survived"]
# 
# lReg = sk.linear_model.LogisticRegression()
# lReg.fit(X, y)
# answer = pd.DataFrame(np.zeros(shape = (testData.shape[0], 2)).astype(int),
#                       columns = ["PassengerId", "Survived"])
# answer.loc[:, "Survived"] = lReg.predict(XTest).astype(int)
# answer.loc[:, "PassengerId"] = range(892, 892 + testData.shape[0])
# answer.to_csv("answer.txt", encoding='utf-8', index=False)
# 
# =============================================================================
