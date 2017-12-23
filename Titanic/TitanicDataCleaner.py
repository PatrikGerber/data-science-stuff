import sys
sys.path.append("/home/patrik/Programming/data-science-stuff")

import pandas as pd

from sklearn import preprocessing
from _Imputer import _Imputer
from Titanic.AgeImputer import AgeImputer

# =============================================================================
# The clean method imputes missing values of Age, Fare and Embarked, 
# Furthermore it encodes the features Embarked and Sex using 
# One-Hot-Encoding
# =============================================================================

class TitanicDataCleaner:
    @staticmethod
    def clean(data):
        data = TitanicDataCleaner.imputeFare(data)
        data = TitanicDataCleaner.imputeAge(data)
        data = TitanicDataCleaner.imputeEmbarked(data)
        data = TitanicDataCleaner.encode(data, ["Embarked", "Sex"])
        return data
    
    @staticmethod
    def imputeFare(data):
        imp = _Imputer(strategy = "median", features = ["Fare"])
        return imp.fit_transform(data)
    
    @staticmethod
    def imputeAge(data):
        ageImputer = AgeImputer()
        return ageImputer.fit_transform(data)
    
    @staticmethod
    def imputeEmbarked(data):
        most_common = data["Embarked"].value_counts().index[0]
        data.loc[data.isnull()["Embarked"], "Embarked"] = most_common
        return data
    
    @staticmethod
    def encode(data, features):
        return pd.get_dummies(data = data, columns = features, drop_first = True)