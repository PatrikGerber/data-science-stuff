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
    def clean(self, data):
        data = self.imputeFare(data)
        data = self.imputeAge(data)
        data = self.imputeEmbarked(data)
        data = self.encode(data, ["Embarked", "Sex"])
        return data
    
    def imputeFare(self, data):
        imp = _Imputer(strategy = "median", features = ["Fare"])
        return imp.fit_transform(data)
    
    def imputeAge(self, data):
        ageImputer = AgeImputer()
        return ageImputer.fit_transform(data)
    
    def imputeEmbarked(self, data):
        most_common = data["Embarked"].value_counts().index[0]
        data.loc[data.isnull()["Embarked"], "Embarked"] = most_common
        return data
    
    def encode(self, data, features):
        return pd.get_dummies(data = data, columns = features, drop_first = True)