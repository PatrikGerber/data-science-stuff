import pandas as pd
import numpy as nd

from sklearn.preprocessing import Imputer

# =============================================================================
# Imputes missing values in the specified columns using the specified strategy.
# Only works with numeric values
# =============================================================================

class _Imputer:
    def __init__(self, features, strategy = "mean"):
        self.imputer = Imputer(strategy = strategy)
        self.features = features
        self.nFeatures = len(features)
        
    def fit(self, X, y = None):
        nRows = X.shape[0]
        self.imputer.fit(X.loc[:, self.features].values.reshape(nRows, self.nFeatures))
        return self
    
    def transform(self, X):
        nRows = X.shape[0]
        X_withoutFeatures = X.drop(self.features, axis = 1)
        transformedFeaturesFrame = pd.DataFrame(
                                   self.imputer.transform(
                                        X.loc[:, self.features].values.reshape(nRows, self.nFeatures)), 
                                   columns = self.features)
        
        return pd.concat([X_withoutFeatures, transformedFeaturesFrame], axis = 1)
        
    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X)
        return self.transform(X)