import pandas as pd

from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# Given a feature, it one-hot-encodes that specific feature
# =============================================================================

class _OneHotEncoder:
    def __init__(self, feature, sparse = False, dropColumn):
        self.encoder = OneHotEncoder(sparse = sparse)
        self.feature = feature
        
    def fit(self, X, y = None):
        nRows = X.shape[0]
        self.encoder.fit(X.loc[:, self.feature].values.reshape(nRows, 1))
        return self
    
    def transform(self, X):
        nRows = X.shape[0]
        X_withoutFeature = X.drop(self.feature, axis = 1)
        transformedFeatureFrame = pd.DataFrame(
                                   self.encoder.transform(
                                        X.loc[:, self.feature].values.reshape(nRows, 1)))
        
        return pd.concat([X_withoutFeature, transformedFeatureFrame], axis = 1)
        
    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X)
        return self.transform(X)