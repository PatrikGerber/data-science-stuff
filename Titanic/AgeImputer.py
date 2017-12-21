import numpy as np

# =============================================================================
# Note: this code is only applciable to the Titanic dataset
# Imputes missing values of Age, using median imputation, depending on the 
# Sex and the Fare paid by the passanger
# =============================================================================

class AgeImputer:
    def __init__(self):
        self.med1 = -1
        self.med2 = -1
        self.med3 = -1
        self.med4 = -1
        self.fareMedian = -1
        
    def fit(self, X, y = None):
        self.fareMedian = np.median(X["Fare"])
        non_null = 1 - X["Age"].isnull()
        
        subset1 = (X["Sex"] == "male") & ((X["Fare"] > self.fareMedian) == True)
        subset1 = subset1 & non_null
        self.med1 = np.median(X[subset1]["Age"])
        
        subset2 = (X["Sex"] == "male") & ((X["Fare"] > self.fareMedian) == False)
        subset2 = subset2 & non_null
        self.med2 = np.median(X[subset2]["Age"])
        
        subset3 = (X["Sex"] == "female") & ((X["Fare"] > self.fareMedian) == True)
        subset3 = subset3 & non_null
        self.med3 = np.median(X[subset3]["Age"])
        
        subset4 = (X["Sex"] == "female") & ((X["Fare"] > self.fareMedian) == False)
        subset4 = subset4 & non_null
        self.med4 = np.median(X[subset4]["Age"])
        
        return self
    
    def transform(self, X):
        answer = X.copy()
        nulls = X["Age"].isnull()
        
        subset1 = (X["Sex"] == "male") & ((X["Fare"] > self.fareMedian) == True)
        subset1 = subset1 & nulls
        answer.loc[subset1, "Age"] = self.med1
        
        subset2 = (X["Sex"] == "male") & ((X["Fare"] > self.fareMedian) == False)
        subset2 = subset2 & nulls
        answer.loc[subset2, "Age"] = self.med2
        
        subset3 = (X["Sex"] == "female") & ((X["Fare"] > self.fareMedian) == True)
        subset3 = subset3 & nulls
        answer.loc[subset3, "Age"] = self.med3
        
        subset4 = (X["Sex"] == "female") & ((X["Fare"] > self.fareMedian) == False)
        subset4 = subset4 & nulls
        answer.loc[subset4, "Age"] = self.med4
        
        return answer
                
    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X)
        self.transform(X)