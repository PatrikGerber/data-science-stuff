import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model

data = pd.read_csv("C:\\Users\\Internet\\Downloads\\test.csv")
lRes = sk.linear_model.LogisticRegression()
