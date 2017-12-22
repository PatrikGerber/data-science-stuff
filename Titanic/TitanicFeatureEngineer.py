import pandas as pd
import re

class TitanicFeatureEngineer:
    def engineer(self, data):
        data = self.createFamilySize(data)
        data = self.createNPassengers(data)
        data = self.createActualFare(data)
        data = self.createIsAlone(data)
        data = self.createTitle(data)
        data = self.dropUnusedFeatures(data)
        return data
            
    def createFamilySize(self, data):
        data["FamilySize"] = data["Parch"] + data["SibSp"]
        return data
    
    def createNPassengers(self, data):
        nPassengers = pd.DataFrame(data["Ticket"].value_counts()[data["Ticket"].values].values, 
                                   columns = ["#Passengers"])
        return pd.concat([data, nPassengers], axis = 1)

    def createActualFare(self, data):
        data["ActualFare"] = data["Fare"] / data["#Passengers"]
        return data
    
    def createIsAlone(self, data):
        data["IsAlone"] = (data["Parch"] + data["SibSp"] == 0) + 0
        return data

    def createTitle(self, data):
        pattern = "[^ ]*\. "
        for index in range(0, len(data)):
            match = re.search(pattern, data.loc[index, "Name"]).group()
            if match:
                data.loc[index, "Title"] = match
            else:
                data.loc[index, "Title"] = "None"
        
        data.loc[(data["Title"].value_counts()[data["Title"]] < 3).values, "Title"] = "None"
        data = pd.get_dummies(data, columns = ["Title"], drop_first = True)
        return data

    def dropUnusedFeatures(self, data):
        data.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis = 1, inplace = True)
        return data
