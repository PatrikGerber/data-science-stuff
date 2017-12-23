import pandas as pd
import re

class TitanicFeatureEngineer:
    def engineer(self, data):
        data = self.createFamilySize(data)
        data = self.createNPassengers(data)
        data = self.createActualFare(data)
        data = self.createIsAlone(data)
        data = self.createNameLength(data)
        data = self.createTitle(data)
        data = self.createCabinLetter(data)
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
    
    def createNameLength(self, data):
        for index in range(data.shape[0]):
            data.loc[index, "NameLength"]  = len(data.loc[index, "Name"])
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
    
    def createCabinLetter(self, data):
        cabins = data["Cabin"]
        cabinFrame = pd.DataFrame(list(map(lambda s: str(s)[0], cabins)), columns = ["CabinLetter"])
        return pd.concat([data, pd.get_dummies(data = cabinFrame, columns = ["CabinLetter"], drop_first = True)], axis = 1)

    def dropUnusedFeatures(self, data):
        data.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis = 1, inplace = True)
        return data
