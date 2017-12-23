import pandas as pd

class TitanicDataManager:
    @staticmethod
    def loadData():
        data_train = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/train.csv")
        data_predict = pd.read_csv("/home/patrik/Programming/data-science-stuff/Titanic/test.csv")
        y = data_train.loc[:, "Survived"]
        data_train.drop("Survived", axis = 1, inplace = True)
        return (data_train, y, data_predict)
    
    @staticmethod
    def savePredictions(fittedEstimator, data_predict):
        Ids = pd.DataFrame(list(range(892, 892 + data_predict.shape[0])), columns = ["PassengerId"])
        survived = pd.DataFrame(fittedEstimator.predict(data_predict).astype(int), columns = ["Survived"])
        answer = pd.concat([Ids, survived], axis = 1)
        answer.to_csv("/home/patrik/Programming/data-science-stuff/Titanic/answer.txt", encoding='utf-8', index=False)
        return
                