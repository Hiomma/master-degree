import collections as cl

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


class HPClassifier(BaseEstimator):
    def __init__(self, n_Samples=None):
        super().__init__()
        self.n_Samples = n_Samples
        self.classifiers = []
        self.frequencies = []

    def fit(self, data, target):
        classifiers = []

        #recebe as frequencias
        self.frequencies = cl.Counter(target)

        #gera os classificadores
        for nr_Index in range(self.n_Samples):
            if nr_Index != 0:
                data, target = resample(data, target, random_state=nr_Index-1)

            gaussianClassifier = GaussianNB()
            decisionClassifier = DecisionTreeClassifier()
            knnClassifier = KNeighborsClassifier(n_neighbors=1)

            gaussianClassifier.fit(data, target)
            decisionClassifier.fit(data, target)
            knnClassifier.fit(data, target)

            classifiers.append(gaussianClassifier)
            classifiers.append(decisionClassifier)
            classifiers.append(knnClassifier)

        self.classifiers = classifiers

    def predict(self, data):
        result = []
        predictions = []

        #faz o predict de cada classificador e joga num array
        for classifier in self.classifiers:
            prediction = classifier.predict(data)
            predictions.append(prediction)

        dataFrame = pd.DataFrame(data=predictions)
        data = np.asarray(data)

        #verifica o mais votado
        for index in dataFrame.columns:
            column = dataFrame[index]
            
            choice = self.getBetterChoice(column)
            result.append(choice)

        return np.asarray(result)

    def getBetterChoice(self, column):
        frequencies = cl.Counter(column).items()
        betterFrequency = 0
        better = None

        for key, value in frequencies:
            #caso o valor é maior que a maior frequencia
            if(value > betterFrequency):
                better = key
                betterFrequency = value
            #caso seja igual, ele verifica quem é o mais frequente da base pra retornar
            elif(value == betterFrequency):
                for selfKey, selfValue in self.frequencies.most_common():
                    if(selfKey == better):
                        break
                    if(selfKey == key):
                        better = key
                        betterFrequency = value

        return better
