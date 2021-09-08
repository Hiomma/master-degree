from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import pandas as pd
import collections as cl
import numpy as np
from hillClimbing import Hill_Climbing
from genetic import Genetic
from simulatedAnnealing import Simulated_Annealing


class HPClassifier(BaseEstimator):
    def __init__(self, n_Samples=None, nm_Metaheuristic="HillClimbing"):
        super().__init__()
        self.n_Samples = n_Samples
        self.classifiers = []
        self.frequencies = []
        self.nm_Metaheuristic = nm_Metaheuristic

    def fit(self, data, target):
        classifiers = []
        
        #recebe as frequencias
        self.frequencies = cl.Counter(target)

        #gera os classificadores
        for nr_Index in range(self.n_Samples):
            dataAux, targetAux = resample(data, target, random_state=nr_Index)

            gaussianClassifier = GaussianNB()
            decisionClassifier = DecisionTreeClassifier()
            knnClassifier = KNeighborsClassifier(n_neighbors=1)

            gaussianClassifier.fit(dataAux, targetAux)
            decisionClassifier.fit(dataAux, targetAux)
            knnClassifier.fit(dataAux, targetAux)

            classifiers.append(gaussianClassifier)
            classifiers.append(decisionClassifier)
            classifiers.append(knnClassifier)
        
        optimal_state =  []

        if(self.nm_Metaheuristic == "HillClimbing"):
          optimal_state  = Hill_Climbing(classifiers, data, target, 120)
        elif(self.nm_Metaheuristic == "SimulatedAnnealing"):
          optimal_state = Simulated_Annealing(200, 0.1, classifiers, data, target,10, 120)
        else:
          optimal_state = Genetic(classifiers, data, target, 20, 100, 0.9, 0.1, 120, 20)
          
        print(optimal_state, "optimal")

        classifiersList = []
        for nr_Index, classifier in enumerate(classifiers):
            if optimal_state[nr_Index] == 1:
                classifiersList.append(classifier)
        
        self.classifiers = classifiersList

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