from sklearn import datasets
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from tabulate import tabulate
from heterogeneousPooling import HPClassifier


def Get_Stats(data, target, classifier, grade={'classifier__n_estimators': [10, 25, 50, 100]}):
    scaler = StandardScaler()

    pipeline = Pipeline([("scaler", scaler), ('classifier', classifier)])

    grade = grade

    gridSearch = GridSearchCV(pipeline, param_grid=grade,
                              scoring='accuracy', cv=4)

    cv = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=36851234)

    scores = cross_val_score(
        gridSearch, data, target, scoring='accuracy', cv=cv)

    mean = scores.mean()
    std = scores.std()
    lower, upper = stats.norm.interval(
        0.95, loc=mean, scale=std / np.sqrt(len(scores)))

    return [scores, mean, std, lower, upper]


def Get_Digits():
    digits = datasets.load_digits()
    target = digits.target
    data = digits.data
    return data, target


def Get_Wine():
    wine = datasets.load_wine()
    target = wine.target
    data = wine.data
    return data, target


def Get_Breast_Cancer():
    cancer = datasets.load_breast_cancer()
    target = cancer.target
    data = cancer.data
    return data, target

def Calculate_Digits_Classifiers():
    data, target = Get_Digits()
    classifier = HPClassifier()
    hillResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["HillClimbing"]})

    data, target = Get_Digits()
    classifier = HPClassifier()
    simulatedResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["SimulatedAnnealing"]})

    data, target = Get_Digits()
    classifier = HPClassifier()
    geneticResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["Genetic"]})

    print("\n")

    print("------------------------ DIGITS --------------------------")

    print("Hill Climbing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], hillResults[1:]], headers="firstrow"))
    
    print("Simulated Annealing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], simulatedResults[1:]], headers="firstrow"))
           
    print("Genetic: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], geneticResults[1:]], headers="firstrow"))

    return [hillResults, simulatedResults, geneticResults]


def Calculate_Wine_Classifiers():
    data, target = Get_Wine()
    classifier = HPClassifier()
    hillResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["HillClimbing"]})

    data, target = Get_Wine()
    classifier = HPClassifier()
    simulatedResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["SimulatedAnnealing"]})

    data, target = Get_Wine()
    classifier = HPClassifier()
    geneticResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["Genetic"]})

    print("\n")

    print("------------------------ WINE --------------------------")

    print("Hill Climbing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], hillResults[1:]], headers="firstrow"))
    
    print("Simulated Annealing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], simulatedResults[1:]], headers="firstrow"))
           
    print("Genetic: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], geneticResults[1:]], headers="firstrow"))

    return [hillResults, simulatedResults, geneticResults]


def Calculate_Breast_Cancer_Classifiers():
    data, target = Get_Breast_Cancer()
    classifier = HPClassifier()
    hillResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["HillClimbing"]})

    data, target = Get_Breast_Cancer()
    classifier = HPClassifier()
    simulatedResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["SimulatedAnnealing"]})

    data, target = Get_Breast_Cancer()
    classifier = HPClassifier()
    geneticResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [3, 5, 7], 'classifier__nm_Metaheuristic': ["Genetic"]})

    print("\n")

    print("------------------------ BREAST CANCER --------------------------")

    print("Hill Climbing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], hillResults[1:]], headers="firstrow"))
    
    print("Simulated Annealing: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], simulatedResults[1:]], headers="firstrow"))
           
    print("Genetic: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"], geneticResults[1:]], headers="firstrow"))

    return [hillResults, simulatedResults, geneticResults]


def Get_Accuracies(results):
    accuracies = []

    for item in results:
        accuracies.append(item[0])

    return accuracies
