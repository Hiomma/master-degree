import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import datasets
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        gridSearch, data, target, scoring='accuracy', cv=cv, n_jobs=-1)

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
    classifier = BaggingClassifier()
    baggingResults = Get_Stats(data, target, classifier)

    data, target = Get_Digits()
    classifier = AdaBoostClassifier()
    adaBoostResults = Get_Stats(data, target, classifier)

    data, target = Get_Digits()
    classifier = RandomForestClassifier()
    randomForestResults = Get_Stats(data, target, classifier)

    data, target = Get_Digits()
    classifier = HPClassifier()
    hpResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [1, 3, 5, 7]})

    print("\n")

    print("------------------------ DIGITS --------------------------")

    print("Bagging: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          baggingResults[1:]], headers="firstrow"))

    print("AdaBoost: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          adaBoostResults[1:]], headers="firstrow"))

    print("RandomForest: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          randomForestResults[1:]], headers="firstrow"))

    print("Heterogeneous Pooling: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          hpResults[1:]], headers="firstrow"))

    return [baggingResults, adaBoostResults, randomForestResults, hpResults]


def Calculate_Wine_Classifiers():
    data, target = Get_Wine()
    classifier = BaggingClassifier()
    baggingResults = Get_Stats(data, target, classifier)

    data, target = Get_Wine()
    classifier = AdaBoostClassifier()
    adaBoostResults = Get_Stats(data, target, classifier)

    data, target = Get_Wine()
    classifier = RandomForestClassifier()
    randomForestResults = Get_Stats(data, target, classifier)

    data, target = Get_Wine()
    classifier = HPClassifier()
    hpResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [1, 3, 5, 7]})

    print("\n")

    print("------------------------ WINE --------------------------")

    print("Bagging: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          baggingResults[1:]], headers="firstrow"))

    print("AdaBoost: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          adaBoostResults[1:]], headers="firstrow"))

    print("RandomForest: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          randomForestResults[1:]], headers="firstrow"))

    print("Heterogeneous Pooling: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          hpResults[1:]], headers="firstrow"))

    return [baggingResults, adaBoostResults, randomForestResults, hpResults]


def Calculate_Breast_Cancer_Classifiers():
    data, target = Get_Breast_Cancer()
    classifier = BaggingClassifier()
    baggingResults = Get_Stats(data, target, classifier)

    data, target = Get_Breast_Cancer()
    classifier = AdaBoostClassifier()
    adaBoostResults = Get_Stats(data, target, classifier)

    data, target = Get_Breast_Cancer()
    classifier = RandomForestClassifier()
    randomForestResults = Get_Stats(data, target, classifier)

    data, target = Get_Breast_Cancer()
    classifier = HPClassifier()
    hpResults = Get_Stats(data, target, classifier, {
                          'classifier__n_Samples': [1, 3, 5, 7]})

    print("\n")

    print("------------------------ BREAST CANCER --------------------------")

    print("Bagging: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          baggingResults[1:]], headers="firstrow"))

    print("AdaBoost: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          adaBoostResults[1:]], headers="firstrow"))

    print("RandomForest: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          randomForestResults[1:]], headers="firstrow"))

    print("Heterogeneous Pooling: ")
    print(tabulate([["Média", "STD", "Inferior", "Superior"],
          hpResults[1:]], headers="firstrow"))

    return [baggingResults, adaBoostResults, randomForestResults, hpResults]


def Get_Accuracies(results):
    accuracies = []

    for item in results:
        accuracies.append(item[0])

    return accuracies
