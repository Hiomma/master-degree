import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from tabulate import tabulate

from trainer import (Calculate_Breast_Cancer_Classifiers,
                     Calculate_Digits_Classifiers, Calculate_Wine_Classifiers,
                     Get_Accuracies)


def Print_Tests(newResults):
    result = []

    names = ["Bagging", "AdaBoost", "Random Forest", "HP"]

    #Faz a tabela de p-values
    for i, primaryResult in enumerate(newResults):
        line = []

        for j, secundaryResult in enumerate(newResults):
            #caso i é maior que j, é t-test
            if(i > j):
                _, pTest = ttest_rel(primaryResult[0], secundaryResult[0])
                line.append(names[i] + ' | ' + names[j] +
                            ': {:0.4f}'.format(pTest))
            #caso j é maior que i, é wilcoxon
            elif(j > i):
                _, pWilcoxon = wilcoxon(primaryResult[0], secundaryResult[0])
                line.append(names[i] + ' | ' + names[j] +
                            ': {:0.4f}'.format(pWilcoxon))
            #caso igual, ele imprime o nome
            else:
                line.append(names[i])

        result.append(line)

    print(tabulate(result))


def Get_Boxplot(results, name, names=["Bagging", "AdaBoost", "Random Forest", "HP"]):
    auxResult = []

    #coloca os nomes nos corresposdentes resultados
    for nr_Index, ds_Name in enumerate(names):
        for result in results[nr_Index]:
            auxResult.append((result, ds_Name))

    data = pd.DataFrame(auxResult, columns=["scores", "names"])

    plt.figure() 
    sns.boxplot(x='names', y='scores', data=data, showmeans=True)
    plt.savefig(name + '.png')


digitsResults = Calculate_Digits_Classifiers()
wineResults = Calculate_Wine_Classifiers()
cancerResults = Calculate_Breast_Cancer_Classifiers()

Get_Boxplot(Get_Accuracies(digitsResults), "Digits")
Get_Boxplot(Get_Accuracies(wineResults), "Wine")
Get_Boxplot(Get_Accuracies(cancerResults), "Cancer")


print("Digits:")
Print_Tests(digitsResults)
print("Wine:")
Print_Tests(wineResults)
print("Cancer:")
Print_Tests(cancerResults)
