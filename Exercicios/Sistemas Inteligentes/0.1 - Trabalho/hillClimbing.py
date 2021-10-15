import time
from sklearn.metrics import accuracy_score
import pandas as pd
import collections as cl
import numpy as np

def Get_Most_Frequent(column, baseFrequencies):
        frequencies = cl.Counter(column).items()
        betterFrequency = 0
        better = None
        
        for key, value in frequencies:
            if(value > betterFrequency):
                better = key
                betterFrequency = value
            elif(value == betterFrequency):
                for selfKey, selfValue in baseFrequencies.most_common():
                    if(selfKey == better):
                        break
                    if(selfKey == key):
                        better = key
                        betterFrequency = value
        return better

def Get_State_Size(state):
    size = 0
    for nr_Selected in state:
        if nr_Selected == 1:
            size += 1

    return size


def Evaluate_State(state, classifiers, data, target):
    classifiersList = []
    predictions = []
    result = []
    
    for nr_Index, b_Using in enumerate(state):
        if b_Using == 1:
            classifiersList.append(classifiers[nr_Index])

    for classifier in classifiersList:
        prediction = classifier.predict(data)
        predictions.append(prediction)


    dataFrame = pd.DataFrame(data=predictions)

    frequencies = cl.Counter(target)  

    for index in dataFrame.columns:
        column = dataFrame[index]
        choice = Get_Most_Frequent(column, frequencies)
        result.append(choice)

    return accuracy_score(target, result)


def Generate_States(initial_state):
    states = []
    for i in range(len(initial_state)):
        aux = initial_state.copy()
        aux[i] = initial_state[i] + 1
        if(aux[i] < 2):
            states.append(aux)
    return states


def Hill_Climbing(classifiers, data, target, max_time):
    start = time.process_time()
    current_state = [0]*len(classifiers)
    optimal_value = 0
    optimal_size = 0
    optimal_state = current_state
    valid_states = len(classifiers)
    end = 0
    
    while end-start <= max_time and valid_states != 0:
        possible_states = Generate_States(optimal_state)
        valid_states = len(possible_states)
        last_value = optimal_value

        for state in possible_states:
            aux_value = Evaluate_State(state, classifiers, data, target)
            aux_size = Get_State_Size(state)
            if aux_size != 0:
                if aux_value >= optimal_value:
                    optimal_value = aux_value
                    optimal_size = aux_size
                    optimal_state = state
                else:
                    valid_states = valid_states - 1
            
        end = time.process_time()

    return optimal_state
