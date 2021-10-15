import time
import random
import math
from sklearn.metrics import accuracy_score
import collections as cl
import pandas as pd
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

def Generate_Initial_State(classifiers):
    initial_state = []
    zero_state = [0] * len(classifiers)
    for i in range(len(classifiers)):
        initial_state.append(random.randint(0,1))

    if np.array_equal(zero_state, initial_state):
      initial_state = Generate_Initial_State(classifiers)

    return initial_state

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

    if(state == []):
      return 0
    
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

def States_Total_Value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum

def Random_State(states):
    index = random.randint(0,len(states)-1)
    return states[index]

def Change_State(state,position,value):
    state[position] = value
    return state

def Generate_Neighborhood(state):
    neighborhood = []
    zero_state = [0]*len(state)

    for i in range(len(state)):
        aux = state.copy()
        new_state = Change_State(aux,i,1)
        if not np.array_equal(state, new_state):
          neighborhood.append(new_state)
    for i in range(len(state)):
        aux = state.copy()
        new_state = Change_State(aux,i,0)
        if not np.array_equal(state, new_state):
          if not np.array_equal(zero_state, new_state):
             neighborhood.append(new_state)

    return neighborhood

def Change_Probability(value,best_value,t):
    p = 1/(math.exp(1)**((best_value-value)/t))
    r = random.uniform(0,1)
    if r < p:
        return True
    else:
        return False

def Simulated_Annealing(t,alfa,classifiers,data, target,iter_max,max_time):
    state = Generate_Initial_State(classifiers)
    solution = state
    max_value = Evaluate_State(solution, classifiers, data, target)
    start = time.process_time()
    end = 0
    
    while t >= 1 and end-start <= max_time:        
        
        for _ in range(iter_max):    
            neighborhood = Generate_Neighborhood(state)
            if neighborhood == []:
                return solution,max_value,Get_State_Size(solution)                
            aux = Random_State(neighborhood)
            aux_value = Evaluate_State(aux, classifiers, data, target)
            aux_size = Get_State_Size(aux)
            state_value = Evaluate_State(state, classifiers, data, target)
            if aux_value > state_value:
                state = aux
                if aux_value > max_value:
                    solution = aux
                    max_value = aux_value
            else:
                if Change_Probability(aux_value,state_value,t):
                    state = aux
        t = t*alfa
        end = time.process_time()

    return solution