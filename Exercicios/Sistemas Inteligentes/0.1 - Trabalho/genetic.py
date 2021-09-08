import time
import random
import math
from sklearn.metrics import accuracy_score
import pandas as pd
import collections as cl
import numpy as np

def Get_State_Size(state):
    size = 0
    for nr_Selected in state:
        if nr_Selected == 1:
            size += 1

    return size

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

def Evaluate_Population (pop, classifiers, data, target):
    eval = []
    for s in pop:
        eval = eval + [(Evaluate_State(s, classifiers, data, target), s)]
    return eval   

def Get_States_Total_Value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum

def Roulette_Construction(states):
    aux_states = []
    roulette = []
    total_value = Get_States_Total_Value(states)

    for state in states:
        value = state[0]
        if total_value != 0:
            ratio = value/total_value
        else:
            ratio = 1
        aux_states.append((ratio,state[1]))
 
    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value,state[1])
        roulette.append(s)
    return roulette

def Roulette_Run (rounds, roulette):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0,1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected

def Generate_Initial_State(classifiers):
    initial_state = []
    zero_state = [0] * len(classifiers)
    for i in range(len(classifiers)):
        initial_state.append(random.randint(0,1))

    if np.array_equal(zero_state, initial_state):
      initial_state = Generate_Initial_State(classifiers)

    return initial_state

def First(x):
    return x[0]

def Get_Selection(value_population,n):
    aux_population = Roulette_Construction(value_population)
    new_population = Roulette_Run(n, aux_population)
    return new_population

def Get_Crossover(dad,mom):
    r = random.randint(0, len(dad) - 1)
    son = dad[:r]+mom[r:]
    daug = mom[:r]+dad[r:]
    return son, daug

def Get_Mutation (indiv):
    individual = indiv.copy()
    rand = random.randint(0, len(individual) - 1)
    if individual[rand] > 0:
        r = random.uniform(0,1)
        if r > 0.5:
            individual[rand] = 1
        else:
            individual[rand] = 0
    else:
        individual[rand] = 1
        
    return individual

def Get_Initial_Population(n, classifiers):
    pop = []
    count = 0
    while count < n:
        individual = Generate_Initial_State(classifiers)
        pop = pop + [individual]
        count += 1
    return pop

def Get_Convergent(population):
    conv = False
    if population != []:
        base = population[0]
        i = 0
        while i < len(population):
            if base != population[i]:
                return False
            i += 1
        return True

def Get_Elitism (val_pop, pct):
    n = math.floor((pct/100)*len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted (val_pop, key = First, reverse = True)[:n]
    elite = [s for v,s in val_elite]
    return elite

def Get_Crossover_Step (population, crossover_ratio):
    new_pop = []
    
    for _ in range (round(len(population)/2)):
        rand = random.uniform(0, 1)
        fst_ind = random.randint(0, len(population) - 1)
        scd_ind = random.randint(0, len(population) - 1)
        parent1 = population[fst_ind] 
        parent2 = population[scd_ind]

        if rand <= crossover_ratio:
            offspring1, offspring2 = Get_Crossover(parent1, parent2)            
            
            if Get_State_Size(offspring1) == 0:
                offspring1 = parent1
            if Get_State_Size(offspring2) == 0:
                offspring2 = parent2
        else:
            offspring1, offspring2 = parent1, parent2
                
        new_pop = new_pop + [offspring1, offspring2]
        
    return new_pop

def Get_Mutation_Step (population, mutation_ratio):
    ind = 0
    for individual in population:
        rand = random.uniform(0, 1)

        if rand <= mutation_ratio:
            mutated = Get_Mutation(individual)
            if Get_State_Size(mutated) > 0:
              population[ind] = mutated
                
        ind+=1
        
    return population   

def Genetic (classifiers, data, target, pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct):
    start = time.process_time()
    opt_state = [0] * len(classifiers)
    opt_value = 0
    pop = Get_Initial_Population(pop_size, classifiers)
    conv = Get_Convergent(pop)
    iter = 0    
    end = 0

    while not conv and iter < max_iter and end-start <= max_time:
        val_pop = Evaluate_Population (pop, classifiers, data, target)
        new_pop = Get_Elitism (val_pop, elite_pct)
        best = new_pop[0]
        val_best = Evaluate_State(best, classifiers, data, target)

        if (val_best > opt_value):
            opt_state = best
            opt_value = val_best

        selected = Get_Selection(val_pop, pop_size - len(new_pop)) 
        crossed = Get_Crossover_Step(selected, cross_ratio)
        mutated = Get_Mutation_Step(crossed, mut_ratio)
        pop = new_pop + mutated
        conv = Get_Convergent(pop)
        iter+=1
        end = time.process_time()
        
  
    return opt_state