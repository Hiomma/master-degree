# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmax
from numpy import max
from numpy import around
import random
from matplotlib import pyplot
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def Evaluate_State(state, data, target):
    #finding the distance between centroids and all the data points
    distances = cdist(data, state,'euclidean') #Step 2

    k = len(state)

    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3

    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(10):
        _state = state
        state = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            if(len(data[points == idx]) == 0):
                state.append(_state[idx])
            else:
                state.append(data[points == idx].mean(axis=0))

        state = np.vstack(state) #Updated Centroids

        distances = cdist(data, state,'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return accuracy_score(target, points), state

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    array = []
    for i in range(len(mutated)):
        auxiliar = []
        for g in range(len(bounds)):
            auxiliar.append(clip(mutated[i][g],bounds[g][0],bounds[g][1]))
        if not np.array_equal(array, []):
            array = np.vstack([array, auxiliar])
        else:
            array.append(auxiliar)
    mutated_bound = array
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(len(mutated))
    # generate trial vector by binomial crossover
    trial = np.array([mutated[i] if p[i] < cr else target[i] for i in range(len(mutated))])
    return trial

def differential_evolution(pop_size, iter, F, cr, data, target, centroids):
    # initialise population of candidate solutions randomly within the specified bounds
    bounds = np.array([[min(data[:,i]),max(data[:,i])] for i in range(data.shape[1])])

    pop = np.array([[[random.uniform(bounds[i][0], bounds[i][1]) for i in range(data.shape[1])] for _ in range(centroids)] for _ in range(pop_size)])
    pop = np.squeeze(pop)
    # evaluate initial population of candidate solutions
    obj_all = np.array([Evaluate_State(ind, data, target) for ind in pop])
    # find the best performing vector of initial population
    best_vector = obj_all[argmax(obj_all[:,0]),1]
    best_obj = max(obj_all[:,0])
    prev_obj = best_obj
    # initialise list to store the objective function value at each iteration
    obj_iter = list()
    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = Evaluate_State(pop[j], data, target)
            # compute objective function value for trial vector
            obj_trial = Evaluate_State(trial, data, target)
            # perform selection
            if obj_trial[0] > obj_target[0]:
                # replace the target vector with the trial vector
                pop[j] = trial[1]
                # store the new objective function value
                obj_all[j, 0] = obj_trial[0]
        # find the best performing vector at each iteration
        best_obj = max(obj_all[:,0])

        if best_obj > prev_obj:
            best_vector = obj_all[argmax(obj_all[:,0]),1]
            prev_obj = best_obj
            obj_iter.append(best_obj)
            # report progress at each iteration
        #   print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, prev_obj, obj_iter]


# define population size
pop_size = 10

# define number of iterations
iter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

labelencoder = LabelEncoder()

def Start_Iris(b_ES = True):
    data = pd.read_csv('databases/iris.data', names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    target = labelencoder.fit_transform(data['class'].values)
    data = data.drop('class', axis = 1).values

    scores = []
    vectors = []

    centroids = len(set(target))

    for i in range(10):
            solution = differential_evolution(pop_size, iter, F, cr, data, target, centroids)
            scores.append(solution[1])
            vectors.append(solution[0])

    scores = np.array(scores)

    print("\n Iris \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print("Min: ", scores.min())
    print("Max: ", scores.max())
    print("Best vector", vectors[scores.argmax()])

def Start_Wine(b_ES = True):
    data = pd.read_csv('databases/wine.data', names = ['class', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'diluted', 'proline'])

    target = data['class'].values
    data_drop = data.drop('class',axis=1)
    data = data_drop.values

    sc = StandardScaler()
    data = sc.fit_transform(data)

    scores = []
    vectors = []

    centroids  = len(set(target))

    for i in range(10):
            solution = differential_evolution(pop_size, iter, F, cr, data, target, centroids)
            scores.append(solution[1])
            vectors.append(solution[0])

    scores = np.array(scores)

    print("\n Wine \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print("Min: ", scores.min())
    print("Max: ", scores.max())
    print("Best vector", vectors[scores.argmax()])

def Start_Breast_Cancer(b_ES = True):
    data = pd.read_csv('databases/breast-cancer.data', names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                                                                'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                                                                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                                                                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                                                                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                                                                'fractal_dimension_se', 'radius_worst', 'texture_worst',
                                                                'perimeter_worst', 'area_worst', 'smoothness_worst',
                                                                'compactness_worst', 'concavity_worst', 'concave points_worst',
                                                                'symmetry_worst', 'fractal_dimension_worst'])

    data = data.drop('id',axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

    datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:31]))
    datas.columns = list(data.iloc[:,1:31].columns)
    target = data['diagnosis']
    data = datas.values

    scores = []
    vectors = []

    centroids = len(set(target))

    for i in range(10):
            solution = differential_evolution(pop_size, iter, F, cr, data, target, centroids)
            scores.append(solution[1])
            vectors.append(solution[0])

    scores = np.array(scores)

    print("\n Breast Cancer \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print("Min: ", scores.min())
    print("Max: ", scores.max())
    print("Best vector", vectors[scores.argmax()])


Start_Iris(False)
Start_Wine(False)
Start_Breast_Cancer(False)