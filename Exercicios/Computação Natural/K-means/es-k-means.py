
# evolution strategy (mu + lambda) of the ackley objective function
from numpy import asarray
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from scipy.spatial.distance import cdist
from numpy.random import seed
from sklearn.metrics import accuracy_score
import pandas as pd
import random
import numpy as np

# objective function
def objective(state, data, target):
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

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
    #print("Point:", point)
    #print("Bounds:", bounds)

    # enumerate all dimensions of the point
    for d in range(len(point)):
        # check if out of bounds for this dimension
        for j in range(len(point[d])):
            if point[d][j] < bounds[j][0] or point[d][j] > bounds[j][1]:
                return False

    return True

# evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam, centroids, data, target):
    bounds = np.array([[min(data[:,i]),max(data[:,i])] for i in range(data.shape[1])])

    best, best_eval = 0, 0
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = [[[random.uniform(bounds[i][0], bounds[i][1]) for i in range(data.shape[1])] for _ in range(centroids)] for _ in range(lam)]
    # perform the search

    population = np.array(population)

    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = np.array([objective(c, data, target) for c in population])
        # rank scores in ascending order
        ranks = argsort(argsort(scores[:, 0]))[::-1]
        # select the indexes for the top mu ranked solutions
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]

        # create children from parents
        children = list()

        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i][0] > best_eval:
                best, best_eval = scores[i][1], scores[i][0]
            #  print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            # keep the parent
            children.append(scores[i][1])
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = scores[i][1] + randn(np.array(scores[i][1]).shape[0], np.array(scores[i][1]).shape[1]) * step_size
                children.append(child)
        # replace population with children
        population = np.array(children)

    return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([0, 8.0])
# define the total iterations
n_iter = 100
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 4
# the number of children generated by parents
lam = 20

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
            best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam, centroids, data, target)
            scores.append(score)
            vectors.append(best)

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
        best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam, centroids, data, target)
        scores.append(score)
        vectors.append(best)

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
            best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam, centroids, data, target)
            scores.append(score)
            vectors.append(best)

    scores = np.array(scores)

    print("\n Breast Cancer \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print("Min: ", scores.min())
    print("Max: ", scores.max())
    print("Best vector", vectors[scores.argmax()])


Start_Iris()
Start_Wine()
Start_Breast_Cancer()