
# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Import PySwarms
import pyswarms as ps

n_inputs = 30
n_hidden = 60
n_classes = 2

num_samples = 569


def logits_function(p, X):
    # Roll-back the weights and biases
    W1 = p[0:1800].reshape((n_inputs,n_hidden))
    b1 = p[1800:1860].reshape((n_hidden,))
    W2 = p[1860:1980].reshape((n_hidden,n_classes))
    b2 = p[1980:1982].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
    return logits          # Logits for Layer 2


# Forward propagation
def forward_prop(params, X,y):
    logits = logits_function(params, X)

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    corect_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(corect_logprobs) / num_samples

    return loss


def Get_PSO(f):
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=1000)
    return pos


def predict(pos, X):
    logits = logits_function(pos, X)
    y_pred = np.argmax(logits, axis=1)
    return y_pred


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

labelencoder = LabelEncoder()

def Start_Breast_Cancer():
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

    def f(x):
        n_particles = x.shape[0]
        j = [forward_prop(x[i], data, target) for i in range(n_particles)]
        return np.array(j)

    pos = Get_PSO(f)
    scores = (predict(pos, data) == target)

    print("\n Breast Cancer \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


Start_Breast_Cancer()
