import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Import PySwarms
import pyswarms as ps

n_inputs = 13
n_hidden = 20
n_classes = 3

num_samples = 178


def logits_function(p, X):
    # Roll-back the weights and biases
    W1 = p[0:260].reshape((n_inputs,n_hidden))
    b1 = p[260:280].reshape((n_hidden,))
    W2 = p[280:340].reshape((n_hidden,n_classes))
    b2 = p[340:343].reshape((n_classes,))

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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

labelencoder = LabelEncoder()

def Start_Wine():
    data = pd.read_csv('databases/wine.data', names = ['class', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'diluted', 'proline'])

    target = labelencoder.fit_transform(data['class'].values)
    data_drop = data.drop('class',axis=1)
    data = data_drop.values

    sc = StandardScaler()
    data = sc.fit_transform(data)

    def f(x):
        n_particles = x.shape[0]
        j = [forward_prop(x[i], data, target) for i in range(n_particles)]
        return np.array(j)

    pos = Get_PSO(f)
    scores = (predict(pos, data) == target)

    print("\n Wine \n")
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

Start_Wine()