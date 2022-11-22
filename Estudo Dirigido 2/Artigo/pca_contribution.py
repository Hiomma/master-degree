import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_distribution
import pandas as pd


class M:
    alfa = None
    mu = None
    st = None
    P = None
    S = None
    a = None
    n = None


class Y:
    z = None
    thr = None
    FP = None


def t2t(X):
    alfa = 0.95
    limvar = 0.95

    n, m = X.shape
    mu = np.mean(X, axis=0)
    st = np.std(X, axis=0)

    X = X - mu
    X = X / st

    _, S, v = np.linalg.svd(np.cov(X, rowvar=False))

    sd = S
    sst = np.sum(sd)
    sd = sd / sst
    ss = sd[0]

    a = 0

    while ss < limvar:
        a = a + 1
        ss = ss + sd[a]

    P = v[:, 0:(a + 1)]
    r = np.dot(X, (np.eye(m) - np.dot(P, P.T.conj())))

    M.alfa = alfa
    M.mu = mu
    M.st = st
    M.P = P
    M.S = np.diag(S)
    M.a = a
    M.n = n

    if a < m:
        M.r_var = np.var(r)
    else:
        M.r_var = []

    return M


def t2s(y, M):
    II = len(y)
    n, m = y.shape
    y = y - M.mu
    y = y / M.st

    D = np.linalg.multi_dot([M.P, np.linalg.inv(M.S[0:(M.a + 1), 0:(M.a + 1)]), (M.P.T.conj())])

    F = D
    limiar = (M.a * (M.n - 1) * (M.n + 1) / (M.n * (M.n - M.a))) * f_distribution.ppf(M.alfa, M.a, M.n - M.a)

    z = []

    for i in range(n):
        yn = y[i, :]

        z.append(np.linalg.multi_dot([yn, F, yn.T.conj()]))

    Y.z = z
    Y.thr = limiar

    return Y


def _pca_contribution(x, M):
    T = np.dot(x, M.P)
    (m, c) = M.P.shape

    ctr = np.zeros(m)
    idx = []

    t2 = (M.a * (M.n - 1) * (M.n + 1) / (M.n * (M.n - M.a))) * f_distribution.ppf(M.alfa, M.a, M.n - M.a)

    for j in range(c):
        if ((T[j] / np.sqrt(M.S[j, j])) ** 2) > ((1 / M.a) * t2):
            idx.append(j)

    cont = [];
    c = len(idx);

    if c > 0:
        for i in range(c):
            for j in range(m):
                tn = idx[i]
                ti = T[tn]
                pij = M.P[j, tn]
                aux = (ti / M.S[tn, tn]) * pij * x[j]

                if (len(cont) - 1) < i:
                    cont.append([])
                if aux > 0:
                    cont[i].append(aux)
                else:
                    cont[i].append(0)

        if c > 1:
            cont = np.sum(cont, axis=0)
        else:
            cont = np.asarray(cont).flatten()

        ctr = cont / np.sum(cont)

    return ctr


def get_pca_contribution(x, M):
    C = []
    n, m = x.shape
    x = x - M.mu

    for i in range(n):
        C.append(_pca_contribution(x[i, :], M))

    C = np.array(C)

    df = pd.DataFrame(np.vstack([np.sum(C, axis=0), [i for i in range(52)]]).T, columns=["values", "variables"])
    df.sort_values(by=["values"], ascending=False, inplace=True)

    return df


def plot_variable_with_threshold(Y):
    plt.plot(np.c_[Y.z, np.ones(len(Y.z)) * Y.thr])
    plt.axhline(Y.thr, color='k', linestyle='-', linewidth=2)
    plt.show()