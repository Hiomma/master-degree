from TE import TE
import numpy as np


def get_faulty_by_id(fault_id):
    te = TE()
    te.rootdir = 'te/'
    te.datadir = te.rootdir

    fault_num = str(fault_id).rjust(2, '0')

    # Get the fault by id
    te.read_train_test_pair(fault_num=fault_num, standardize=False)

    X_train_faulty = te.Xtrain
    X_test_faulty = te.Xtest

    # Get the normalized data from 00 file
    te.read_train_test_pair(fault_num='00', standardize=False)

    te.Xtrain = te.Xtrain.T

    # Setting up the dataset
    X_train = np.concatenate((te.Xtrain, X_train_faulty))
    X_test = X_test_faulty
    y_train = np.concatenate((np.zeros(te.Xtrain.shape[0]), [1 for i in range(X_train_faulty.shape[0])]))
    y_test = np.concatenate((np.zeros(160), [1 for i in range(X_test.shape[0] - 160)]))

    return X_train, y_train, X_test, y_test


def get_faulties(fault_id):
    te = TE()
    te.rootdir = 'te/'
    te.datadir = te.rootdir

    X_train = X_test = y_train = y_test = []

    for i in range(22):
        # Getting the fault by id
        te.read_train_test_pair(fault_num=str(i).rjust(2, '0'), standardize=False)

        X_aux = []

        # If it's 0, it will start both variables with normalized data
        if i == 0:
            X_aux= te.Xtrain.T[:160]
            X_train = te.Xtrain.T[:160]
        # If it's 1, it will start X_test variable with test data and if it's not the selected variable, its amount will be reduced to 40
        elif i == 1:
            if i == fault_id:
                X_train = np.concatenate((X_train, te.Xtrain))
                X_aux = te.Xtrain
                X_test = te.Xtest
            else:
                X_train = np.concatenate((X_train, te.Xtrain[:40]))
                X_aux = te.Xtrain[:40]
                X_test = te.Xtest[:40]
        # The rest of faults will be added here as before, but concatenating them
        else:

            if i == fault_id:
                X_train = np.concatenate((X_train, te.Xtrain))
                X_test = np.concatenate((X_test, te.Xtest))
                X_aux = te.Xtrain
            else:
                X_train = np.concatenate((X_train, te.Xtrain[:40]))
                X_test = np.concatenate((X_test, te.Xtest[:40]))
                X_aux = te.Xtrain[:40]

        # Setting up Y data, treating if it's a reduced variable or not (0 and the selected fault is not affected)
        if i == fault_id:
            y_train = np.concatenate((y_train, [1 for i in range(X_aux.shape[0])]))
            y_test = np.concatenate((y_test, np.zeros(160), [1 for i in range(te.Xtest.shape[0] - 160)]))
        else:
            y_train = np.concatenate((y_train, np.zeros(np.array(X_aux).shape[0])))
            if i != 0:
                y_test = np.concatenate((y_test, np.zeros(40)))

    return X_train, y_train, X_test, y_test

def get_all_faulties():
    te = TE()
    te.rootdir = 'te/'
    te.datadir = te.rootdir

    X_train = X_test = y_train = y_test = []

    for i in range(1,22):
        # Getting the fault by id
        te.read_train_test_pair(fault_num=str(i).rjust(2, '0'), standardize=False)

        if len(y_test) == 0:
            X_train = te.Xtrain
            X_test = te.Xtest[160:]
            y_train = [i for i in range(X_train.shape[0])]
            y_test = [i for i in range(X_test.shape[0])]
        else:  
            X_train = np.concatenate((X_train, te.Xtrain))
            X_test = np.concatenate((X_test, te.Xtest[160:]))
            y_train = np.concatenate((y_train, [i for i in range(te.Xtrain.shape[0])]))
            y_test = np.concatenate((y_test, [i for i in range(te.Xtest[160:].shape[0])]))

    return X_train, y_train, X_test, y_test

