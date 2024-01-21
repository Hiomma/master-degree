import numpy as np
import pandas as pd

all_faults = []

def start_df():
    global all_faults

    all_faults = []

    for i in range(1, 23):
        df_fault = pd.read_csv(f'./cstr/X_{i}.csv', sep=';')
        all_faults.append(df_fault.to_numpy()[:, :-4])

def _get_fault_from_array(fault):
    return all_faults[fault-1]

def get_data(selected_fault, faults = range(1, 23)):
    X_train = X_test = y_train = y_test = []

    for i in faults:
        df_fault = _get_fault_from_array(i)

        X_aux = X_aux_test = []
        data = df_fault
        X_fault = data[:, 0:-1]
        X_fault = X_fault[:,:-2]
        labels = data[:, -1]

        index_selected = (500, 800, 1000)
        index_not_selected = (500, 600, 700)

        if i == 1:
            X_aux = X_fault
            X_train = X_fault
        elif len(X_test) == 0:
            if i == selected_fault:
                X_train = X_fault[index_selected[0]:index_selected[1]]
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_test = X_fault[index_selected[1]:index_selected[2]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = X_fault[index_not_selected[0]:index_not_selected[1]]
                X_aux = X_fault[index_not_selected[0]:index_not_selected[1]]
                X_test = X_fault[index_not_selected[1]:index_not_selected[2]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]
        elif len(X_test) == 1:
            if i == selected_fault:
                X_train = np.concatenate((X_train, X_fault[index_selected[0]:index_selected[1]]))
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_test = X_fault[index_selected[1]:index_selected[2]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = np.concatenate((X_train,X_fault[index_selected[0]:index_not_selected[1]]))
                X_aux = X_fault[index_not_selected[0]:index_not_selected[1]]
                X_test = X_fault[index_not_selected[1]:index_not_selected[2]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]
        else:
            if i == selected_fault:
                X_train = np.concatenate((X_train,X_fault[index_selected[0]:index_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_selected[1]:index_selected[2]]))
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = np.concatenate((X_train,X_fault[index_not_selected[0]:index_not_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_not_selected[1]:index_not_selected[2]]))
                X_aux =X_fault[index_not_selected[0]:index_not_selected[1]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]

        if i == selected_fault:
            y_train = np.concatenate((y_train, [1 for _ in range(X_aux.shape[0])]))
            y_test = np.concatenate((y_test, [1 for _ in range(np.array(X_aux_test).shape[0])]))
        else:
            y_train = np.concatenate((y_train, np.zeros(np.array(X_aux).shape[0])))
            y_test = np.concatenate((y_test,np.zeros(np.array(X_aux_test).shape[0])))

    return X_train, y_train, X_test, y_test

def get_one_failure(selected_fault):
    X_train = X_test = y_train = y_test = []
    
    df_fault = _get_fault_from_array(selected_fault)
    data = df_fault
    X_fault = data[:, 0:-1]
    index_selected = (50, 800, 1000)

    X_train = X_fault[index_selected[0]:index_selected[1]]
    X_test = np.concatenate((X_fault[0:50],X_fault[index_selected[1]:index_selected[2]]))

    y_train = np.concatenate(([0 for _ in range(200)], [1 for _ in range(550)]))
    y_test = np.concatenate(([0 for _ in range(50)], [1 for _ in range(200)]))

    return X_train, y_train, X_test, y_test

def get_all_data(selected_fault):
    X_train = X_test = y_train = y_test = []

    for i in range(1, 23):
        df_fault = pd.read_csv(f'./cstr/X_{i}.csv', sep=';')

        X_aux = X_aux_test = []
        featname = list(df_fault.columns.values)
        data = df_fault.to_numpy()
        X_fault = data[:, 0:-1]
        labels = data[:, -1]

        index_selected = (25, 80, 100)
        index_not_selected = (25, 80, 100)

        if i == 1:
            X_aux = X_fault
            X_train = X_fault
        elif i == 2:
            if i == selected_fault:
                X_train = np.concatenate((X_train, X_fault[25:index_selected[1]]))
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_test = X_fault[index_selected[1]:index_selected[2]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = np.concatenate((X_train,X_fault[25:index_not_selected[1]]))
                X_aux = X_fault[index_not_selected[0]:index_not_selected[1]]
                X_test = X_fault[index_not_selected[1]:index_not_selected[2]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]
        else:
            if i == selected_fault:
                X_train = np.concatenate((X_train,X_fault[25:index_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_selected[1]:index_selected[2]]))
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = np.concatenate((X_train,X_fault[25:index_not_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_not_selected[1]:index_not_selected[2]]))
                X_aux =X_fault[index_not_selected[0]:index_not_selected[1]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]

        if i == selected_fault:
            y_train = np.concatenate((y_train, [1 for _ in range(X_aux.shape[0])]))
            y_test = np.concatenate((y_test, [1 for _ in range(np.array(X_aux_test).shape[0])]))
        else:
            y_train = np.concatenate((y_train, np.zeros(np.array(X_aux).shape[0])))
            y_test = np.concatenate((y_test,np.zeros(np.array(X_aux_test).shape[0])))

    return X_train, y_train, X_test, y_test

def get_one_fault(selected_fault):
    df_fault = pd.read_csv(f'./cstr/X_{selected_fault}.csv', sep=';')

    featname = list(df_fault.columns.values)
    data = df_fault.to_numpy()
    X_fault = data[:, 0:-1]
    labels = data[:, -1]

    X_train = np.concatenate((X_fault[:15], X_fault[25:80]))
    X_test = np.concatenate((X_fault[15:25], X_fault[80:100]))
    y_train = np.concatenate((np.zeros(15), [1 for _ in range(55)]))
    y_test = np.concatenate((np.zeros(10), [1 for _ in range(20)]))

    return X_train, y_train, X_test, y_test

def get_all_faulty_data(selected_fault):
    X_train = X_test = y_train = y_test = []

    for i in range(2, 23):
        df_fault = pd.read_csv(f'./cstr/X_{i}.csv', sep=';')

        X_aux = X_aux_test = []
        featname = list(df_fault.columns.values)
        data = df_fault.to_numpy()
        X_fault = data[:, 0:-1]
        labels = data[:, -1]

        index_selected = (25, 80, 100)
        index_not_selected = (25, 80, 100)

        if i == 2:
            if i == selected_fault:
                X_train = X_fault[25:index_selected[1]]
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_test = X_fault[index_selected[1]:index_selected[2]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = X_fault[25:index_not_selected[1]]
                X_aux = X_fault[index_not_selected[0]:index_not_selected[1]]
                X_test = X_fault[index_not_selected[1]:index_not_selected[2]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]
        else:
            if i == selected_fault:
                X_train = np.concatenate((X_train,X_fault[25:index_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_selected[1]:index_selected[2]]))
                X_aux = X_fault[index_selected[0]:index_selected[1]]
                X_aux_test = X_fault[index_selected[1]:index_selected[2]]
            else:
                X_train = np.concatenate((X_train,X_fault[25:index_not_selected[1]]))
                X_test = np.concatenate((X_test, X_fault[index_not_selected[1]:index_not_selected[2]]))
                X_aux =X_fault[index_not_selected[0]:index_not_selected[1]]
                X_aux_test = X_fault[index_not_selected[1]:index_not_selected[2]]

        if i == selected_fault:
            y_train = np.concatenate((y_train, [1 for _ in range(X_aux.shape[0])]))
            y_test = np.concatenate((y_test, [1 for _ in range(np.array(X_aux_test).shape[0])]))
        else:
            y_train = np.concatenate((y_train, np.zeros(np.array(X_aux).shape[0])))
            y_test = np.concatenate((y_test,np.zeros(np.array(X_aux_test).shape[0])))

    return X_train, y_train, X_test, y_test

def get_one_vs_one_data(rangeFor = range(2, 23)):
    X_train = X_test = y_train = y_test = []

    for i in rangeFor:
        df_fault = pd.read_csv(f'./cstr/X_{i}.csv', sep=';')

        X_aux = X_aux_test = []
        featname = list(df_fault.columns.values)
        data = df_fault.to_numpy()
        X_fault = data[:, 0:-1]
        labels = data[:, -1]

        indexes = (25, 80, 100)

        if i == 2:
                X_train = X_fault[25:indexes[1]]
                X_aux = X_fault[indexes[0]:indexes[1]]
                X_test = X_fault[indexes[1]:indexes[2]]
                X_aux_test = X_fault[indexes[1]:indexes[2]]
        else:
                X_train = np.concatenate((X_train,X_fault[25:indexes[1]]))
                X_test = np.concatenate((X_test, X_fault[indexes[1]:indexes[2]]))
                X_aux =X_fault[indexes[0]:indexes[1]]
                X_aux_test = X_fault[indexes[1]:indexes[2]]

        y_train = np.concatenate((y_train, [i for _ in range(X_aux.shape[0])]))
        y_test = np.concatenate((y_test, [i for _ in range(np.array(X_aux_test).shape[0])]))

    return X_train, y_train, X_test, y_test
#%%
