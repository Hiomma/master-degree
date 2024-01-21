# https://pandas.pydata.org/pandas-docs/stable/index.html
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report,\
    accuracy_score, f1_score
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
import shap
'''
sys.path.append('/home/thomas/Nextcloud2/book/soft/lib/')
sys.path.append('E:/Nextcloud2/book/soft/lib/')
from LinearMachine import LinearMachine
'''

from sklearn.neighbors import KNeighborsClassifier

def get_shap_contribution(model, X_train, y_train, X_test, sorting = False):
    model.fit(X_train, y_train)

    # Set the explainer with the model + shap
    explainer = shap.TreeExplainer(model)

    # Return the shap values with the X_test dataset
    shap_values = explainer.shap_values(X_test)

    # Constributions
    vals = np.abs(shap_values).mean(0)

    print(X_test.shape[1])
    print(vals.shape)

    # Returning the values as DataFrame
    feature_importance = pd.DataFrame(list(zip([i for i in range(X_test.shape[1])], sum(vals))), columns = ['col_name','feature_importance_vals'])

    if sorting:
        feature_importance.sort_values(by = ['feature_importance_vals'], ascending = False,inplace = True)

    feature_importance.set_index('col_name', inplace = True)

    return feature_importance

def read_X(fname='X.csv', sep=';'):
    """Read data matrix into a pandas dataframe."""
    df = pd.read_csv('X.csv', sep=';')

    # print(df)
    # print('index: ', df.index)
    # print('columns: ', df.columns)
    # print('info: ', df.info)
    # print('shape: ', df.shape); input('...')

    print(tabulate(df, headers='keys', tablefmt='psql'))
    return df


def dataframe2sklearn(df):
    """Convert a pandas dataframe into sklearn readable format X, y, Y."""
    featname = list(df.columns.values)
    data = df.to_numpy()
    X = data[:, 0:-1]
    labels = data[:, -1]

    # print('data=\n', data, 'shape=', data.shape)
    # print('X=\n', X, 'shape=', X.shape)
    # print('featname=\n', featname)
    # print('labels=\n', labels, 'shape=', labels.shape)
    return X, labels, featname


def ploty_signals(X, labels, featname):
    """Plot multichannel signal."""
    n, d = X.shape

    # plot the multi-channel signal
    x = np.linspace(0, n-1, n)
    numsensors = d
    # tex_setup(usetex=usetex)
    fig, ax = plt.subplots(numsensors, 1, sharex=True)

    def cm2inch(cm): return cm/2.54
    widthcm = 15  # Real sizes later in the LaTeX file
    heigthcm = 25
    fig.set_size_inches([cm2inch(widthcm), cm2inch(heigthcm)])
    ax[numsensors-1].set_xlabel('$t$')
    allsignal = X

    # https://matplotlib.org/3.1.1/api/cm_api.html#matplotlib.cm.get_cmap
    colors = plt.colormaps['tab20']

    # Analyze transitions between labels
    condition_change = []
    t = 0
    while t < n-1:
        transition = labels[t] is not labels[t+1]
        if transition:
            # ax.axvline(x=t)
            condition_change.append(t)
            print('t=%5d label transition: %s ===>%s' %
                  (t, labels[t], labels[t+1]))
        t += 1
    print('condition_change=', condition_change)

    for j in range(numsensors):
        # s = numsensors - j - 1
        s = j
        ax[s].plot(x, allsignal[:, s], linewidth=1.00,
                   color=colors(s/numsensors))
        # ylabel = '$x_{' + str(s+1) + '} =$' + featname[s]
        # ylabel = featname[s]; usetex=None
        ylabel = '$'+featname[s]+'$'
        usetex = True
        ax[s].tick_params(axis='both', which='major', labelsize=7)
        ax[s].tick_params(axis='both', which='minor', labelsize=7)

        ax[s].set_ylabel(ylabel, rotation=0, #labelpad=10,
                         verticalalignment='center',
                         horizontalalignment='right',
                         usetex=usetex, fontsize=8)
        for i in range(len(condition_change)):
            xpos = condition_change[i]
            ax[s].axvline(x=xpos, linestyle='-',
                          linewidth=1.0, label='Transição')
            if s is numsensors-1:
                ylim = ax[s].get_ylim()
                yoff = ylim[0] - 2.5
                # print('ylim=', ylim)
                ax[s].text(xpos, yoff,
                           'fault $'+str(labels[condition_change[i]+1])+'$',
                           horizontalalignment='center',
                           verticalalignment='top', color='red', rotation=90,
                           fontsize=7)

    #plt.tight_layout()
    # dropfigdir = None
    figname = 'cstr_signals.png'  # .pgf
    print('Saving individual signal plot in ', figname)
    # fig.savefig(figname, bbox_inches='tight')
    plt.show()


def pair_plot(df, featname):
    """Plot pairwise 2-D feature space."""
    # Seaborn does not work with np arrays. We need a Dataframe
    df = df.set_axis(featname, axis='columns')

    # print('df.columns=\n', df.columns)

    print('Generating pair plot ...')
    # kde = Kernel Density Estimation, based on Gaussian kernels
    kernel_wid_1D = 0.5
    kernel_wid_2D = 1.0
    pp = sns.pairplot(df, hue=featname[-1], diag_kind='kde')  # ,
                # diag_kws={'bw_adjust': kernel_wid_1D})
    pp.map_upper(sns.scatterplot)
    pp.map_lower(sns.kdeplot, linewidths=0.5)
    pp.map_diag(sns.kdeplot, linewidth=0.5)
    # pp.map_lower(sns.kdeplot, bw_adjust=kernel_wid_2D, linewidths=0.5)
    #pp.map_diag(sns.kdeplot, bw_adjust=kernel_wid_1D, linewidth=0.5)

    # pp.map_diag(sns.kdeplot, linewidth=0.5)
    figname = 'cstr_pairplot.svg'  # .pgf
    print('Saving pair plot in ', figname)
    # plt.savefig(figname, bbox_inches='tight')
    plt.show()
    # pair_plor documentation -> https://seaborn.pydata.org/generated/seaborn.pairplot.html
    # https://seaborn.pydata.org/generated/seaborn.kdeplot.html


def numeric_labels(y):
    """Return numeric labels."""
    # One-Hot if more than two classes, bianry for two labels
    classes = np.unique(y)
    numclasses = len(classes)
    if numclasses == 2:
        neg_label = -1
    else:
        neg_label = 0
    lb = LabelBinarizer(neg_label=neg_label)
    Y = lb.fit_transform(y)
    # print('Y=\n', Y, 'shape=', Y.shape)
    return Y


def resubstitution(X, y, featname):
    """Classification with resubstitution, Linear Machine classifier."""

    classes = np.unique(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    '''
    model = LinearMachine()
    model.fit(X, y)
    params = model.get_params(deep=True)
    W, bias, weights = params['W'], params['bias'], params['weights']
    print('W=\n', W)
    print('bias=\n', bias)
    print('weights=\n', weights)
    '''
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    y_pred = model.predict(X)

    print('\n==========> Resubstitution of training data:\n')
    print('Classification Report for all features and all classes: ')
    print(classification_report(y, y_pred, target_names=classes, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y, y_pred)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y, y_pred))


def k_fold(X, y, featname, K=10):
    """Classification with k-fold."""

    classes = np.unique(y)
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=K, shuffle=False)
    y_pred_overall = []
    y_test_overall = []

    #model = LinearMachine()
    model = KNeighborsClassifier(n_neighbors=3)

    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Y_PRED -----------------")
        print(y_pred)

        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    accuracy = 100*accuracy_score(y_test_overall, y_pred_overall)
    print('\n==========> K-fold cross validation:\n')
    print('Model ', K, '- Fold Classification Report: ')
    print(classification_report(y_test_overall, y_pred_overall,
                                target_names=classes, digits=3))
    print('Accuracy=', '%.2f %%' % accuracy)
    print('Macro-averaged F1=', '%.3f'
          % (f1_score(y_test_overall, y_pred_overall, average='macro')))
    print('Micro-averaged F1=', '%.3f' %
          (f1_score(y_test_overall, y_pred_overall, average='micro')))
    print('Model Confusion Matrix: ')
    print(confusion_matrix(y_test_overall, y_pred_overall))

def random_forest(X):
    X_fault = X
    index_selected = (200, 800, 1000)

    X_train = X_fault[index_selected[0]:index_selected[1]]
    X_test = np.concatenate((X_fault[0:200], X_fault[index_selected[1]:index_selected[2]]))

    y_train = np.concatenate(([0 for _ in range(300)], [1 for _ in range(300)]))
    y_test = np.concatenate(([0 for _ in range(200)], [1 for _ in range(200)]))

    X_temporary = np.array([])
    X_test_temporary = np.array([])

    model = RandomForestClassifier(random_state=1234)
    contributions = get_shap_contribution(model, X_train, y_train, X_test, True)

    contribution_indexes = contributions.index.to_numpy()

    for j in range(X_train.shape[1]):
        print("Contributions")
        print(contribution_indexes)

        # Setting up a new dataset with the selected variables
        X_temporary = np.concatenate((X_temporary, np.array([X_train[:, contribution_indexes[0]]]).T), axis=1) if X_temporary.shape[0] > 0 else np.array([X_train[:,contribution_indexes[0]]]).T
        X_test_temporary = np.concatenate((X_test_temporary, np.array([X_test[:, contribution_indexes[0]]]).T), axis=1) if X_test_temporary.shape[0] > 0 else np.array([X_test[:, contribution_indexes[0]]]).T

        print(contribution_indexes)

        randomForest = RandomForestClassifier(random_state=1234)
        randomForest.fit(X_temporary, y_train)
        predicted = randomForest.predict(X_test_temporary)

        f1_aux = f1_score(y_test, predicted, average="weighted")
        print(f1_aux)
        accuracy = accuracy_score(y_test, predicted)
        print(accuracy)

        contribution_indexes = np.delete(contribution_indexes, 0)

        print(predicted)

def main():
    """Execute main program."""
    df = read_X()
    X, y, featname = dataframe2sklearn(df)
    X = X[:, :-4]
    ploty_signals(X, y, featname)
    # pair_plot(df, featname)
    resubstitution(X, y, featname)
    k_fold(X, y, featname)
    random_forest(X)


if __name__ == "__main__":
    main()
