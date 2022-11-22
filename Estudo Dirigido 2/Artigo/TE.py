import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pickle
# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1] # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)


class TE():

    XMEAS = ['Input Feed - A feed (stream 1)'	,       	#	1
        'Input Feed - D feed (stream 2)'	,       	#	2
        'Input Feed - E feed (stream 3)'	,       	#	3
        'Input Feed - A and C feed (stream 4)'	,       	#	4
        'Miscellaneous - Recycle flow (stream 8)'	,	#	5
        'Reactor feed rate (stream 6)'	,                 	#	6
        'Reactor pressure'	,                           	#	7
        'Reactor level'	,                                	#	8
        'Reactor temperature'	,                           	#	9
        'Miscellaneous - Purge rate (stream 9)'	,       	#	10
        'Separator - Product separator temperature'	,	#	11
        'Separator - Product separator level'	,       	#	12
        'Separator - Product separator pressure'	,	#	13
        'Separator - Product separator underflow (stream 10)'	,	#	14
        'Stripper level'	,                           	#	15
        'Stripper pressure'	,                           	#	16
        'Stripper underflow (stream 11)'             	,	#	17
        'Stripper temperature'	,                           	#	18
        'Stripper steam flow'	,                           	#	19
        'Miscellaneous - Compressor work'	,       	#	20
        'Miscellaneous - Reactor cooling water outlet temperature'	,	#	21
        'Miscellaneous - Separator cooling water outlet temperature'	,	#	22
        'Reactor Feed Analysis - Component A'	,	#	23
        'Reactor Feed Analysis - Component B'	,	#	24
        'Reactor Feed Analysis - Component C'	,	#	25
        'Reactor Feed Analysis - Component D'	,	#	26
        'Reactor Feed Analysis - Component E'	,	#	27
        'Reactor Feed Analysis - Component F'	,	#	28
        'Purge gas analysis - Component A'	,	#	29
        'Purge gas analysis - Component B'	,	#	30
        'Purge gas analysis - Component C'	,	#	31
        'Purge gas analysis - Component D'	,	#	32
        'Purge gas analysis - Component E'	,	#	33
        'Purge gas analysis - Component F'	,	#	34
        'Purge gas analysis - Component G'	,	#	35
        'Purge gas analysis - Component H'	,	#	36
        'Product analysis -  Component D'	,	#	37
        'Product analysis - Component E'	,	#	38
        'Product analysis - Component F'	,	#	39
        'Product analysis - Component G'	,	#	40
        'Product analysis - Component H']		#	41
			
    XMV = ['D feed flow (stream 2)'	,                 	#	1 (42)
        'E feed flow (stream 3)'	,                 	#	2 (43)
        'A feed flow (stream 1)'	,                 	#	3 (44)
        'A and C feed flow (stream 4)'	,                 	#	4 (45)
        'Compressor recycle valve'	,                 	#	5 (46)
        'Purge valve (stream 9)'	,                 	#	6 (47)
        'Separator pot liquid flow (stream 10)'	,       	#	7 (48)
        'Stripper liquid product flow (stream 11)'	,	#	8 (49)
        'Stripper steam valve'	,                           	#	9 (50)
        'Reactor cooling water flow'	,                 	#	10 (51)
        'Condenser cooling water flow'	,                 	#	11 (52)
        'Agitator speed']             # constant 50%			12 (53)


    def __init__(self):
        #print('Executing __init__() ....')
        self.rootdir = None
        self.datadir = None
        self.Xtrain = None
        self.Xtest = None
        self.fault_num = None
        self.fault_start = 160 # fault starts after 160 samples in the test file
        self.block = True
        self.featname = self.XMEAS + self.XMV
        self.extendedfeatname = list(self.featname)
        self.categoryfeatname = list(self.featname)
        self.numfeat = len(self.featname)
        self.numfeatutil = self.numfeat - 1 # without constant feature 53

        #print('TE.extendedfeatname=', self.extendedfeatname);
        #print('TE.featname=', self.featname); quit()
        self.Xtrain_all = [] # all training data files in one data matrix
        self.Xtest_all = [] # all test data files in one data matrix
        self.labels_all = [] # all labels
        self.all_fault_num = [str(i).zfill(2) for i in range(1, 22)] # '00' is no fault, but normal data, not included
        self.condition_classes = ['00'] + self.all_fault_num
        

    def read_train_test_pair(self, fault_num='01', standardize=True):
        ''' Read a training and test pair from the predefined TE datasets and put them
        into the respective data structures
        '''
        assert self.datadir is not None, 'TE data dir must not be empty'
        self.fault_num = fault_num
        ftrain = self.datadir + 'd' + fault_num + '.dat'
        ftest = self.datadir + 'd' + fault_num + '_te.dat'
        self.Xtrain = self.datacsvreadTE(ftrain)
        self.Xtest = self.datacsvreadTE(ftest)
        if standardize:
            self.standardize()

    def datacsvreadTE(self, filename, delimiter = ' ', verbose=False):
        """Read data from predefined TE file. The files have 52 columns.
           The last constant feature 53(Agitator speed) is already
           eliminated in that files.
        """
        if verbose:
            print('===> datacsvreadTE> Reading TE data from file ',
                  filename, '...')
        f = open(filename, 'rt')
        reader = csv.reader(f, delimiter=delimiter)
        row1 = next(reader)
        ncol = len(row1) # Read first line and count columns
        # count number of non-empty strings in first row
        nfeat = 0
        for j in range(ncol):
            cell = row1[j]
            if cell != '':
                nfeat = nfeat + 1
                #print('%.2e' % float(cell))

        f.seek(0)              # go back to beginning of file
        #print('ncol=', ncol, 'nfeat=', nfeat)
        
        x = np.zeros(nfeat)
        X = np.empty((0, nfeat))
        r = 0
        for row in reader:
            #print(row)
            c = 0
            ncol = len(row)
            for j in range(ncol):
                cell = row[j]
                if cell != '':
                    x[c] = float(cell)
                    #print('r=%4d' % r, 'j=%4d' % j, 'c=%4d' % c, 'x=%.4e' % x[c])
                    c = c + 1
            r = r + 1
            X = np.append(X, [x], axis=0)
            #if r > 0: # DBG
            #    break
        #print('X.shape=\n', X.shape, '\nX=\n', X, 'file=', filename); input('...')
        return X

if __name__ == "__main__":
    te = TE()
    te.rootdir = '/home/thomas/Nextcloud2/software/TE/Tennessee_Eastman/'
    te.datadir = te.rootdir + 'TE_process/data/'

    here = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(here)
    print('here=', here, 'sys.path=', sys.path) ; input('...')

    test_one_against_one(te) ; raise Exception()
    #te.read_train_test_pair_all() ; raise Exception()
    #test_read_all_pairs(te) ; raise Exception()
    #test_pair_classification(te) ; raise Exception()
    #test_train_test_pair_signal_plot(te) ; raise Exception()
    #test1(te) ; raise Exception()
     
    dropdir = te.datadir +'../data_matlab/'
    dropdir = '/tmp/'
    for fault_num in te.all_fault_num:
        te.save_matlab_train_test_pair(fault_num, discard_first_test=160, dropdir=dropdir)

