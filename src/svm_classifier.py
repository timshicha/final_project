""" CS-445/545 ML Final Project
    Neural Network for breast cancer detection
    
    The following code refered to equations and code snipet from
    the text book "Machine Learning: An Algorithmic Perspective" of Stephen Marsland
"""

import time, warnings
import numpy as np
import cvxopt as cvxopt
from cvxopt import solvers
from csv import reader
import matplotlib.pyplot as plt

def read_csv(file_name):
    """ Read data from csv file
            args:
                file_name: the csv file name
            return: numpy array of the data
    """
    data = []
    with open(file_name, 'r') as obj:
        csv_obj = reader(obj)
        header = next(csv_obj)
        if header != None:
            for row in csv_obj:
                x = []
                for i in range(len(row)-1):
                    x.append(float(row[i]))
                data.append(x)

    return np.array(data) 

def create_kernel(X, op, sigma, degree):
    """ Create the matrix kernel
            args:
                X: the input matrix
                op: 2-Polynomial kernel; 3-Gaussian kernel
                sigma: sigma parameter for selected kernel
                degre: degree parameter for polynomial kernel

    """
    N = X.shape[0]
    K = np.dot(X,X.T)
    xsquared = []
    if op == 2:
        K = (1. + (1./sigma)*K)**degree

    elif op == 3:
        xsquared = (np.diag(K)*np.ones((1,N))).T
        b = np.ones((N,1))
        K -= 0.5*(np.dot(xsquared,b.T) + np.dot(b,xsquared.T))
        K = np.exp(K/(2.*sigma**2))
    return K, xsquared

def classifier(X, Y, test_data, op, C, sigma, degree):
    """
        X: input data matrix
        Y: target matrix
        test_data: test data matrix
        op: 2-polynomial kernel; 3-Gaussian kernel
        C: C parameter of the quadratic solver
        sigma, degree: parameter for kernel
    """
    N = X.shape[0]
    K, xsquared = create_kernel(X, op, sigma, degree)
    P = np.dot(np.dot(Y, Y.T),K)
    q = -np.ones((N,1))
    if C == 0:
        G = -np.eye(N)
        h = np.zeros((N,1))
    else:
        G = np.concatenate((np.eye(N),-np.eye(N)))
        h = np.concatenate((C*np.ones((N,1)),np.zeros((N,1))))
    A = np.reshape(Y, (1,N))
    b = 0.0
    # Turn off cvxopt solvers progress
    solvers.options['show_progress'] = False
    solution = solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
    x_sol = np.array(solution['x'])
    # Pick support vectors by threshold
    sv_indexs = np.where(x_sol > THRESHOLD)[0]
    num_sv = len(sv_indexs)
    #print(f'Number of support vectors found: {num_sv}' )
    #print(f'Maximum coefficient: {x_sol.max()}')
    # Get support vector points from training data
    sv_X = X[sv_indexs,:]
    # Get coefficients
    coeff = x_sol[sv_indexs]
    sv_Y = Y[sv_indexs]
    b = np.sum(sv_Y)
    for n in range(num_sv):
        b -= np.sum(coeff*sv_Y*np.reshape(K[sv_indexs[n],sv_indexs],(num_sv,1)))
    b /= len(coeff)
    # Classify test data
    if op == 2:
        K = (1. + 1./sigma*np.dot(test_data,sv_X.T))**degree
        y = np.zeros((test_data.shape[0],1))
        for j in range(test_data.shape[0]):
            for i in range(num_sv):
                y[j] += coeff[i]*Y[i]*K[j,i]
            y[j] += b
        return np.sign(y)
    elif op == 3:
        K = np.dot(test_data,sv_X.T)
        c = (1./sigma * np.sum(test_data**2,axis=1)*np.ones((1,test_data.shape[0]))).T
        c = np.dot(c,np.ones((1,K.shape[1])))
        aa = np.dot(xsquared[sv_indexs],np.ones((1,K.shape[0]))).T
        K = K - 0.5*c - 0.5*aa
        K = np.exp(K/(2.*sigma**2))
        y = np.zeros((test_data.shape[0],1))
        for j in range(test_data.shape[0]):
            for i in range(num_sv):
                y[j] += coeff[i]*sv_Y[i]*K[j,i]
            y[j] += b
        return np.sign(y)

def svm_training(file_names):
    """ Read data set and classify data
            args:
                file_names: an array of data file names
    """
    for file in file_names:
        print(f'**** Using data set: {file[0]} and {file[1]}')
        training = read_csv(file[0])
        test = read_csv(file[1])
        X = training[:,1:]
        Y = training[:, 0]
        Y[Y == 0] = -1
        test_X = test[:,1:]
        test_Y = test[:,0]
        test_Y[test_Y == 0] = -1

        result = classifier(X,Y,test_X, 3, 0, SIGMA, DEGREE)
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for i in range(result.size):
            if result[i][0] > 0:
                if test_Y[i] > 0:
                    TP += 1
                else:
                    FP += 1
            else:
                if test_Y[i] < 0:
                    TN += 1
                else:
                    FN += 1
        print(f'Confusion Matrix: TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}')
        print(f'Malignant precision: {TP/(TP+FP)}')
        print(f'Benign precision: {TN/(TN+FN)}')
        print(f'Accuracy: {((TP + TN)/(TP + TN + FP + FN))*100} %')

def plot_accuracy(file_names):
    """ Plot accuracies by gamma
            args:
                file_name: training and test data file names
    """
    training = read_csv(file_names[0])
    test = read_csv(file_names[1])
    X = training[:,1:]
    Y = training[:, 0]
    Y[Y == 0] = -1
    test_X = test[:,1:]
    test_Y = test[:,0]
    test_Y[test_Y == 0] = -1
    xx = []
    yy = []
    for sig in range(30,70):
        result = classifier(X,Y,test_X, 3, 0, sig, DEGREE)
        mul = result.T[0] * test_Y
        accuracy = (np.count_nonzero(mul > 0)/mul.size)*100
        xx.append(sig)
        yy.append(accuracy)
    plt.plot(xx,yy)
    plt.xlabel("Gama")
    plt.ylabel("Accuracies (%)")
    plt.title("Accuracies by Gaussian Kernel's Gama")
    plt.show()
        
def checked_int(num):
    """ Method to check if a string can be convert to an integer
            args:
                num: a string need to convert to integer
            return: a corresponding integer, or -1 of the string is invalid
    """
    try:
        return int(num)
    except ValueError:
        return -1

""" Main Method """
SIGMA = 53
DEGREE = 3
THRESHOLD =1e-6

files = [('data/training10.csv', 'data/test10.csv'), 
        ('data/training25.csv', 'data/test25.csv'), 
        ('data/training100.csv', 'data/test100.csv')]
#suppress warnings for very low exponential parameter
warnings.filterwarnings('ignore')

while True:
    print('PROGRAM FEATURES:')
    print('1. Train and classify a set of data')
    print('2. Plot accuracy by kernel parameters')
    num_op = 0
    while True:
        num_in = input('Enter 1 or 2: ')
        num_op = checked_int(num_in)
        if num_op in [1,2]:
            break
        else:
            print('Error: input option must be 1 or 2')
    if num_op == 1:
        svm_training(files)
    else:
        plot_accuracy(files[len(files)-1])
    con_in = input('Do you want to continue? (y/n): ')
    if con_in == 'y':
        continue
    else: 
        break
