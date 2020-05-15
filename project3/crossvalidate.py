"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm


def crossvalidate(xTr, yTr, ktype, Cs, paras):

    bestC, bestP, lowest_error = 0, 0, 100 #bestC, bestP, lowest_error = 0, 0, 0

    numC = len(Cs)
    numP = len(paras)
    errors = np.zeros((numC, numP))

    d, n = xTr.shape
    for i in range(numC):
        C = Cs[i]
        for j in range(numP):
            para = paras[j]
            svmclassify = trainsvm(xTr, yTr, C, ktype, para)
            # print(svmclassify(xTr).shape)
            # print(yTr.shape)
            errors[i,j] = sum(svmclassify(xTr).T!=yTr)/n

            if lowest_error > errors[i,j]:
                bestC = C
                bestP = para
                lowest_error = errors[i,j]

    return bestC, bestP, lowest_error, errors


    