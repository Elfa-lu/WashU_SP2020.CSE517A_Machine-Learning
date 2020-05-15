"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""
import numpy as np
from cvxopt import matrix

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]

    # Q = K.dot(yTr.T * yTr)
    # p = -np.ones((n,1))
    #
    # G = yTr.T
    # h = C * np.ones((n,1))
    #
    # A = np.zeros((n,1))
    # b = np.zeros((1,1))

    Q = yTr * K * (yTr.T)  # K.dot(yTr.T * yTr)
    p = -np.ones((n, 1))

    G = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    h = np.concatenate((np.ones((n,1))*C, np.zeros((n, 1))), axis=0)

    A = yTr.T
    b = np.zeros((1,1))


    return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

