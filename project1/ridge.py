
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    loss = np.sum(np.square((w.T.dot(xTr) - yTr))) + lambdaa * w.T.dot(w) # array to scalar
    gradient = 2 * (xTr.dot(xTr.T).dot(w) - xTr.dot(yTr.T) + lambdaa*w)

    return loss,gradient



# N=50
# D=5
# lambdaa = 0
# xTr=np.concatenate((np.random.randn(D,N),np.random.randn(D,N)+2),axis=1)
# yTr=np.concatenate((np.ones((1,N)),-np.ones((1,N))),axis=1)
# w=np.zeros((xTr.shape[0],1))

