import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    # print(w.shape, xTr.shape, yTr.shape)  (1024, 1) (1024, 4000) (1, 4000)
    # yTr:1*N    wT*xTr:1*N
    exp = np.exp(-yTr * (w.T.dot(xTr)))
    loss = np.sum((np.log(exp + 1)))

    # gradient: 1*D
    gradient = xTr.dot((-yTr*(exp/(exp+1))).T)

    return loss,gradient
