from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    loss_i = np.maximum(0, 1-yTr*(np.dot(w.T,xTr)))
    # print((lambdaa * w.T.dot(w)).shape)
    loss = np.sum(loss_i) + lambdaa * w.T.dot(w)#lambdaa*(w.T.dot(w))

    # print("hinge"+str(loss.shape))
    # loss_i>0 returns a True/False array - 0/1
    gradient = -(((loss_i>0)*yTr).dot(xTr.T)).T + 2*lambdaa*w
    #print((((loss_i>0)*yTr)).shape)
    #print(xTr.shape)
    return loss, gradient



# loss, gradient = 0, 0
#
# for (x_, y_) in zip(xTr.T, yTr.T): #xTr.shape=1024*4000 x_.shape=(1024,)
#     v = y_ * np.dot(w.T,x_)
#     #print(v.shape, w.shape, x_.shape, y_.shape)
#     loss += max(0, 1 - v)
#     gradient += 0 if v > 1 else -y_ * x_




