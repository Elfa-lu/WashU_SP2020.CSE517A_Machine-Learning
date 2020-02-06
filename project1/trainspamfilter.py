
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):

    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!

    f = lambda w : hinge(w,xTr,yTr,1)
    # f = lambda w : ridge(w,xTr,yTr,1)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-03,1000) #1e-06
    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained
