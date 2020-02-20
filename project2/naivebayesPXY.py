#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#    
#    Output:
#    posprob: probability vector of p(x|y=1) (dx1)
#    negprob: probability vector of p(x|y=-1) (dx1)
# =============================================================================


    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
        #Xnew = np.concatenate((X, X0), axis=1) #concatenate to column
    Ynew = np.hstack((Y, Y0))
    
    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
# =============================================================================
# fill in code here
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # count by every feature(row - axis=1)
    posprob = np.sum(Xnew[:,np.asarray(Ynew==1).flatten()], axis=1) / np.sum(Ynew==1)
    negprob = np.sum(Xnew[:,np.asarray(Ynew==-1).flatten()], axis=1) / np.sum(Ynew==-1)

    # normalization
    posprob = posprob / np.sum(posprob)
    negprob = negprob / np.sum(negprob)

    return posprob,negprob


# catesumpos = np.sum(X[:, np.asarray(Y == 1).flatten()], axis=1)
# catesumneg = np.sum(X[:, np.asarray(Y == -1).flatten()], axis=1)
# posprob = catesumpos / np.sum(catesumpos)
# =============================================================================
if __name__=="__main__":
    from genTrainFeatures import genTrainFeatures
    xTr,yTr = genTrainFeatures()
    posprob, negprob = naivebayesPXY(xTr, yTr)
    print(np.sum(xTr[:,np.asarray(yTr==1).flatten()], axis=1))
    print((yTr==1).shape)
    print(np.sum(posprob))