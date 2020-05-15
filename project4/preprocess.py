# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray

    d, n =  np.shape(xTr)
    d_, n_ = np.shape(xTe)
    # m = np.zeros((d,1))
    # u = np.zeros((d,d))
    ## << Remove 2 lines above and insert your solution here

    #  [:None] creates an axis with length 1.
    m = np.mean(xTr, axis=1)[:, None]
    m_xTr = np.repeat(np.mean(xTr, axis=1)[:, None], n, axis=1)
    m_xTe = np.repeat(np.mean(xTr, axis=1)[:, None], n_, axis=1)

    u = np.ones((d, 1)) / np.repeat(np.std(xTr, axis=1)[:, None], n, axis=1)
    u = np.diag(np.diag(u))

    xTr = u @ (xTr - m_xTr)
    xTe = u @ (xTe - m_xTe)

    # mean = np.mean(xTr, axis=1)
    # std = np.std(xTr, axis=1)
    # u = np.diag(1 / std)
    # m = mean.reshape([d, 1])
    # xTr = u @ (xTr - m)
    # xTe = u @ (xTe - m)

    ## >>
    return xTr, xTe, u, m