# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: remus
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
#% function [as,zs]=forward_pass(W,xTr,trans_func)
#%
#% INPUT:
#% W weights (list of numpy array)
#% xTr dxn numpy array (each column is an input vector)
#% trans_func transition function to apply for inner layers
#%
#% OUTPUTS:
#%
#% as = result of forward pass 
#% zs = result of forward pass (zs[0] output layer of the forward pass) 
#%
    n = np.shape(xTr)[1]
    
    ## CHECK!  -JERRY
    
    # First, we add the constant weight
    zzs = [None]*(len(W)+1);   zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas = [None]*(len(W)+1);   aas[-1] = xTr
    
    # Do the forward process here
    for i in range(len(W)-1, -1, -1):
        # INSERT CODE
        #<<kqw
        aas[i] = W[i] @ zzs[i+1]
        zzs[i] = np.vstack((trans_func(aas[i]), np.ones((1,n))))

    # INSERT CODE: (last one is special, no transition function)
    ##<<kqw
    zzs[0]=W[0]@zzs[1]
    aas[0]=zzs[0]
    ##>>kqwend
    
    return aas, zzs
