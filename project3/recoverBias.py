"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    #
    # bias = 0
    # pos = 0
    # dis = C / 2.0
    #
    # for i in range(0, len(alphas)):
    #     alpha = alphas[i]
    #     tmp = np.abs(alpha - dis)
    #     if dis > tmp:
    #         dis = tmp
    #         pos = i
    #
    # bias = yTr[pos] - (alphas.T) * (yTr.T).dot(K[:, pos])


    # This is most stable if you pick an alpha_i that is furthest from C and 0.
    y_idx = np.argmax(np.abs(alphas*(C - alphas))) #y_idx = np.argmin(np.abs(alphas - C*0.5))
    # print(alphas[90])
    # print(C)
    # print (y_idx)
    # print(alphas.shape, yTr.shape)
    # print(K[y_idx, :])
    bias = yTr[y_idx] - np.multiply(yTr, alphas).T.dot(K[y_idx, :]) #(1.0 / yTr[y_idx])
    #print(bias)

    return bias
    
