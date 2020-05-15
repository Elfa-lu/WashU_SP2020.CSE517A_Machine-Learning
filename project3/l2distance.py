import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))

    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y.
    X_norm = np.sum(X ** 2, axis=0)
    Z_norm = np.sum(Z ** 2, axis=0)

    X_nm = np.repeat(X_norm[:, None], m, axis=1)
    Z_nm = np.repeat(Z_norm[None, :], n, axis=0)
    D = X_nm + Z_nm - 2 * np.dot(X.T, Z)

    D[D < 0] = 0
    D = np.sqrt(D)

    return D


    # xx = np.sum(X ** 2, axis=0).reshape(-1, 1)  # size:nx1
    # xx = np.repeat(xx, m, axis=1)  # size:nxm
    # zz = np.sum(Z ** 2, axis=0).reshape(1, -1)  # size:1xm
    # zz = np.repeat(zz, n, axis=0)  # size:nxm
    # xz = X.T @ Z
    # D = xx - 2 * xz + zz
    #
    # D[D < 0] = 0
    # D = D ** 0.5