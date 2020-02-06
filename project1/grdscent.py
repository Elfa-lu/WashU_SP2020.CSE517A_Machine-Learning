
import numpy as np
from scipy.misc import derivative
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector

    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    w = w0
    last_step_loss = float('inf')

    for i in range(maxiter):
        # The first parameter of grdescent is a function which takes a weight vector and returns loss and gradient.
        loss, gradient = func(w)
        # print(loss.shape)
        # include the tolerance variable to stop early if the norm of the gradient is less than the tolerance value
        if np.linalg.norm(gradient) < tolerance:
            break

        # increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
        # and decrease it by a factor 0.5 if the loss went up
        if last_step_loss >= loss:
            stepsize = stepsize * 1.01
            # print(stepsize)
        else:
            stepsize = stepsize * 0.5
        w = w - stepsize * gradient
        last_step_loss = loss  # ????


    return w

# from ridge import ridge
# f = lambda w : ridge(w,xTr,yTr,1)
# w_trained = grdescent (f,np.zeros((xTr.shape[0],1)),1e-06,1000)