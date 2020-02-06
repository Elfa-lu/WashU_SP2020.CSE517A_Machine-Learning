from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit

from scipy import io
import numpy as np

# load the data:
data = io.loadmat('D:/WashU/2020SPR/CSE517 Machine Learning/data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr)


# evaluate spam filter on test set using default threshold
spamfilter(xTv,yTv,w_trained)



# from ridge import ridge
# w = np.zeros((xTr.shape[0],1))
# loss, gradient = ridge(w,xTr,yTr,1)
# print(loss)
