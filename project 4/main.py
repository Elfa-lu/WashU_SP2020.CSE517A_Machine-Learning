import pickle
import numpy as np

# run this to save edit best_parameters.pickle which will be used to determine performance on the autograder
# Also feel free to use this file to do any testing as it will not be called by the autograder

best_parameters = {
    'TRANSNAME' : 'ReLU',
    'ROUNDS' : 49,
    'ITER' : 10,
    'STEPSIZE' : 0.05,
    'wst' : np.array([1,20,20,20,13])
}

with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_parameters, f)



# import scipy.io as sio
# import numpy as np
# bostonData = sio.loadmat('./boston.mat')
# xTr = bostonData['xTr']
# xTe = bostonData['xTe']