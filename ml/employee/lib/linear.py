import numpy as np

class linear(object):

    def __init__(self,n_inputs):
        self.weights = np.zeros((n_inputs+1,1))

    def train(self,data):
        X,Y = data
        X = np.pad(X,((0,0),(1,0)),constant_values=1)
        self.weights = np.linalg.inv((X.T)@X)@(X.T)@Y
        Xpred = np.array([self.predict(X[i,1:]) for i in range(X.shape[0])]).reshape(-1,1)
        err = np.sum((Y-Xpred)*(Y - Xpred))/Y.size
        self.error = err**0.5
    
    def predict(self,X):
        X = X.reshape(1,-1)
        X = np.pad(X,((0,0),(1,0)),constant_values=1)
        Y = X@self.weights
        return Y