import numpy as np
import math
class conv_layer(object):

    def __init__(self,n_filters,filter_shape,input_shape,pooling_shape=(1,1),stride=1,activation="sigmoid"):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pooling_shape = pooling_shape
        self.activation = activation
        h_ = int(1 + (input_shape[-2] - filter_shape[-2])/stride)
        w_ = int(1 + (input_shape[-1] - filter_shape[-1])/stride)
        self.topool_neurons = np.zeros((n_filters,h_,w_))
        h__ = math.ceil(h_/pooling_shape[0])
        w__ = math.ceil(w_/pooling_shape[1])
        self.neurons = np.zeros((n_filters,h__,w__))
        self.output_shape = (n_filters,h__,w__)
        self.weights = np.random.randn(n_filters,filter_shape[0],filter_shape[1],filter_shape[2])
        self.bias = np.random.randn(n_filters)
        self.stride = stride

    def forward_pass(self,inputs):
        inputs = inputs.copy()
        inputs = inputs.reshape(self.input_shape)
        #vectorisation needed!!!!
        for i in range(self.n_filters):
            for j in range(self.topool_neurons.shape[1]):
                for k in range(self.topool_neurons.shape[2]):
                    self.topool_neurons[i][j][k] = np.sum(inputs[:,j*self.stride:\
                        j*self.stride + self.filter_shape[1],k*self.stride:\
                            k*self.stride + self.filter_shape[2]]*self.weights[i])
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    self.neurons[i][j][k] = np.max(self.topool_neurons[i][j*self.pooling_shape[0]][k*self.pooling_shape[1]])
        
        return self.neurons


a = conv_layer(64,(1,5,5),(1,28,28),(1,1))
k = a.forward_pass(np.random.randn(784))
print(k.shape)