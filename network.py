import numpy as np
from .layer import layer
from .conv_layer import conv_layer
from .pool_layer import pool_layer
from tqdm import tqdm
import math

class network(object):

    def __init__(self):
        """
        A neural network that can contain multiple; simple or convolutional neural layers
        """
        self.layers = []
    
    def add_layer(self,input_size,output_size,activation="sigmoid"):
        """
        Adds a fully connected layer to the network
        """
        self.layers.append(layer(input_size,output_size,activation))

    def add_conv(self,input_shape,output_shape,activation="sigmoid"):
        """
        Adds a convolutional layer to  the network\n
        Param:\n
        input_shape: tuple containing-(channels,height,width) of input\n
        output_shape: tuple containing-(number of filters,height,width) of output\n
        activation:activation
        """
        self.layers.append(conv_layer(input_shape,output_shape,activation))

    def add_pool(self,input_shape,output_shape):
        """
        Adds a pooling layer will be integrated with conv layer soon..:)
        """
        self.layers.append(pool_layer(input_shape,output_shape))

    def predict(self,inputs,getData=False):
        """
        returns the output for the given inputs
        """
        if getData==True:
            data = []
            data.append(inputs)
            for l in self.layers:
                inputs = l.forward_pass(inputs)
                data.append(inputs)
            return data
        else:
            for l in self.layers:
                inputs = l.forward_pass(inputs)
            return inputs

    def train(self,data,batch_size,step_size=0.01,epoch=1,print_acc=None):
        """
        trains the network using gradient decent\n
        it prints training and test data acuuracy every epoch\n
        Param:\n
        data: tuple containing two or four numpy arrays: train_X,train_Y,test_X,test_Y\n
        batch_size: number of dtata samples evaluated per descent step\n
        step_size: step size is calcuated using step_size*exp(-10*accuracy)\n
        epoch: number of iterations through the training data if none given defaults to one\n
        """
        if len(data)==4:
            train_X,train_Y,test_X,test_Y = data
        else:
            train_X,train_Y = data
        
        for count in range(epoch):
            for i in tqdm(range(train_Y.shape[0]//batch_size)):
                for j in range(batch_size):
                    neurons = self.predict(train_X[i*batch_size +j],True)
                    dif = neurons[-1] - train_Y[i*batch_size +j]
                    for k in range(len(self.layers)):
                        dif = self.layers[-(1+k)].back_prop(dif,neurons[-(2+k)],neurons[-(1+k)],step_size)
                for l in self.layers:
                    l.grad_param(batch_size)
            print("epoch ",count+1," completed")
            if print_acc:
                print_acc(self,data)


