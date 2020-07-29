import numpy as np
from .layer import layer
import math
class network(object):

    def __init__(self):
        """
        A neural network that can contain multiple, simple or convolutional neural layers
        """
        self.layers = []

    def init_from_file(self,file):
        """
        reads pretrained weights and biases from file
        """
        nlayers = int(file.readline())
        dim1 = int(file.readline())
        dim2 = int(file.readline())
        self.layers.append(layer(dim2,-1))
        for i in range(nlayers-1):
            weights = np.zeros((dim1,dim2))
            bias = np.zeros(dim1)
            for j in range(weights.shape[0]):
                for k in range(weights.shape[1]):
                    weights[j][k] = float(file.readline())
            dim2 = int(file.readline())
            for j in range(bias.shape[0]):
                bias[j] = float(file.readline())
            self.layers.append(layer.layer(dim2,self.layers[-1].neurons.size))
            self.layers[-1].change_param(weights,bias)
            if i!=nlayers-2:
                dim1 = int(file.readline())
                dim2 = int(file.readline())
    
    def add_layer(self,layer_size,activation="sigmoid"):
        """
        Adds a layer to the network
        Param:
        layer_size: number of neurons in the layer
        activation: type of neural acivation : 'sigmoid','ReLU','LeakyReLU'
        """
        if self.layers.__len__()>0:
            self.layers.append(layer(layer_size,self.layers[-1].size,activation))
        else:
            self.layers.append(layer(layer_size,-1,activation))

    def predict(self,inputs):
        """
        returns the output for the given inputs
        """
        if inputs.size == self.layers[0].neurons.size:
            self.layers[0].set_neurons(inputs)
        for i in range(self.layers.__len__() -1):
            inputs = self.layers[i+1].forward_pass(inputs)
        return inputs

    def check_accuracy(self,img,lab):
        """
        returns the number of correct predictions made from given data
        """
        count=0
        error = 0
        for i in range(img.shape[0]):
            predicted = self.predict(img[i])[0]
            p = lab[i]
            error += math.fabs(p-predicted)
        error /= img.shape[0]
        return error

    def train(self,data,batch_size,step_size=0.01,epoch=1):
        """
        trains the network using gradient decent
        it prints training and test data acuuracy every epoch
        Param:
        data: tuple containing two or four numpy arrays: train_img,train_lab,test_img,test_lab
        batch_size: number of dtata samples evaluated per descent step
        step_size: step size is calcuated using step_size*exp(-10*accuracy)
        epoch: number of iterations through the training data if none given defaults to one
        """
        if data.__len__()==4:
            train_img,train_lab,test_img,test_lab=data
        else:
            train_img,train_lab=data
        for count in range(epoch):
            accuracy = self.check_accuracy(train_img,train_lab)
            #ss = step_size*math.exp(-10*accuracy)
            ss = step_size
            print("training accuracy:\t",accuracy)
            if data.__len__()==4:
                print("test accuracy:\t\t",self.check_accuracy(test_img,test_lab))
            for i in range(train_img.shape[0]//batch_size):
                for j in range(batch_size):
                    dif = self.predict(train_img[i*batch_size +j]) - train_lab[i*batch_size +j]
                    for k in range(self.layers.__len__()-2,-1,-1):
                        dif = self.layers[k+1].back_prop(dif,self.layers[k],ss)
                for j in range(self.layers.__len__()-1):
                    self.layers[j+1].grad_param(batch_size)


