import numpy as np
class layer(object):

    def __init__(self,size,prev_size=-1,activation="sigmoid",trainable=True):
        """
        initializes a new layer with random weights and biases
        Param:
        size: size of layer
        prev_size: size of prev_layer used for weights
        activation: type of activation function of neurons
        trainable: trainable or not(used for discriminatory networks)
        """
        self.size = size
        self.neurons = np.zeros(size)
        self.is_input = True#for input layer of network
        self.trainable = trainable
        if prev_size!=-1:
            self.bias = np.random.randn(size) * size**(-0.5)
            self.weights = np.random.randn(size,prev_size) * size**(-0.5)
            self.activation = activation
            self.is_input = False
            #for gradient calc of weights and biases
            self.gradw = np.zeros_like(self.weights)
            self.gradb = np.zeros_like(self.bias)

    def make_non_trainable(self):
        """
        to change trainability after initialization
        """
        self.trainable=False

    def change_param(self,weights,bias):
        """
        usefull for initializing weights and bias from pretrained network
        """
        self.weights = weights
        self.bias = bias

    def set_neurons(self,inputs):
        """
        only used for initializing input layer
        Param:
        inputs: values of neurons of this layer
        """
        self.neurons = inputs
    
    def forward_pass(self,inputs):
        """
        used to predict values of current layer
        Param:
        inputs: values of neurons of previous layer
        Returns:
        outputs: values of neurons of this layer
        """
        outputs = self.weights@inputs + self.bias
        #calculating activation of neurons
        if self.activation=="sigmoid":
            outputs = 1/(1+np.exp(-outputs))
        elif self.activation=="ReLU":
            outputs = outputs * np.multiply(outputs>0,1)
        elif self.activation=="LeakyReLU":
            outputs = outputs * np.multiply(outputs>0,1) + outputs*np.multiply(outputs<0,0.01)
        self.neurons = outputs
        return outputs

    def back_prop(self,dif,prev_layer,step_size):
        """
        Back Propogation, adds to grad variables
        Param:
        dif: gradient of neurons of this layer
        prev_layer: previous layer of network for gradient of weights calc
        Returns:
        n_dif: gradient for previous layer neurons
        """
        n_dif = dif@self.weights
        if self.trainable:
            if self.activation=="sigmoid":
                dif *= self.neurons**2 - self.neurons
            elif self.activation=="ReLU":
                dif *= -np.multiply(self.neurons>0,1)
            elif self.activation=="LeakyReLU":
                dif *= np.multiply(self.neurons>0,1) + np.multiply(self.neurons<0,100)
            self.gradb += dif * step_size
            self.gradw += np.outer(dif,prev_layer.neurons) * step_size
        return n_dif

    def grad_param(self,batch_size):
        """
        Changes weights and biases using the precalculated grad variables from back_prop
        """
        self.weights+=self.gradw/batch_size
        self.bias+=self.gradb/batch_size
        self.gradw = np.zeros_like(self.weights)
        self.gradb = np.zeros_like(self.bias)

    def write_trained(self,out):
        out.write(str(self.size)+"\n")
        if self.is_input==True:
            return
        for i in self.weights:
            for j in i:
                out.write(str(j)+"\n")
        for i in self.bias:
            out.write(str(i)+"\n")