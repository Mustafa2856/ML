import numpy as np
class layer(object):

    def __init__(self,input_size,output_size,activation="sigmoid",trainable=True):
        """
        initializes a new layer with random weights and biases\n
        Param:\n
        input_size: size of inputs\n
        output_size: size of output\n
        activation: type of activation function of neurons\n
        trainable: trainable or not(used for discriminatory networks)\n
        """
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = trainable
        self.bias = np.random.randn(output_size)/output_size**0.5
        self.weights = np.random.randn(output_size,input_size)/output_size**0.5
        '''
        self.bias = np.zeros(output_size)
        self.weights = np.zeros((output_size,input_size))
        '''
        self.activation = activation
        self.gradw = np.zeros_like(self.weights)
        self.gradb = np.zeros_like(self.bias)

    def make_non_trainable(self):
        """
        to change trainability after initialization
        """
        self.trainable=False

    def init_param(self,weights,bias):
        """
        usefull for initializing weights and bias from pretrained network
        """
        self.weights = weights
        self.bias = bias
    
    def forward_pass(self,inputs):
        """
        used to predict values of current layer\n
        Param:\n
        inputs: values of neurons of previous layer\n
        Returns:\n
        outputs: values of neurons of this layer\n
        """
        if inputs.size != self.input_size:
            print("Input size differs from the defined size!!\n",inputs.size,self.input_size)
            return None
        outputs = self.weights@inputs + self.bias
        if self.activation=="sigmoid":
            outputs = 1/(1+np.exp(-outputs))
        elif self.activation=="ReLU":
            outputs = outputs * np.multiply(outputs>0,1)
        elif self.activation=="LeakyReLU":
            outputs = outputs * np.multiply(outputs>0,1) + outputs*np.multiply(outputs<0,0.01)
        return outputs

    def back_prop(self,dif,inputs,outputs,step_size):
        """
        Back Propogation, adds to grad variables\n
        Param:\n
        dif: gradient of neurons of this layer\n
        inputs: inputs given for which gradient is calculated\n
        Returns:\n
        n_dif: gradient for previous layer neurons\n
        """
        n_dif = dif@self.weights
        if self.trainable:
            if self.activation=="sigmoid":
                dif *= outputs - outputs**2
            elif self.activation=="ReLU":
                dif *= np.multiply(outputs>0,1)
            elif self.activation=="LeakyReLU":
                dif *= np.multiply(outputs>0,1) + np.multiply(outputs<0,100)
            self.gradb -= dif * step_size
            self.gradw -= np.outer(dif,inputs) * step_size
        return n_dif

    def grad_param(self,batch_size):
        """
        Changes weights and biases using the precalculated grad variables from back_prop\n
        """
        self.weights+=self.gradw/batch_size
        self.bias+=self.gradb/batch_size
        self.gradw = np.zeros_like(self.weights)
        self.gradb = np.zeros_like(self.bias)