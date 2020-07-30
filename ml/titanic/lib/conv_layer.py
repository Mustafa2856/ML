import numpy as np
import math

def img2col(img,filter_shape,stride=1):
    """
    py implementaion of MATLAB's img2col func
    """
    img = img.T
    r,c,ch = img.shape
    s0,s1,s3 = img.strides
    nrows = r-filter_shape[1]+1
    ncols = c-filter_shape[2]+1
    shp = ch,filter_shape[1],filter_shape[2],nrows,ncols
    strd = s3,s0,s1,s0,s1
    output = np.lib.stride_tricks.as_strided(img,shape=shp,strides=strd)
    return output.reshape(filter_shape[0]*filter_shape[1]*filter_shape[2],-1)[:,::stride]

class conv_layer(object):

    def __init__(self,n_filters,filter_shape,input_shape,pooling_shape=(1,1),stride=1,trainable=True):
        """
        A 2D convolutional layer:
        Param:
        n_filters: number of filters
        filter_shape: tuple containing (channels,filter_height,filter_weight)
        input_shape: tuple containing (channels,input_height,input_width)
        pooling shape: tuple :(pool height,width): !!!currently not working
        stride: stride
        trainable: bool value depicting if it can be trained
        """
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pooling_shape = pooling_shape
        h_ = int(1 + (input_shape[-2] - filter_shape[-2])/stride)
        w_ = int(1 + (input_shape[-1] - filter_shape[-1])/stride)
        self.topool_neuronshape = (n_filters,h_,w_)
        h__ = math.ceil(h_/pooling_shape[0])
        w__ = math.ceil(w_/pooling_shape[1])
        self.neurons = np.zeros((n_filters*h__*w__))
        self.output_shape = (n_filters,h__,w__)
        self.bias = np.zeros((n_filters,h_*w_))
        self.size = self.neurons.size
        self.weights = np.random.randn(n_filters,filter_shape[0]*filter_shape[1]*filter_shape[2])/self.size**0.5
        bias = np.random.randn(n_filters)/self.size**0.5
        for i in range(n_filters):
            self.bias[i,:] = np.full(h_*w_,bias[i])
        self.stride = stride
        self.trainable = trainable
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

    def forward_pass(self,inputs):
        """
        returns layers output for given inputs
        """
        inputs = inputs.copy()
        inputs = inputs.reshape(self.input_shape)
        img = img2col(inputs,self.filter_shape,self.stride)
        output = self.weights@img + self.bias
        self.neurons = output.reshape(-1)
        return self.neurons

    def back_prop(self,diff,prev_layer,step_size):
        """
        back propogates diff values and trains if trainable
        """
        invw = self.weights.reshape((self.n_filters,self.filter_shape[0],self.filter_shape[1],self.filter_shape[2]))[:,:,::-1,::-1]
        invw = np.swapaxes(invw,0,1)
        diff = diff.reshape(self.bias.shape)
        k = self.filter_shape[1]
        l = self.filter_shape[2]
        g=np.pad(diff.reshape(self.output_shape),((0,0),(k-1,k-1),(l-1,l-1)))
        h=img2col(g,(self.n_filters,self.filter_shape[1],self.filter_shape[2]),1)
        ndif = invw.reshape((self.filter_shape[0],-1))@h
        if self.trainable:
            self.gradw -= diff@(img2col(prev_layer.neurons.reshape(self.input_shape),self.filter_shape,1).T) * step_size
            self.gradb -= np.full(self.bias.shape,np.sum(diff,axis=1).reshape((self.n_filters,1))) * step_size
        return ndif
    
    def grad_param(self,batch_size):
        """
        Changes weights and biases using the precalculated grad variables from back_prop
        """
        self.weights+=self.gradw/batch_size
        self.bias+=self.gradb/batch_size
        self.gradw = np.zeros_like(self.weights)
        self.gradb = np.zeros_like(self.bias)