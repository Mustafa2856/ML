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

class pool_layer(object):

    def __init__(self,input_shape,output_shape):
        """
        A pooling layer:\n
        Param:\n
        input_shape: tuple containing-(channels,height,width) of input\n
        output_shape: tuple containing-(number of filters,height,width) of output\n
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_shape = (input_shape[0],input_shape[1]-output_shape[1]+1,input_shape[2]-output_shape[2]+1)
        size = output_shape[0]*output_shape[1]*output_shape[2]
        self.weights = np.zeros((output_shape[0],input_shape[0],self.filter_shape[1]*self.filter_shape[2]))
        for i in range(output_shape[0]):
            self.weights[i][i] = np.full(self.filter_shape[1]*self.filter_shape[2],1.0/(self.filter_shape[1]*self.filter_shape[2]))
        self.weights = self.weights.reshape((output_shape[0],-1))
        self.bias = np.zeros((output_shape[0],output_shape[1]*output_shape[2]))

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
        img = img2col(inputs,self.filter_shape)
        output = self.weights@img
        return output.reshape(-1)

    def back_prop(self,dif,inputs,outputs,step_size):
        """
        back propogates dif values
        """
        invw = self.weights.reshape((self.output_shape[0],self.filter_shape[0],self.filter_shape[1],self.filter_shape[2]))[:,:,::-1,::-1]
        invw = np.swapaxes(invw,0,1)
        k = self.filter_shape[1]
        l = self.filter_shape[2]
        g=np.pad(dif.reshape(self.output_shape),((0,0),(k-1,k-1),(l-1,l-1)))
        h=img2col(g,(self.output_shape[0],self.filter_shape[1],self.filter_shape[2]))
        ndif = invw.reshape((self.filter_shape[0],-1))@h
        return ndif
    
    def grad_param(self,batch_size):
        """
        does nothing only for compatibility
        """
        pass