import lib_2
import numpy as np
import time
#sample digit recog to test the library
a = lib_2.network()
a.add_conv((1,28,28),(20,24,24),'ReLU')
a.add_pool((20,24,24),(20,12,12))
a.add_conv((20,12,12),(40,8,8),'ReLU')
a.add_pool((40,8,8),(40,5,5))
a.add_layer(40*5*5,100,'ReLU')
a.add_layer(100,10)
ti = open("/home/mustafa/ML/digit_recog/data/train_img","rb")
tl = open("/home/mustafa/ML/digit_recog/data/train_lab","rb")
tti = open("/home/mustafa/ML/digit_recog/data/test_img","rb")
ttl = open("/home/mustafa/ML/digit_recog/data/test_lab","rb")
ti.seek(16)
tti.seek(16)
tl.seek(8)
ttl.seek(8)
train_img = np.zeros((60000,784))
train_lab = np.zeros((60000,10))
test_img = np.zeros((10000,784))
test_lab = np.zeros((10000,10))
print("reading data",end=" ")
for i in range(60000):
    ba = int.from_bytes(tl.read(1),byteorder="big")
    train_lab[i][ba]=1
    train_img[i] = np.array(list(ti.read(784)))/255
for i in range(10000):
    ba = int.from_bytes(ttl.read(1),byteorder="big")
    test_lab[i][ba]=1
    test_img[i] = np.array(list(tti.read(784)))/255
print("done")

def print_acc(network,data):
    if len(data)==4:
        train_X,train_Y,test_X,test_Y = data
    else:
        train_x,train_Y = data
    count=0
    for i in range(train_Y.shape[0]):
        p = network.predict(train_X[i])
        p = np.where(p==np.max(p))
        if train_Y[i][p]==1:
            count+=1
    print("Training accuracy:",count/train_X.shape[0])
    count=0
    if len(data)==4:
        for i in range(test_Y.shape[0]):
            p = network.predict(test_X[i])
            p = np.where(p==np.max(p))
            if test_Y[i][p]==1:
                count+=1
        print("Test accuracy:",count/test_X.shape[0])

a.train((train_img,train_lab,test_img,test_lab),10,0.1,10,print_acc)
'''
from matplotlib import pyplot as plt
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(a.layers[0].weights[i].reshape((28,28)) + np.full((28,28),0.5),cmap='coolwarm_r')
plt.show()
'''
