import network
import numpy as np
import time
a = network.network()
a.add_layer(784)
a.add_conv(16,(1,5,5),(1,28,28))
a.add_conv(8,(16,5,5),(16,24,24))
a.add_layer(100,'ReLU')
a.add_layer(10)
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
a.train((train_img,train_lab,test_img,test_lab),10,0.00035,10)
