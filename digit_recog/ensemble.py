import lib
import numpy as np
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

networks = []
n_networks = 5
for i in range(n_networks):
    networks.append(lib.network())
    networks[-1].add_layer(784)
    networks[-1].add_conv(20,(1,5,5),(1,28,28))
    networks[-1].add_conv(40,(20,5,5),(20,24,24))
    networks[-1].add_layer(100,'ReLU')
    networks[-1].add_layer(10)
    
for i in range(n_networks):
    networks[i].train((train_img,train_lab,test_img,test_lab),10,0.001,8)
count=0
for i in tqdm(range(test_img.shape[0])):
    pred = np.zeros(10)
    for j in range(n_networks):
        pred += networks[j].predict(test_img[i])
    p = np.where(pred==np.max(pred))
    if test_lab[i][p]==1:
        count+=1
print('Accuracy: ',count/test_img.shape[0])