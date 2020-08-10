import lib_2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#reading data
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
print("done")

dis = lib_2.network()
dis.add_layer(784,100)
dis.add_layer(100,10)

gen = lib_2.network()
gen.add_layer(10,100)
gen.add_layer(100,784)

def dprint_acc(network,data):
	train_X,train_Y = data
	count=0
	for i in range(train_Y.shape[0]):
		p = network.predict(train_X[i])
		p = np.where(np.max(p)==p)
		if train_Y[i][p]==1:
			count+=1
	print("Training accuracy:",count/train_X.shape[0])
	
def gprint_acc(size):
	count=0
	for i in range(size):
		k = np.random.randint(0,10)
		x = np.zeros(10)
		x[k]=1
		p = gen.predict(x)
		p = dis.predict(p)
		p = np.where(np.max(p)==p)
		if x[p]==1:
			count+=1
	print('Training accuracy:',count/size)

dis.train((train_img,train_lab),10,1,2,dprint_acc)

for _ in range(1):
	print('training generator')
	dis.layers[0].make_non_trainable()
	dis.layers[1].make_non_trainable()
	gep,bat,bat_s = 1,10000,10
	for count in range(gep):
		for i in tqdm(range(bat)):
			for j in range(bat_s):
				k = np.random.randint(0,10)
				x = np.zeros(10)
				x[k]=1
				gn = gen.predict(x,True)
				dn = dis.predict(gn[-1],True)
				dif = dn[-1] - x
				for k in range(len(dis.layers)):
				    dif = dis.layers[-(1+k)].back_prop(dif,dn[-(2+k)],dn[-(1+k)],0.1)
				for k in range(len(gen.layers)):
				    dif = gen.layers[-(1+k)].back_prop(dif,gn[-(2+k)],gn[-(1+k)],0.1)
			for l in gen.layers:
				l.grad_param(bat_s)
		print('epoch ',count+1,' completed')
		gprint_acc(bat)

from matplotlib import pyplot as plt
for i in range(9):
	plt.subplot(3,3,i+1)
	x = np.zeros(10)
	x[i] = 1
	plt.imshow(gen.predict(x).reshape((28,28)))
plt.show()
