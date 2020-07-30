import numpy as np
import lib
import pandas as pd
df = pd.read_csv('titanic.csv')
df = df.drop(['Name'],axis=1)
df = df.replace(['male','female'],[0,1])
df = df.sample(frac=1).reset_index(drop=True)
df['Age'].fillna(30,inplace=True)
lab = df[['Survived']].values
dat = df.drop(['Survived'],axis=1).values
dat[:,5] = dat[:,5]/500
dat[:,2] = dat[:,2]/80
x,y,xt,yt = dat,lab,dat,lab
nn = []
n = 5
for i in range(n):
    nn.append(lib.network())
    nn[-1].add_layer(6)
    nn[-1].add_layer(10,'ReLU')
    nn[-1].add_layer(1)
    nn[-1].train((x,y,xt,yt),1,0.001,50)
    print('network ',i,' trained, accuracy :',nn[-1].check_accuracy(xt,yt))

count=0
for i in range(yt.size):
    predicted=0
    for j in range(n):
        predicted += nn[j].predict(xt[i])
    predicted/=n
    if predicted>=0.5 and yt[i]==1:
        count+=1
    elif predicted<0.5 and yt[i]==0:
        count+=1

print("Accuracy : ",count/yt.size)