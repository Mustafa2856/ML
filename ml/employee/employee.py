import lib
import pandas as pd
import numpy as np
df = pd.read_csv("Train.csv")
df = df.replace(['M','F'],[0,1])
df = df.replace(['Married','Single'],[0,1])
h = pd.get_dummies(df['Hometown'])
u = pd.get_dummies(df['Unit'])
d = pd.get_dummies(df['Decision_skill_possess'])
cb = pd.get_dummies(df['Compensation_and_Benefits'])
X = df.drop(['Employee_ID','Hometown','Unit','Decision_skill_possess','Compensation_and_Benefits','Attrition_rate'],axis=1)
for i in X:
    X[i].fillna(X[i].mean(),inplace=True)
X = pd.concat([X,h,u,d,cb],axis=1)
max_l = [X[i].max() for i in X]
X /= max_l
X = X.values
Y = df['Attrition_rate'].values
l = lib.network()
l.add_layer(44)
l.add_layer(100,'ReLU')
l.add_layer(1)
l.train((X,Y),2,0.5,10)
tdf = pd.read_csv('Test.csv')
tdf = tdf.replace(['M','F'],[0,1])
tdf = tdf.replace(['Married','Single'],[0,1])
th = pd.get_dummies(tdf['Hometown'])
tu = pd.get_dummies(tdf['Unit'])
td = pd.get_dummies(tdf['Decision_skill_possess'])
tcb = pd.get_dummies(tdf['Compensation_and_Benefits'])
tX = tdf.drop(['Employee_ID','Hometown','Unit','Decision_skill_possess','Compensation_and_Benefits'],axis=1)
for i in tX:
    tX[i].fillna(tX[i].mean(),inplace=True)
tX = pd.concat([tX,th,tu,td,tcb],axis=1)
k=10
for i in range(10):
    print(l.predict(X[k+i]),Y[k+i])