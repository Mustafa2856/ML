import lib
import numpy as np
import pandas as pd
df = pd.read_csv("HR_comma_sep.csv")
df = df.sample(frac=1).reset_index(drop=True)
df = df.replace(['low','medium','high'],[0.0,0.5,1.0])
dep = pd.get_dummies(df['Department'])
df = pd.concat([df,dep],axis=1)
data = df.drop(['Department','left'],axis=1).rename_axis('ID').values
data[:,3]*=(1.0/310.0)
lab = df['left'].rename_axis('ID').values
train_img = data[:13000]
train_lab = lab[:13000]
test_img = data[13000:]
test_lab = lab[13000:]
nn = lib.network()
nn.add_layer(18)
nn.add_layer(1)
nn.train((train_img,train_lab,test_img,test_lab),5,0.1,100)
print(nn.check_accuracy(test_img,test_lab))