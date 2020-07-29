import sklearn.datasets
import numpy as np
import pandas as pd
import lib
ds = sklearn.datasets.load_iris()
data = ds['data']
label = ds['target']
df = pd.concat([pd.DataFrame(data=data),pd.DataFrame(data=label,columns=['out',])],axis=1)
df = df.sample(frac=1).reset_index(drop=True)
data = df.drop(['out',],axis=1).rename_axis('ID').values
lab = pd.get_dummies(df['out']).rename_axis('ID').values
train_img = data[:120]
train_lab = lab[:120]
test_img = data[120:]
test_lab = lab[120:]
nn = lib.network()
nn.add_layer(4)
nn.add_layer(8,'ReLU')
nn.add_layer(3)
nn.train((train_img,train_lab,test_img,test_lab),1,0.01,100)