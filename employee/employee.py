import lib
import pandas as pd
import numpy as np
import warnings
import math
warnings.simplefilter("ignore")
df = pd.read_csv("/home/mustafa/ML/employee/Train.csv")
train_data = np.zeros((18,7000))
attribute_values = ["VAR1",
                    "VAR2","VAR3","VAR4","VAR5","VAR6","VAR7"]
for i in range(attribute_values.__len__()):
    if i==1 or i==3:
        continue
    for j in range(7000):
        if df[attribute_values[i]].isnull()[j]:
            df[attribute_values[i]][j]=0
    train_data[i] = np.array(df[attribute_values[i]])
for i in range(7000):
    if df[attribute_values[1]][i]=='F':
        train_data[1][i] = 0
    else:
        train_data[1][i] = 1
    if df[attribute_values[3]][i]=='Married':
        train_data[3][i] = 0
    else:
        train_data[3][i] = 1
train_data = np.transpose(train_data)
train_lab = np.array(df['Attrition_rate'])
a = lib.network()
a.add_layer(18)
a.add_layer(120)
a.add_layer(1)
a.train((train_data,train_lab),1,0.001,500)
for i in range(70):
    print(a.predict(train_data[i])[0],train_lab[i])