import pandas as pd
import numpy as np
import linear
df = pd.read_csv('carprices.csv')
models = pd.get_dummies(df['Car Model'])
inp = pd.concat([df,models],axis=1).drop(['Audi A5','Car Model','Sell Price($)'],axis=1)
lab = df[['Sell Price($)']].rename_axis('ID').values
img = inp.rename_axis('ID').values
m = linear.linear(4)
m.train((img,lab))
print(m.accuracy)
print(m.predict(np.array([45000,4,0,1])))
print(m.predict(np.array([86000,7,1,0])))