import sklearn.preprocessing,sklearn.model_selection,sklearn.ensemble
import pandas as pd
traindf = pd.read_csv('train.csv')
traindf = traindf.drop(['pet_id','issue_date','listing_date'],axis=1)
X = traindf.drop(['breed_category','pet_category'],axis=1)
le = sklearn.preprocessing.LabelEncoder()
X['color_type'] = le.fit_transform(X['color_type'])
for i in X:
    X[i].fillna(X[i].mean(),inplace=True)
Yb = traindf['breed_category']
Yp = traindf['pet_category']
modelb = sklearn.ensemble.AdaBoostClassifier(n_estimators=500)
modelb.fit(X,Yb)
modelp = sklearn.ensemble.AdaBoostClassifier(n_estimators=500)
modelp.fit(X,Yp)
testdf = pd.read_csv('test.csv')
xt = testdf.drop(['pet_id','issue_date','listing_date'],axis=1)
xt['color_type'] = le.transform(xt['color_type'])
for i in xt:
    xt[i].fillna(xt[i].mean(),inplace=True)
ytb = pd.DataFrame(modelb.predict(xt),columns=['breed_category'])
ytp = pd.DataFrame(modelp.predict(xt),columns=['pet_category'])
yt = pd.concat([testdf['pet_id'],ytb,ytp],axis=1)
yt.to_csv('submission.csv',index=False)