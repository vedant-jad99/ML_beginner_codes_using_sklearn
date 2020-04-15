import numpy as np
import pandas as pd 
import scipy.optimize as opt
import pylab as pl 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score

data = pd.read_csv('/home/thedarkcoder/Desktop/ML/Coursera/datasets/ChurnData.csv')
data = data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
data['churn'] = data['churn'].astype(int)

X = np.asarray(data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(data['churn'])

X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
acc = jaccard_similarity_score(y_test, yhat)
print(acc)