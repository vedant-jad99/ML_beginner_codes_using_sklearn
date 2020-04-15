import numpy as np 
import pandas as pd 
import pylab as pl 
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

cell_df = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/cell_samples.csv")
# print(cell_df.head())

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

clf = svm.SVC(kernel= 'rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

print(classification_report(y_test, yhat))