# % matplotlib inline
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/FuelCompsumtion.csv") 
print(len(data))

data_refine = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(data_refine)) < 0.8
train = data_refine[msk]
test = data_refine[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print(regr.coef_)
print(regr.intercept_)

test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("Mean Squared Error :", np.mean((test_y_hat - test_y)**2))
print("Variance Score :",regr.score(test_x, test_y))
print("R2-Score :",r2_score(test_y_hat, test_y))