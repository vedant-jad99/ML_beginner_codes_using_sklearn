# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/FuelCompsumtion.csv") 
print(len(data))

data_refine = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(data_refine)) < 0.8
train = data_refine[msk]
test = data_refine[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 3)
train_x_poly = poly.fit_transform(train_x)

regr = linear_model.LinearRegression()
regr.fit(train_x_poly, train_y)
plt.scatter(data_refine.ENGINESIZE, data_refine.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
xx = np.arange(0.0, 10.0, 1.0)
yy = regr.intercept_[0] + regr.coef_[0][1] * xx + regr.coef_[0][2] * np.power(xx, 2) + regr.coef_[0][3] * np.power(xx, 3)
plt.plot(xx, yy, '-r')
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_hat = regr.predict(test_x_poly)
print("Mean Squared Error :",np.mean((test_y_hat - test_y) ** 2))
print("R2-score:",r2_score(test_y_hat , test_y) )