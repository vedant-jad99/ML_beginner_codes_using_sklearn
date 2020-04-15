# %matplotlib inline
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/FuelCompsumtion.csv") 
print(len(data))
# print(data)   
# print(data.head())
# print(data.describe())

data_refine =  data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# data_refine.hist()
# plt.show()

# plt.scatter(data_refine.ENGINESIZE, data_refine.CO2EMISSIONS, color = 'blue')
# plt.xlabel("Enginesize")
# plt.ylabel("Emissions")
# plt.show()

msk = np.random.rand(len(data_refine)) < 0.8
train = data_refine[msk]
test = data_refine[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

print(regr.coef_)
print(regr.intercept_)

plt.scatter(data_refine.ENGINESIZE, data_refine.CO2EMISSIONS, color = 'blue')
plt.plot(train_x,regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Enginesize")
plt.ylabel("Emissions")
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

mae = np.mean(np.absolute(test_y_hat - test_y))
mse = np.mean((test_y_hat - test_y) ** 2)
r2 = r2_score(test_y_hat, test_y)   

print("Mean Absolute Error :", mae, "\nMean Sqaured Error :",mse,"\nR2 Score :",r2)