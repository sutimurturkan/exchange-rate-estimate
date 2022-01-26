import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Read the csv file
if os.path.exists("USDtoARS.csv"): 
    df = pd.read_csv("USDtoARS.csv")
    df.drop(df.columns[[2,4, 5,6,7,8,9,10,11,12,13,14,15]], axis = 1, inplace = True)

# Prepare data
x = df['Days since 01/01/2000'].to_numpy()
y = df['Price'].to_numpy()

plt.scatter(x, y, c = 'green', edgecolors='k')
plt.grid(True)
plt.show()

df['Price'].plot.hist(bins=80, figsize=(8,5))

df['Price'].plot.density()

#cut the years
x = x[: 280]
y = y[: 280]

#Checking accuracy using r^2
r = np.corrcoef(x, y)[0][1]
r_sq = r * r
print (r_sq)


#training and testing variables
x_train = x
y_train = y
x_test = x
y_test = y
x_train= x_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#linear regression model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

#predict
y_pred = regr.predict(x_test)

#coefficients
print('Coefficients: \n', regr.coef_)

#plot outputs
plt.scatter(x_test, y_test,  color='red')
plt.plot(x_test, y_pred, color='blue')

plt.xticks(())
plt.yticks(())

plt.show()

x_predict = [[7886]]  
y_predict = regr.predict(x_predict)
print(y_predict)

