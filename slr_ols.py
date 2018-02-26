#I have implemented SLR using Ordinary Least Square Optimization on the food truck data named "simple_l.csv" file present in my data directory.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

data=pd.read_csv('E:\Divya\learning\simple_l.csv')

X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

X_train,x_test,Y_train,y_test= train_test_split(X,Y, test_size=0.2, random_state=0)

plt.scatter(X,Y)

x_mean= np.mean(X_train)
y_mean= np.mean(Y_train)

m=len(X_train)

num=0
deno=0

for i in range(m):
    num+= (X_train[i]-x_mean)*(Y_train[i]-y_mean)
    deno+= (X_train[i]-x_mean)**2

b1= num/deno
b0= y_mean-(b1*x_mean)

print(b1,b0)

max_x= np.max(X_train) + 100
min_x=np.min(X_train)-100

x=np.linspace(min_x,max_x,1000)
y=b0+b1*x

plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X_train, Y_train, c='#ef5423', label='Scatter Plot')

plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

rmse=0
for i in range(m):
    y_pred=b0+b1*x_test[i]
    rmse+= (y_test[i]- y_pred)**2
rmse=np.sqrt(rmse/m)

