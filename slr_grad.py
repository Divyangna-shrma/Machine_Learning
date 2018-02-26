#I have implemented SLR using Gradient Descent Optimization using the food truck data named "simple_l.csv" file present in my data directory.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('E:\Divya\learning\simple_l.csv')

data.insert(0,'Intercept',1)

X=data.iloc[:,:-1].values
Y=data.iloc[:,2].values
theta= np.array([0,0])

m=len(X)
alpha=0.01
iteration=8000
def costfunction(X,Y,theta):
    J= np.sum((X.dot(theta)-Y)**2)/2*m
    return J

print(costfunction(X,Y,theta))

def gradient(X,Y,theta,alpha,iteration):
    cost_history=[0]*iteration
    for i in range(iteration):
        hypothesis=X.dot(theta)
        loss=hypothesis-Y
        gradient=X.T.dot(loss)/m
        theta=theta-alpha*gradient
        cost=costfunction(X,Y,theta)
        cost_history[i]=cost
    return theta,cost_history

(t,c)=gradient(X,Y,theta,alpha,iteration)
print(t)

print(costfunction(X,Y,t))

y_pred=X.dot(t)

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(Y,y_pred))

print(rmse)

from sklearn.metrics import r2_score

print(r2_score(Y, y_pred))

print(c)









