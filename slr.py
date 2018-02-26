#In this I've implemented simple linear regression to predict profits for a food truck.
#Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
#The chain already has trucks in various cities and you have data for profits and populations from the cities.
#You'd like to figure out what the expected profit of a new food truck might be given only the population of the city that 
#it would be placed in.
#Let's start by examining the data which is in a file called "ex1data1.txt" in the "data" directory of my repository above.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

data=pd.read_csv('E:\Divya\learning\simple_l.csv',header=None, names=['Size','Profit'])

X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

X_train,x_test,Y_train,y_test= train_test_split(X,Y ,test_size=0.2, random_state=0)

reg=LinearRegression()
reg.fit(X_train,Y_train)

y_pred=reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

r2=reg.score(x_test,y_test)



