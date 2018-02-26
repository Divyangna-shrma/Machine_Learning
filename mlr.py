#I've implemented Multivariate Linear regression to predict the price that a house will sell for.
#The difference this time around is we have more than one dependent variable. 
#We're given both the size of the house in square feet, and the number of bedrooms in the house. 
#The data is present by tha name of "data.csv" in my "data directory.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

data= pd.read_csv('E:\data.csv',header=None, names=['Size','Bed','Profit'])
X= data.iloc[:,:-1].values
Y= data.iloc[:,2].values

X_train,x_test,Y_train,y_test=train_test_split(X,Y, test_size=0.2, random_state=0)

reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred=reg.predict(x_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))

r2=reg.score(x_test,y_test)

print(rmse)
print(r2)

        







