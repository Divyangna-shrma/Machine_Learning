#I've implemented MLR using Stochastic Gradient Descent Optimization Technique on the same "data.csv" file to predict price of house.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

data=pd.read_csv('E:\data.csv', header=None, names=['Size','bed','profit'])


X = data.as_matrix(columns=["Size","bed"]) 
Y=data['profit'].tolist()

X_train,x_test,Y_train,y_test= train_test_split(X,Y ,test_size=0.2, random_state=0)

fitter=SGDRegressor(loss="squared_loss", penalty=None)
fitter.max_iter= np.ceil(10**5/len(Y_train))
fitter.fit(X_train,Y_train)


y_pred = fitter.predict(x_test)

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_pred,y_test)






