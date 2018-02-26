#I have implemented SLR using Stochastic Gradient Descent Optimization on the food truck data named "simple_l.csv" file 
#present in my data directory.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import train_test_split


data=pd.read_csv('E:\Divya\learning\simple_l.csv', header=None, names=['population','profit'])


X = data.as_matrix(columns=["population"]) 
Y=data['profit'].tolist()

X_train,x_test,Y_train,y_test= train_test_split(X,Y ,test_size=0.2, random_state=0)

fitter=SGDRegressor(loss="squared_loss", penalty=None)
fitter.max_iter= np.ceil(10**5/len(Y_train))
fitter.fit(X_train,Y_train)


y_pred = fitter.predict(x_test)

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=fitter.score(x_test,y_test)






