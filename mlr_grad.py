import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

data= pd.read_csv('E:\data.csv', header=None, names=['Size','Bed','Profit'])

data = (data - data.mean()) / data.std()   

data.insert(0,'Intercept',1)

X=data.iloc[:,:-1].values
Y=data.iloc[:,2].values
theta= np.array([0,0,0])

m=len(X)
alpha=0.01
iteration=1000
def costfunction(X,Y,theta):
    J= np.sum((X.dot(theta)-Y)**2)/2*m
    return J

print(costfunction(X,Y,theta))

def gradient(X,Y,theta,alpha,iteration):
    cost_history=[0]*iteration
    for i in range(iteration):
        hypothesis=X.dot(theta)
        loss=hypothesis-Y
        print(loss)
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

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iteration), c, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 



# Ploting the variables as scatter plot

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(data['Size'],data['Bed'],data['Profit'], color='#ef1234')
#plt.show()


def grad_desc(X,Y,theta,rate,iters):
    cost_history=[0]*iters
    for i in range(iters):
        hypothesis= logistic_func(X,theta)
        loss= hypothesis-Y
        grad= (X.T.dot(loss))/m
        theta= theta- rate*grad
        cost=cost_func(X,Y,theta)
        cost_history[i]=cost
    return theta,cost_history

(t,c)=grad_desc(X,Y,theta,rate,iters)

print(cost_func(X,Y,t))

