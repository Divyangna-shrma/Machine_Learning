#I've built a Logistic Regression Model using Stochastic Gradient Descent that estimates the probability of admission based on the
#exam scores. The data is present in the "data" directory of my repo by the name "logistic.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

data=pd.read_csv("E:\Divya\learning\logistic.csv", names=["Exam1","Exam2","Admitted"])

data.insert(0,"Interscept",1)

X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values
theta=np.array([0,0,0])
#positive=data[data['Admitted'].isin([1])]
#negative=data[data['Admitted'].isin([0])]

#plt.scatter(positive['Exam1'],positive['Exam2'], c='b')
#plt.scatter(negative['Exam1'],negative['Exam2'], c='r')

rate=0.001
iters=5000
m=len(X)


def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def predict(X, theta):
  z = np.dot(X, theta)
  return sigmoid(z)

def cost_function(X, Y, theta):
    observations = len(Y)

    predictions = predict(X, theta)

    #Take the error when label=1
    class1_cost = -Y*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-Y)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost

print(cost_function(X,Y,theta))

def update_weights(X, Y, theta, rate):

    N = len(X)

    #1 - Get Predictions
    predictions = predict(X, theta)

    gradient = np.dot(X.T,  predictions - Y)

    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= rate

    #5 - Subtract from our weights to minimize cost
    theta = theta-gradient

    return theta


def train(X, Y, theta, rate, iters):
    cost_history = []

    for i in range(iters):
        theta = update_weights(X, Y, theta, rate)

        #Calculate error for auditing purposes
        cost = cost_function(X, Y, theta)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print ("iter: "+str(i) + " cost: "+str(cost))

    return theta, cost_history

t, cost_it = train(X,Y,theta,rate,iters)
print(cost_function(X,Y,t))
#print(np.sum(y_flip == predicted_y))

y_pred= predict(X,t)

def classify(y_pred):  
    probability=y_pred
    return[1 if probability>0.65 else 0 for probability in y_pred]
        

y_new = classify(y_pred)


print(rmse)


correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(y_new, Y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print ('accuracy = {0}%'.format(accuracy))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()


