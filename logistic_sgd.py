import numpy as np
import pandas as pd
import matplotlib.pyplot as plyplt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import sklearn

data=pd.read_csv("E:\Divya\learning\logistic.csv", names=["Exam1","Exam2","Admitted"])

data.insert(0,'Interscept',1)

X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values

normalized_range = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

Y.shape =  (100,) #scikit expects this
X = normalized_range.fit_transform(X)

X_train,x_test,Y_train,y_test= train_test_split(X,Y, test_size=0.3, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)

scikit_score = log_reg.score(x_test,y_test)



