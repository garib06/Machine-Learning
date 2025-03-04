# -*- coding: utf-8 -*-
"""ML.ipynb

# **Linear Regression**
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

data=load_iris()
data

X=pd.DataFrame(data.data,columns=['sepal length','sepal width','petal length','petal width'])
print(X)

Y=pd.DataFrame(data.target,columns=['Species'])
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)

y_predict=model.predict(X_test)

mse = mean_squared_error(Y_test, y_predict)
r2 = r2_score(Y_test, y_predict)*100

print("Mean Squared Error:", mse)
print("R-squared:", r2)

plt.scatter(X['sepal length'], Y['Species'], color="blue", label="Actual")

# or
plt.scatter(X['sepal width'], Y['Species'], color="blue", label="Actual")

# or
plt.scatter(X['petal length'], Y['Species'], color="blue", label="Actual")

# or
plt.scatter(X['petal width'], Y['Species'], color="blue", label="Actual")

plt.plot(X_test, y_predict, color="red", linewidth=2, label="Predicted")
plt.xlabel("X values")
plt.ylabel("Y values")

plt.legend()

plt.show()

"""# Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.metrics import classification_report as classification_report_function

X, y = data.data, (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LogisticRegression()

model.fit(X_train,Y_train)

y_predict=model.predict(X_test)
y_predict

accuracy_score = accuracy_score(Y_test,y_predict)*100
accuracy_score

score=r2_score(Y_test,y_predict)*100
score

error=mean_squared_error(Y_test,y_predict)
error

model.coef_

model.intercept_

report=classification_report_function(Y_test,y_predict)
print(report)

sigmoid=1/(1+np.exp(-y_predict))
plt.plot(sigmoid , label='sigmoid Function')

"""# **Lasso Regression**"""

from sklearn.linear_model import Lasso

data=load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

"""df['target']=data.target"""

X=pd.DataFrame(data.data,columns=['sepal length','sepal width','petal length','petal width'])

Y=pd.DataFrame(data.target,columns=['Species'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

lasso_regression=Lasso(alpha=0.001,random_state=42)

lasso_regression.fit(X_train,Y_train)

test_prediction=lasso_regression.predict(X_test)

train_prediction=lasso_regression.predict(X_train)

train_rmse=mean_squared_error(Y_train,train_prediction)
test_rmse=mean_squared_error(Y_test,test_prediction)

score=r2_score(Y_test,test_prediction)*100

print("Coefficients:", lasso_regression.coef_)
print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
print("Score is:",score)

sns.pairplot(df,hue='target')

plt.show()

"""# **RIGID REGRESSION**"""

from sklearn.linear_model import Ridge

data=load_iris()

X=pd.DataFrame(data.data,columns=data.feature_names)
Y=pd.DataFrame(data.target,columns=['Species'])

X=pd.DataFrame(data.data,columns=['sepal length','sepal width','petal length','petal width'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

Ridge_Regression=Ridge(alpha=0.01,random_state=42)
Ridge_Regression.fit(X_train,Y_train)

train_prediction=Ridge_Regression.predict(X_train)
test_prediction=Ridge_Regression.predict(X_test)

score=r2_score(Y_test,test_prediction)*100

train_rmse=mean_squared_error(Y_train,train_prediction)
test_rmse=mean_squared_error(Y_test,test_prediction)

print("Coefficent:",Ridge_Regression.coef_)
print("Score:",score)
print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

fig,axes=plt.subplots(2,2,figsize=(10,8))
axes[0,0].scatter(X['sepal length'],Y)
axes[0,0].set_xlabel('sepal length')
axes[0,0].set_ylabel('Species')
axes[0,0].legend()
axes[0,1].plot(X['sepal length'],X['sepal width'],color='red')
axes[0,1].set_xlabel('sepal length')
axes[0,1].set_ylabel('sepal width')
axes[1,0].hist(X)
axes[1,0].set_title('Distribution of sepal length')
axes[1,1].bar([1,2,3],[4,5,6])
axes[1,1].set_title('Bar Plot')

"""# Net Elastic Regression"""

from sklearn.linear_model import ElasticNet

data=load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=pd.DataFrame(data.target,columns=['Species'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=42)
elastic_net.fit(X_train,Y_train)

test_predict=elastic_net.predict(X_test)
train_prediction=elastic_net.predict(X_train)

test_rmse=mean_squared_error(Y_test,test_predict)
train_rmse=mean_squared_error(Y_train,train_prediction)
r2=r2_score(Y_test,test_predict)*100

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
print(r2)

"""# **Random Forest**"""

from sklearn.ensemble import RandomForestClassifier

data=load_iris()

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

score=r2_score(Y_test,Y_pred)*100

squared_error=mean_squared_error(Y_test,Y_pred)

print("Score:",score)
print("Squared Error:",squared_error)

confusion_matrix(Y_test,Y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(Y_test,Y_pred),cmap="Blues",annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()