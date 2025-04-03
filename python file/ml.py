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

"""# K-Means Clustering"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

Kmeans = KMeans(n_clusters=3,random_state=42)
Kmeans.fit(X)

centers = Kmeans.cluster_centers_
labels = Kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='red', s=200)
plt.xlabel(data.feature_names[0])  # Label x-axis
plt.ylabel(data.feature_names[1])  # Label y-axis
plt.title('K-Means Clustering on Iris Dataset')
plt.show()

"""# SVM"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.metrics import classification_report as classification_report_function

data = load_iris()

X = data.data[:,:2]
Y = data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

SVM_Model=SVC(kernel='linear',random_state=42)
SVM_Model.fit(X_train,Y_train)

Y_pred = SVM_Model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print(f'Accuracy Score is {accuracy*100:.2f}%')

def plot_decision_boundary(X, y, model):
  x_min,x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min,y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.4)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',edgecolors='k')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('SVM Decision Boundary')
  plt.show()

plot_decision_boundary(X,Y,SVM_Model)

"""# **DBSCAN Clustering**"""

from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score,adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, Y = load_iris(return_X_y=True)

X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)

labels = dbscan.fit_predict(X)

accuracy = accuracy_score(Y, labels)

print(f'Accuracy Score is {accuracy*100:.2f}%')

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',edgecolors='k',alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()

"""# **PCA**"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X= data.data
Y= data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

elf = RandomForestClassifier()
elf.fit(X_train_pca, Y_train)

Y_pred=elf.predict(X_test_pca)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy Score is {accuracy*100:.2f}%')

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, cmap='viridis', edgecolors='k', alpha=0.7)
plt.colorbar(label='Digital Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization on Iris Dataset')
plt.show()