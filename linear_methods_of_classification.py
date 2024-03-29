# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

"""1."""

from sklearn import datasets
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
labels = iris['target']
#iris_df

corr_iris = iris_df.corr()
corr_iris

unique, counts = np.unique(labels, return_counts=True)
border1, border2, border3 = counts[0], counts[0]+counts[1], len(labels)
class0_df = iris_df.iloc[0:border1]
class1_df = iris_df.iloc[border1:border2]
class2_df = iris_df.iloc[border2:border3]

corr_class0 = class0_df.corr()
#corr_class0

corr_class1 = class1_df.corr()
#corr_class1

corr_class2 = class2_df.corr()
#corr_class2

iris_df['labels'] = labels
sns.pairplot(data=iris_df, hue='labels')

"""2."""

# буду использовать переменные sepal length и sepal width
X = iris_df[['sepal length (cm)','petal width (cm)']].to_numpy()
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.5
mesh_size = 500
x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, mesh_size), np.linspace(y_min, y_max, mesh_size))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm
lda = LDA().fit(X,labels)
qda = QDA().fit(X,labels)
lr = LR().fit(X,labels)
svm_l = svm.SVC(kernel='linear').fit(X,labels)
svm_q = svm.SVC(kernel='poly', degree=2).fit(X,labels)

for  (model, name) in  [(lda, 'linear discriminant'), (qda, 'quadratic discriminant'), (lr,'logistic regression'), (svm_l,'svm (linear kernel)'),  (svm_q, 'svm (quadratic kernel)')]:
    model.fit(X,labels)
    Z = model.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    Z = Z.reshape(mesh_size, mesh_size)
    plt.figure(figsize=(3,3), dpi = 200)
    plt.contourf(x_mesh, y_mesh, Z, cmap ='Reds')
    plt.scatter(X[:, 0], X[:, 1], c = labels, s = 1)
    plt.xlabel('sepal lenght')
    plt.ylabel('petal width')
    plt.title(f'{name}')
    plt.show()

"""3."""

X = iris_df.drop(columns='labels').to_numpy()
lda.fit(X,labels)
t = (labels == lda.predict(X))
t0, t1, t2 = t[labels == 0], t[labels == 1], t[labels == 2]
for pair in itertools.combinations(list(range(0,4)), 2): 
    Xpair = X[:,[pair[0],pair[1]]]
    X0, X1, X2 = Xpair[labels == 0], Xpair[labels == 1], Xpair[labels == 2]
    plt.figure(figsize=(3,3), dpi = 200)
    # class0: dots
    X0t, X0f = X0[t0], X0[~t0]
    plt.plot(X0t[:, 0], X0t[:, 1], '.', color='red')
    plt.plot(X0f[:, 0], X0f[:, 1], '*', color='red')
    # class1: dots
    X1t, X1f = X1[t1], X1[~t1]
    plt.plot(X1t[:, 0], X1t[:, 1], '.', color='blue')
    plt.plot(X1f[:, 0], X1f[:, 1], '*', color='blue')
    # class2: dots
    X2t, X2f = X2[t2], X2[~t2]
    plt.plot(X2t[:, 0], X2t[:, 1], '.', color='green')
    plt.plot(X2f[:, 0], X2f[:, 1], '*', color='green')
    plt.xlabel(f'{iris_df.columns[pair[0]]}')
    plt.ylabel(f'{iris_df.columns[pair[1]]}')
    plt.show()

"""4."""

X = iris_df[['sepal length (cm)','petal width (cm)']]
y = labels
cov0, cov1, cov2 = X.iloc[0:border1].cov().to_numpy(), X.iloc[border1:border2].cov().to_numpy(), X.iloc[border2:border3].cov().to_numpy()
mean0, mean1, mean2 = X.iloc[0:border1].mean().to_numpy(), X.iloc[border1:border2].mean().to_numpy(), X.iloc[border2:border3].mean().to_numpy()
def predict(x):
    a = 0.5 * (np.dot((x-mean1), np.dot(np.linalg.inv(cov1), (x-mean1))) - np.dot((x-mean0), np.dot(np.linalg.inv(cov0), (x-mean0))))
    + np.log(np.sqrt(abs(np.linalg.det(cov1))) / np.sqrt(abs(np.linalg.det(cov0))))
    + np.log(counts[0]/counts[1])
    b = 0.5 * (np.dot((x-mean2), np.dot(np.linalg.inv(cov2), (x-mean2))) - np.dot((x-mean1), np.dot(np.linalg.inv(cov1), (x-mean1))))
    + np.log(np.sqrt(abs(np.linalg.det(cov2))) / np.sqrt(abs(np.linalg.det(cov1))))
    + np.log(counts[1]/counts[2])
    c = 0.5 * (np.dot((x-mean2), np.dot(np.linalg.inv(cov2), (x-mean2))) - np.dot((x-mean0), np.dot(np.linalg.inv(cov0), (x-mean0))))
    + np.log(np.sqrt(abs(np.linalg.det(cov2))) / np.sqrt(abs(np.linalg.det(cov0))))
    + np.log(counts[0]/counts[2])
    if a > 0:
      if c > 0:
        return  0
      else:
        return 2
    if b > 0:
      return 1
    return 2

Z = []
for i in np.c_[x_mesh.ravel(), y_mesh.ravel()]:
    Z.append(predict(i))
Z = np.array(Z).reshape(mesh_size, mesh_size)
plt.figure(figsize=(3,3), dpi = 200)
plt.contourf(x_mesh, y_mesh, Z, cmap ='Reds')
plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 1], c = y,s = 1)
plt.xlabel('petal width')
plt.ylabel('sepal length')
