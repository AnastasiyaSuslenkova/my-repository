# -*- coding: utf-8 -*-

import pandas as pd
import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np

class ClassifiedKNNData:

  def __init__(self, k):
    self.k = k

  def fit(self, df_train):
    self.labels_train = df_train.to_numpy()[:, 2]
    self.df_train = df_train.drop(columns = ['c'],axis = 1)
    self.numberOfClasses = len(np.unique(self.labels_train))

  def predict(self, df):
    n = len(df)
    labels = np.zeros(n)
    for i in range(n):
      testDist = np.sqrt(np.sum(np.square(df.iloc[i].to_numpy() - self.df_train.to_numpy()), axis=1))
      nearest_k_index = sorted(range(len(self.df_train)), key = lambda sub: testDist[sub])[:k]
      stat = np.zeros(self.numberOfClasses)
      for ki in range(self.k):
        stat[int(self.labels_train[nearest_k_index[ki]])]+=1
      labels[i] = np.argmax(stat)
    return labels

  def graph_neigbours (self, point):
    testDist = np.sqrt(np.sum(np.square(point - self.df_train.to_numpy()), axis=1))
    nearest_indexes = sorted(range(len(self.df_train)), key = lambda sub: testDist[sub])[:len(self.df_train)]
    x_near = []
    y_near = []
    for ki in range(k):
      x_near.append(self.df_train.iloc[nearest_indexes[ki]]['x'])
      y_near.append(self.df_train.iloc[nearest_indexes[ki]]['y'])
    x_notnear = []
    y_notnear = []
    for ki in range(k, len(self.df_train)):
      x_notnear.append(self.df_train.iloc[nearest_indexes[ki]]['x'])
      y_notnear.append(self.df_train.iloc[nearest_indexes[ki]]['y'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([point[0]], [point[1]], color='r', linewidth=1000)
    ax.scatter(x_near, y_near, color='b', linewidth=1000)
    ax.scatter(x_notnear, y_notnear, color='g', linewidth=1000)
    plt.show()

#генерирую данные
k = 5
n = 100           #число точек для кластеризации
n_train = 20      #размер обучающей выборки
E = (1,2)
df = pd.DataFrame({'x': [random.normalvariate(random.choice(E), 0.25) for i in range(n)], 'y': [random.normalvariate(random.choice(E), 0.25) for i in range(n)]})
df_train = ({'x':[], 'y':[], 'c':[]})
for i in range(n_train):
  E1 = random.choice(E)
  E2 = random.choice(E)
  df_train['x'].append(random.normalvariate(E1, 0.25))
  df_train['y'].append(random.normalvariate(E2, 0.25))
  if E1 == 1: #
    if E2 == 1:
      df_train['c'].append(0) #нулевой класс: E1=E2=1
    else:
      df_train['c'].append(1) #первый класс: E1=1, E2=2
  elif E2 == 1:
    df_train['c'].append(2) #второй класс: E1=2, E2=1
  else:
    df_train['c'].append(3) #третий класс: E1=E2=2
df_train = pd.DataFrame(df_train)

df_knn = ClassifiedKNNData(k)
df_knn.fit(df_train)
labels = df_knn.predict(df)
x, y = float(input("Введите первую координату точки, для которой будут показаны ближайшие соседи: ")), float(input("вторую: "))
df_knn.graph_neigbours(np.array([x, y]))
print('Сама точка показана красным, ее k ближайших соседей - синим, остальные точки в обучающей выборке - зеленым.')
