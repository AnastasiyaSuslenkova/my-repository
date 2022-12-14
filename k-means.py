# -*- coding: utf-8 -*-

import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ClassifiedKmeansData:

  def __init__ (self, k, df):
    self.k = k
    self.df = df

  def random_centers(self):
    centers = np.zeros((self.k, 2))
    for j in range(self.k):
      i = np.random.choice(range(n), replace=False)
      centers[j] = self.df.to_numpy()[i]
    return centers

  def data_distribution(self, centers):
    labels = np.zeros(len(self.df))
    for i in range(len(self.df)):
      min_distance = float('inf')
      distance = np.sqrt(np.sum(np.square(self.df.iloc[i].to_numpy() - centers), axis=1))
      labels[i] = np.argmin(distance)
    return labels

  def centers_update_and_graph(self, labels):
    centers = np.zeros((self.k, 2))
    fig = plt.figure()
    centers_x = []
    centers_y = []
    for i in range(self.k):
      arrx = []
      arry = []
      for j in range(len(self.df)):
        if labels[j] == i:
          arrx.append(self.df.iloc[j]['x'])
          arry.append(self.df.iloc[j]['y'])
      centers[i] = [np.mean(arrx), np.mean(arry)]
      plt.scatter(arrx, arry)
      centers_x.append(centers[i][0])
      centers_y.append(centers[i][1])
    plt.scatter(centers_x, centers_y)
    plt.show()
    return centers

  def predict (self):
    prev_centers = self.random_centers()
    labels = self.data_distribution(prev_centers)
    while True:
     centers = self.centers_update_and_graph(labels)
     labels = self.data_distribution(centers)
     if np.array_equal(prev_centers, centers):
       self.labels = labels
       break
     prev_centers = centers

#генерирую данные
dim = 2
k = 4
max_value = 2
n = 100 
E = (1,2)
df = pd.DataFrame({'x': [random.normalvariate(random.choice(E), 0.25) for i in range(n)], 'y': [random.normalvariate(random.choice(E), 0.25) for i in range(n)]})

df_kmeans = ClassifiedKmeansData(k, df)
df_kmeans.predict()

