# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split

"""1."""

from google.colab import drive
drive.mount('/content/drive')
import csv
import re
mushroom_df = pd.read_csv('/content/drive/MyDrive/agaricus-lepiota.data', names=['class', 'cap-shape', 'cap-surface', 'cap-color',
'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'],
header=None)

df = mushroom_df[['gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'class']]
#df
labels = mushroom_df['class'].to_numpy()
#labels

sns.catplot(x = 'gill-color', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-shape', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-root', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-surface-above-ring', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-surface-below-ring', data = df, hue = 'class', kind = 'count')

df_train, df_test, labels_train, labels_test = train_test_split(df.drop(columns=['class']), df['class'], test_size=0.2, random_state=1)

"""2."""

def frequency_column(column, j, df_train, labels_train, alpha=1.0): #насколько я знаю, по умолчанию в BernoulliNB тоже по умолчанию alpha=1
  y = labels_train
  prob  = {}
  x = df_train[column] #выбираем одну колонку
  x = x[y == j] #выбираем один класс
  l = len(df_train[column].unique())
  for i in df_train[column].unique():
    prob[i] = (x[x == i].count()+alpha) / (x.count() + alpha*l) #доля элементов со значением i признака column среди класса j
  return prob
def fit(column, df_train, labels_train):
  y = labels_train
  fr_p, fr_e = frequency_column(column,'p',df_train, labels_train), frequency_column(column,'e',df_train, labels_train)
  p_p = y[y == 'p'].count() / y.count()
  p_e = y[y == 'e'].count() / y.count()
  return fr_p, fr_e, p_p, p_e
def predict(X,column, df_train, labels_train):
  y_pred = []
  fr_p, fr_e, p_p, p_e = fit(column, df_train, labels_train)
  for k in X[column]:
    if p_e * fr_e[k] < p_p * fr_p[k]:
      y_pred.append('p')
    else:
      y_pred.append('e')
  return np.array(y_pred)

accur = pd.DataFrame({})
for column in df_test.columns:
  p_test = (predict(df_test, column, df_train, labels_train) == labels_test)
  p_train = (predict(df_train, column, df_train, labels_train) == labels_train)
  accur = accur.append(pd.DataFrame({'accur_train': [p_train.tolist().count(True)/len(p_train)], 'accur_test': [p_test.tolist().count(True)/len(p_test)]}), ignore_index = True)
accur.index = df_test.columns
accur

"""3."""

#from sklearn.preprocessing import LabelEncoder
#df_enc = df.copy() 
#for i in ['gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'class']:
#    df_enc[i]=LabelEncoder().fit_transform(df[i])
#df_enc = pd.get_dummies(data=df_enc, prefix=['stalk-surface-below-ring'])
#df_enc_train, df_enc_test, labels_enc_train, labels_enc_test = train_test_split(df_enc.drop(columns=['class']), df_enc['class'], test_size=0.2, random_state=1)


#from sklearn.naive_bayes import BernoulliNB
#model = BernoulliNB()
#model = model.fit(df_enc_train, labels_enc_train)
#accur_test = (model.predict(df_enc_test) == labels_enc_test).tolist().count(True)/len(df_enc_test)
#accur_test

from sklearn.preprocessing import LabelEncoder
df_enc = df.copy() 
for i in df.columns:
    df_enc[i]=LabelEncoder().fit_transform(df[i])
df_enc_train, df_enc_test, labels_enc_train, labels_enc_test = train_test_split(df_enc.drop(columns=['class']), df_enc['class'], test_size=0.2, random_state=1)


from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB()
model = model.fit(df_enc_train, labels_enc_train)
accur_test = (model.predict(df_enc_test) == labels_enc_test).tolist().count(True)/len(df_enc_test)
accur_test

"""4."""

def frequency(j, df_train, labels_train, alpha=1):
  y = labels_train
  prob  = {}
  for column in df_train.columns:
    prob[column] = frequency_column(column, j, df_train, labels_train, alpha=alpha)
  return prob
def fit(df_train, labels_train, alpha=1):
  y = labels_train
  fr_p, fr_e = frequency('p',df_train, labels_train, alpha=alpha), frequency('e',df_train, labels_train, alpha=alpha)
  p_p = y[y == 'p'].count() / y.count()
  p_e = y[y == 'e'].count() / y.count()
  return fr_p, fr_e, p_p, p_e
def predict(X, df_train, labels_train, alpha=1):
  y_pred = []
  fr_p, fr_e, p_p, p_e = fit(df_train, labels_train, alpha=alpha)
  for i in range(len(X)):
    pr_p, pr_e = p_p, p_e
    for column in X.columns:
      pr_p *= fr_p[column][X[i:i+1][column].values[0]]
      pr_e *= fr_e[column][X[i:i+1][column].values[0]]
    if p_e * pr_e < p_p * pr_p:
      y_pred.append('p')
    else:
        y_pred.append('e')
  return np.array(y_pred)

#df_train, df_test, labels_train, labels_test = train_test_split(df.drop(columns=['class']), df['class'], test_size=0.2, random_state=1)
accur = (predict(df_test, df_train, labels_train) == labels_test).tolist().count(True)/len(df_test)
accur
