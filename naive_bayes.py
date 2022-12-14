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
#   Номер моего студ. билета равен 220386, соответственно 7 вариант:
#gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
#stalk-shape: enlarging=e,tapering=t
#stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
#stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
df = mushroom_df[['gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'class']]
#df
labels = mushroom_df['class'].to_numpy()
#labels

#tp, te = (labels == 'p'), (labels == 'e')
#df_p = pd.DataFrame(df.to_numpy()[tp], columns = ['gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring'])
#df_e = pd.DataFrame(df.to_numpy()[te], columns = ['gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring'])
#unique_p_gc, counts_e_gc = np.unique(df['gill-color'], return_counts=True)
#unique_e_gc, counts_e_gc = np.unique(df_e['gill-color'], return_counts=True)
#sns.barplot(x=unique_gc, y=counts_gc, hue=labels);
#sns.barplot(x=unique_e_gc, y=counts_e_gc, color='red');
#unique_p_ss, counts_p_ss = np.unique(df_p['stalk-shape'], return_counts=True)
#unique_e_ss, counts_e_ss = np.unique(df_e['stalk-shape'], return_counts=True)
#unique_p_sr, counts_p_sr = np.unique(df_p['stalk-root'], return_counts=True)
#unique_e_sr, counts_e_sr = np.unique(df_e['stalk-root'], return_counts=True)
#unique_p_ssar, counts_p_ssar = np.unique(df_p['stalk-surface-above-ring'], return_counts=True)
#unique_e_ssar, counts_e_ssar = np.unique(df_e['stalk-surface-above-ring'], return_counts=True)
#unique_p_ssbr, counts_p_ssbr = np.unique(df_p['stalk-surface-below-ring'], return_counts=True)
#unique_e_ssbr, counts_e_ssbr = np.unique(df_e['stalk-surface-below-ring'], return_counts=True)
sns.catplot(x = 'gill-color', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-shape', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-root', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-surface-above-ring', data = df, hue = 'class', kind = 'count')
sns.catplot(x = 'stalk-surface-below-ring', data = df, hue = 'class', kind = 'count')

#df_test = df.iloc[0:int(len(labels)*0.2)].drop(columns=['class'])
#labels_test = df['class'][0:int(len(labels)*0.2)]
#df_train = df.iloc[int(len(labels)*0.2):len(labels)].drop(columns=['class'])
#labels_train = df['class'][int(len(labels)*0.2):len(labels)]
#df_train, df_test, labels_train, labels_test = train_test_split(df, labels, test_size=0.2, random_state=1)
df_train, df_test, labels_train, labels_test = train_test_split(df.drop(columns=['class']), df['class'], test_size=0.2, random_state=1)
#df_test = df_test.drop(columns=['class'])

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
#df_enc_test = df_enc.iloc[0:int(len(labels)*0.2)].drop(columns=['class'])
#labels_enc_test = df_enc['class'][0:int(len(labels)*0.2)]
#df_enc_train = df_enc.iloc[int(len(labels)*0.2):len(labels)].drop(columns=['class'])
#labels_enc_train = df_enc['class'][int(len(labels)*0.2):len(labels)]

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
#df_enc_test = df_enc.iloc[0:int(len(labels)*0.2)].drop(columns=['class'])
#labels_enc_test = df_enc['class'][0:int(len(labels)*0.2)]
#df_enc_train = df_enc.iloc[int(len(labels)*0.2):len(labels)].drop(columns=['class'])
#labels_enc_train = df_enc['class'][int(len(labels)*0.2):len(labels)]

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
