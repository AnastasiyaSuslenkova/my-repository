# -*- coding: utf-8 -*-
"""
1.
"""

from google.colab import drive
drive.mount('/content/drive')
import csv
import re
import pandas as pd
wine_white = pd.read_csv('/content/drive/MyDrive/winequality-white.csv', names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
'alcohol', 'quality'], sep=';',
header=None)
wine_white = wine_white.drop(labels = [0], axis = 0)
wine_red = pd.read_csv('/content/drive/MyDrive/winequality-red.csv', names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
'alcohol', 'quality'], sep=';',
header=None)
wine_red = wine_red.drop(labels = [0], axis = 0)
df = pd.concat([wine_white, wine_red], ignore_index=True)
labels = ['w' for i in range(len(wine_white))] + ['r' for i in range(len(wine_red))]
for column in df.columns:
  df[column] = pd.to_numeric(df[column])
df

df.describe()

df.info()

"""2."""

from sklearn.model_selection import train_test_split
df_train, df_test, labels_train, labels_test = train_test_split(df, labels, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
d_t = DecisionTreeClassifier(max_depth=3)
d_t = d_t.fit(df_train, labels_train)
fig = plt.figure(figsize=(50, 25))
tree.plot_tree(d_t, feature_names=df.columns)
print(f'f1_weighted score: {f1_score(labels_test, d_t.predict(df_test),average= "weighted")}')
print(f'f1_micro score: {f1_score(labels_test, d_t.predict(df_test),average= "micro")}')
plt.show()

from sklearn.tree import export_text
print(export_text(d_t, feature_names= [i for i in df_test.columns]))

"""3."""

from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
f1_weighted_score_train = []
f1_weighted_score_test = []
for n in range(2,50):
    d_t = DecisionTreeClassifier(max_leaf_nodes=n)
    d_t = d_t.fit(df_train, labels_train)
    #print(f'max_leaf_nodes: {n}')
    f1_weighted_score_train.append(f1_score(labels_train, d_t.predict(df_train),average= "weighted"))
    f1_weighted_score_test.append(f1_score(labels_test, d_t.predict(df_test),average= "weighted"))
    #print(f'f1_weighted score train: {f1_score(labels_train, d_t.predict(df_train),average= "weighted")}')
    #print(f'f1_weighted score test: {f1_score(labels_test, d_t.predict(df_test),average= "weighted")}')
    #print(' ')
plt.figure(figsize=(3,3), dpi = 200)
plt.plot(range(48), f1_weighted_score_train, color='blue')
plt.plot(range(48), f1_weighted_score_test, color='red')
plt.show()

from sklearn.model_selection import cross_val_score
import numpy as np
d_t_g = DecisionTreeClassifier(max_depth=3, criterion='gini')
d_t_g = d_t_g.fit(df_train, labels_train)
#print('gini')
#print(f'f1_weighted score: {f1_score(labels_test, d_t_g.predict(df_test),average= "weighted")}')
#print(' ')
d_t_e = DecisionTreeClassifier(max_depth=3, criterion='entropy')
d_t_e = d_t_e.fit(df_train, labels_train)
#print('entropy')
#print(f'f1_weighted score: {f1_score(labels_test, d_t_e.predict(df_test),average= "weighted")}')
#print(' ')
f1_weighted_score_gini = []
f1_weighted_score_entropy = []
for n in range(2,10):
  d_t_g = DecisionTreeClassifier(max_depth=n, criterion='gini')
  d_t_g = d_t_g.fit(df_train, labels_train)
  d_t_e = DecisionTreeClassifier(max_depth=n, criterion='entropy')
  d_t_e = d_t_e.fit(df_train, labels_train)
  f1_weighted_score_gini.append(f1_score(labels_test, d_t_g.predict(df_test), average= "weighted"))
  f1_weighted_score_entropy.append(f1_score(labels_test, d_t_e.predict(df_test), average= "weighted"))
plt.figure(figsize=(3,3), dpi = 200)
plt.plot(range(8), f1_weighted_score_gini, color='blue')
plt.plot(range(8), f1_weighted_score_entropy, color='red')
plt.show()

"""4."""

import sklearn
from xgboost import XGBClassifier
df_train.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
df_test.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
le = sklearn.preprocessing.LabelEncoder()
y_train = le.fit_transform(labels_train)
y_test = le.fit_transform(labels_test)
g_b = XGBClassifier()
g_b.fit(df_train, y_train)
print(f'f1_weighted score test: {f1_score(y_test, g_b.predict(df_test),average="weighted")}')

from xgboost import plot_tree
for i in range(3):
    plot_tree(g_b, num_trees=i)
plt.show()

feature_imp = g_b.feature_importances_
for feature, imp in zip(df_train.columns, feature_imp):
  print(feature+':', imp)

f1_weighted_score_train = []
f1_weighted_score_test = []
for n_est in range(1, 10):
    g_b = XGBClassifier(n_estimators = n_est)
    g_b.fit(df_train, y_train)
    #print(f'number of trees: {n_est}')
    #print(f'f1_weighted score train: {f1_score(y_train, g_b.predict(df_train),average= "weighted")}')
    #print(f'f1_weighted score test: {f1_score(y_test, g_b.predict(df_test),average="weighted")}')
    #print(' ')
    f1_weighted_score_train.append(f1_score(y_train, g_b.predict(df_train), average= "weighted"))
    f1_weighted_score_test.append(f1_score(y_test, g_b.predict(df_test), average= "weighted"))
plt.figure(figsize=(3,3), dpi = 200)
plt.plot(range(9), f1_weighted_score_train, color='blue')
plt.plot(range(9), f1_weighted_score_test, color='red')
plt.show()

"""5."""

from sklearn.ensemble import RandomForestClassifier
r_f = RandomForestClassifier()
r_f.fit(df_train, y_train);
print(f'f_1 score test: {f1_score(y_test, r_f.predict(df_test),average="weighted")}');

feature_imp = r_f.feature_importances_
for feature, imp in zip(df_train.columns, feature_imp):
  print(feature+':', imp)

f1_weighted_score_train = []
f1_weighted_score_test = []
for n_est in range(1, 10):
    r_f = RandomForestClassifier(n_estimators = n_est)
    r_f.fit(df_train, y_train)
    #print(f'number of trees: {n_est}')
    #print(f'f1_weighted score train: {f1_score(y_train, r_f.predict(df_train),average= "weighted")}')
    #print(f'f1_weighted score test: {f1_score(y_test, r_f.predict(df_test),average="weighted")}')
    #print(' ')
    f1_weighted_score_train.append(f1_score(y_train, r_f.predict(df_train), average= "weighted"))
    f1_weighted_score_test.append(f1_score(y_test, r_f.predict(df_test), average= "weighted"))
plt.figure(figsize=(3,3), dpi = 200)
plt.plot(range(9),f1_weighted_score_train,  color='blue')
plt.plot(range(9),f1_weighted_score_test,  color='red')
plt.show()
