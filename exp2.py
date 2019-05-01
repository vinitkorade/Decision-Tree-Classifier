# -*- coding: utf-8 -*-
# conda install python-graphviz
"""
Created on Thu Mar 28 10:54:21 2019

@author: VINIT KORADE
"""

import pandas as pd
import graphviz

data = pd.read_csv("D:\\LP3\\ML\\Exp2\\data.csv")

from sklearn.preprocessing import LabelEncoder
lble = LabelEncoder()
data.iloc[:,1] = lble.fit_transform(data.iloc[:,1])
data.iloc[:,2] = lble.fit_transform(data.iloc[:,2])
data.iloc[:,3] = lble.fit_transform(data.iloc[:,3])
data.iloc[:,4] = lble.fit_transform(data.iloc[:,4])
data.iloc[:,5] = lble.fit_transform(data.iloc[:,5])

x = data.iloc[:,1:5].values
y = data.iloc[:,5].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.33, random_state= 1)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train, Y_train)
Y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score: ", accuracy_score(Y_test, Y_pred))

from sklearn import tree as tr
dot_data = tr.export_graphviz(dtc, out_file=None, feature_names=['age','income','gender','maretial status'], class_names=['yes','no'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph