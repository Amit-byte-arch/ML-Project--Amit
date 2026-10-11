# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:11:25 2026

@author: Amit Kumar Barnwal  
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'D:\Classwork\8th Oct 2026\7th - SVM\7th - SVM\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

"""from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth = 14, splitter="random", random state = 0) 
classifier.fit(X_train, y_train) """

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 

bias = classifier.score(X_train,y_train)
print(bias) 

variance = classifier.score(X_test,y_test)
print(variance)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred) 
cr 

