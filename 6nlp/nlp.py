# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:54:50 2020

@author: encry973r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

import re


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # join all word as a string
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# split dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# gave 73% accuracy
# import naive bayes class
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# gave 69% accuracy
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy')
# classifier.fit(X_train, y_train)

# gave 70% accuracy
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)

# got 73% accuracy
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)

# predict y for X_test
y_pred = classifier.predict(X_test)

# the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

right = cm[0,0] + cm[1, 1]
wrong = cm[0, 1] + cm[1, 0]
p_right = (right/(right + wrong))*100

print(p_right)

















