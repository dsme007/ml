# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:07:40 2020

@author: encry973r
"""


import numpy as np
import pandas as pd
import re

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0, 1000):
    rev = data['Review'][i]
    # leave alphabets and white spaces
    rev = re.sub('[^a-zA-Z]', ' ', rev)
    # split string and package into list
    rev = rev.split()
    # run stopwords via nltk
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    # join all words
    rev = ' '.join(rev)
    corpus.append(rev)
    