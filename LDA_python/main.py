# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:21:45 2017

@author: mklas
"""

from synthetic_data import SyntheticDataLDA
from corpus import Documents

docs = Documents('ap/ap.txt', 'ap/vocab.txt', 'ap/stopwords.txt')

dokument = docs.documents

synt_data = SyntheticDataLDA(10, 20)

Beta = synt_data.Beta_true
dist = synt_data.topic_distributions
docs = synt_data.documents