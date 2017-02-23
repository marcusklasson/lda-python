# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:54:45 2017

@author: mklas
"""
# stop words from http://www.textfixer.com/tutorials/common-english-words.txt


import numpy as np
from collections import Counter 

class Documents:
    
    def __init__(self, file_name, vocabulary_file, stopwords_file):
        texts = []
        special_chars = '!"#$£@%&/()=?.,+-*\':;_´`1234567890'
        with open(file_name, 'r') as infile:
            copy = False
            text = ''
            for line in infile:
                if copy:
                    if line.strip() == '</TEXT>':
                        #print(text)
                        text = text.lower()
                        texts.append(text)
                        #print(text)
                        text = ''
                        copy = False
                    else:
                        for char in special_chars:
                            line = line.replace(char, '')
                        text += line
                else:
                    if line.strip() == '<TEXT>':
                        copy = True
        tmp_texts = np.array(texts)
        
        self.vocabulary = np.genfromtxt(vocabulary_file,  dtype='str')
        
        stop_words_line = []
        with open(stopwords_file, 'r') as infile:
            data=infile.read().replace(',', ' ')
            for word in data.split():
                stop_words_line.append(word)
        
        self.stop_words = np.array(stop_words_line)
        
        self.documents = []
        for text in tmp_texts:
            words = np.array(text.split())
            
            stopwords_filtered_document = [w for w in words if w not in self.stop_words]
            single_words = [k for k, v in Counter(stopwords_filtered_document).iteritems() if v == 1 ]
            final_filtered_document = [w for w in stopwords_filtered_document if w not in single_words]
            #vocabulary_filtered_document = [w for w in stopwords_filtered_document if w not in self.vocabulary]
            if not final_filtered_document: # Document is empty, Shape = []
                continue
            self.documents.append(final_filtered_document)
