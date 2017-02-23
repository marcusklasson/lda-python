# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:36:35 2017

@author: mklas
"""
import numpy as np

class SyntheticDataLDA:
    
    def __init__(self, nbr_documents, size_vocabulary):
        
        self.nbr_topics = 3
        #self.topic_probability = np.array([0.5, 0.3, 0.2])
        #self.topic_probability = np.random.dirichlet([1,1,1])
        self.topic_distributions = np.array([[0.5, 0.3, 0.2], 
                                             [0.0, 0.25, 0.75]])
        #                                     [0.1, 0.5, 0.4]])
        #                                     [0.3, 0.7, 0.0], 
        #                                     [0.1, 0.9, 0.0],
        #                                     [0.3, 0.3, 0.4]])
        self.true_distributions = []
        
        self.alpha_true = np.ones([1,self.nbr_topics])
        self.Beta_true = np.zeros([self.nbr_topics, size_vocabulary])
        
        for v in range(0, size_vocabulary):
            self.Beta_true[:,v] = np.random.dirichlet([1,1,1])
        self.Beta_true = np.transpose(np.transpose(self.Beta_true) / np.sum(self.Beta_true, axis=1))
        #nbr_documents = np.shape(self.topic_distributions)[0]
        self.documents = []
        for d in range(0,nbr_documents):
            words = []
            N = np.random.poisson(100)
            if d % 2 == 0:
                topic_probability = self.topic_distributions[0]
            else:
                topic_probability = self.topic_distributions[1]

            
            self.true_distributions.append(topic_probability)
            #topic_probability = np.random.dirichlet(self.alpha_true[0])
            #self.topic_distributions.append(topic_probability)
            #topic_probability = self.topic_distributions[d]
            topic_distribution = np.cumsum(topic_probability)
            for n in range(0,N):
                # Sample from theta (topic_probability)
                z_sample = np.random.rand()
                if z_sample <= topic_distribution[0]:
                    z = 1
                elif z_sample > topic_distribution[0] and z_sample <= topic_distribution[1]:
                    z = 2
                else:
                    z = 3
                
                # Sample from Beta
                beta = self.Beta_true[z-1,:]
                beta_distribution = np.cumsum(beta)
                w_sample = np.random.rand()
                
                if w_sample <= beta_distribution[0]:
                    words.append(0)
                else:
                    for k in range(1,size_vocabulary):
                        if w_sample > beta_distribution[k-1] and w_sample <= beta_distribution[k]:
                            words.append(k)
            
            self.documents.append(words)