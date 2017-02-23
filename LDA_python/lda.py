# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:17:38 2017

@author: mklas
"""

import numpy as np
from scipy.special import digamma, gammaln
from corpus import Documents
from synthetic_data import SyntheticDataLDA



def get_words_from_doc(doc):
    #return np.array(doc.split())
    return np.array(doc)
    
def word_pos_in_doc(words, word):
    bool_indicator = np.in1d(words, word)
    #vocab_idx = np.where(vocabulary == word)
    return bool_indicator.astype(int) 

def word_pos_in_vocab(vocabulary, word):
    #bool_indicator = np.in1d(words, word)
    vocab_idx = np.where(vocabulary == word)
    return vocab_idx 
    
def initialize_parameters(documents, vocabulary, k, M):
    Phi = []
    gamma = np.zeros([M,k])
    alpha = np.ones([M,k])
    for m in range(0,M):
        doc = get_words_from_doc(documents[m])
        N = len(doc)
        phi = np.ones([N,k]) * 1/float(k)
        for i in range(0,k):
            gamma[m, i] = alpha[m, i] + N/float(k)
        Phi.append(phi)
        m += 1
    # Initialize Beta
    Beta = np.zeros([k,len(vocabulary)])
    for i in range(0,k):
        tmp = np.random.uniform(0, 1, len(vocabulary))
        Beta[i,:] = tmp / np.sum(tmp)
    return Phi, gamma, alpha, Beta
                
    
def compute_likelihood(Phi, gamma, alpha, Beta, document, vocabulary, k):
    likelihood = 0.0
    V = len(vocabulary)
    words = get_words_from_doc(document)
    N = len(words)
    
    alpha_sum = 0.0
    phi_gamma_sum = 0.0
    phi_logbeta_sum = 0.0
    entropy_sum = 0.0
    gamma_sum = 0.0
    
    alpha_sum += gammaln(np.sum(alpha))  
    gamma_sum -= gammaln(np.sum(gamma)) 
    for i in range(0,k):
        alpha_sum += -gammaln(alpha[i]) + \
                (alpha[i] - 1) * (digamma(gamma[i]) - digamma(np.sum(gamma)))
        
        for n in range(0,N):
            if Phi[n,i] > 0:
                w_indicator = np.sum(np.in1d(vocabulary, words[n]))   
                phi_gamma_sum += Phi[n,i] * (digamma(gamma[i]) - digamma(np.sum(gamma[:])))
                entropy_sum += Phi[n,i] * np.log(Phi[n,i])
                for j in range(0,V):
                    if Beta[i,j] > 0:
                        phi_logbeta_sum += Phi[n,i] * w_indicator * np.log(Beta[i,j])
            
        gamma_sum += gammaln(gamma[i]) - \
                    (gamma[i] - 1) * (digamma(gamma[i]) - digamma(np.sum(gamma[:])))
    
    likelihood += (alpha_sum + phi_gamma_sum + phi_logbeta_sum - gamma_sum - entropy_sum) 
    return likelihood

def E_step(Phi, gamma, alpha, Beta, documents, vocabulary, k, M):
    print('E-step')
    likelihood = 0.0
    convergence_indicator = np.zeros(M)
    for d in range(0,M):
        words = get_words_from_doc(documents[d])
        N = len(words)
        phi = Phi[d]
        
        conv_counter = 0
        while convergence_indicator[d] == 0 and d < len(convergence_indicator):
            
            phi_old = phi
            phi = np.zeros([N,k])
            gamma_old = gamma[d, :]
            
            for n in range(0,N):
                word = words[n]
                vocab_idx = word_pos_in_vocab(vocabulary, word)
                if len(vocab_idx[0]) > 0: # word does not exist in vocabulary
                    for i in range(0,k):                
                        beta = Beta[i, vocab_idx]
                        phi[n, i] = beta[0][0] * np.exp(digamma(gamma[d,i]) - digamma(np.sum(gamma[d,:])))
                    phi[n,:] = phi[n,:] / np.sum(phi[n,:])   
            gamma[d, :] = alpha[d, :] + np.sum(phi, axis=0)    
            
            conv_counter += 1
            # Check if gamma and phi converged
            if np.linalg.norm(phi - phi_old) < 1e-3 and np.linalg.norm(gamma[d,:] - gamma_old) < 1e-3:
                convergence_indicator[d] = 1
                Phi[d] = phi               
                print('Document ' + str(d) + ' needed ' + str(conv_counter) + ' iterations to converge.')
                
                likelihood += compute_likelihood(Phi[d], gamma[d,:], alpha[d,:], Beta, documents[d], vocabulary, k)
                
    return Phi, gamma, likelihood
    
def M_step(Phi, gamma, alpha, documents, vocabulary, k, M):
    print('M-step')
    V = len(vocabulary)
    
    Beta = np.zeros([k,V])
    for d in range(0,M):
        words = get_words_from_doc(documents[d])
        Phi_d = Phi[d]
        for i in range(0,k):
            phi = Phi_d[:,i]
            for j in range(0,V):
                word = vocabulary[j]
                indicator = word_pos_in_doc(words, word)
                Beta[i,j] += np.dot(indicator, phi)
    Beta = np.transpose(np.transpose(Beta) / np.sum(Beta, axis=1))
    
    alpha_new = alpha
    return alpha_new, Beta
    
def variational_EM(Phi_init, gamma_init, alpha_init, Beta_init, documents, vocabulary, k, M):
    print('Variational EM')
    likelihood = 0
    likelihood_old = 0
    iteration = 1 # Initialization step is the first step
    Phi= Phi_init
    gamma = gamma_init
    alpha = alpha_init
    Beta = Beta_init
    while iteration <= 2 or np.abs((likelihood-likelihood_old)/likelihood_old) > 1e-4:
        # Update parameters 
        likelihood_old = likelihood
        Phi_old = Phi 
        gamma_old = gamma 
        alpha_old = alpha
        Beta_old = Beta
    
        Phi, gamma, likelihood = \
            E_step(Phi_old, gamma_old, alpha_old, Beta_old, documents, vocabulary, k, M)
        alpha, Beta = \
            M_step(Phi, gamma, alpha_old, documents, vocabulary, k, M)
                
        print('Iteration ' + str(iteration) + ': Likelihood = ' + str(likelihood))
        iteration += 1
        
        if iteration > 100:
            break
        
    return Phi, gamma, alpha, Beta, likelihood
    
def inference_method(documents, vocabulary):
    #M = len(documents)   # nbr of documents
    M=100
    k = 4          # nbr of latent states z
    
    Phi_init, gamma_init, alpha_init, Beta_init = initialize_parameters(documents, vocabulary, k, M)

    Phi, gamma, alpha, Beta, likelihood = \
        variational_EM(Phi_init, gamma_init, alpha_init, Beta_init, documents, vocabulary, k, M)
    
    return Phi, gamma, alpha, Beta, likelihood
        

# Main Program
vocabulary = np.genfromtxt('ap/vocab.txt',  dtype='str')
document_data = Documents('ap/ap.txt', 'ap/vocab.txt', 'ap/stopwords.txt')
Phi, gamma, alpha, Beta, likelihood = inference_method(document_data.documents, document_data.vocabulary)

"""
size_vocabulary = 500
synthetic_data = SyntheticDataLDA(200, size_vocabulary)
nbr_vocabulary = np.arange(0,size_vocabulary)

Phi, gamma, alpha, Beta, likelihood = inference_method(synthetic_data.documents, nbr_vocabulary)
    
true_Beta = synthetic_data.Beta_true 
true_topic_dist = synthetic_data.true_distributions

tmp = []
for d in range(0, np.shape(gamma)[0]):
    tmp.append(gamma[d,:] / np.sum(gamma[d,:]))
gamma_norm = np.array(tmp)
"""