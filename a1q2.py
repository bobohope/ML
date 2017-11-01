# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
#from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp
from sklearn.cross_validation import KFold
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
#x = x[0:50,:]
d = x.shape[1]
N = x.shape[0]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        #print(predictions)
        losses[j] = ((predictions-y_test)**2).mean()
    print(losses)
    return losses


#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=0):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO get matrix A
    N_train = x_train.shape[0]
    d=x_train.shape[1]
    A=np.zeros((N_train,N_train))
    distance=l2(test_datum,x_train)
    deno=np.exp(logsumexp(-distance/(2*tau**2)))
    #print(distance)
    Aj=-distance/(2*tau**2)
    B=np.max(Aj)
    Adiag=np.exp(Aj-B)/np.sum(np.exp(Aj-B))
    A=Adiag*np.identity(N_train)
#     for i in range (N_train):
#         #print(test_datum.shape)
#         numerator=np.exp(logsumexp(-distance[:,i]/(2*tau**2)))
#         #print(numerator/deno)
#         A[i,i]=numerator/deno
    ## TODO solve for w* and get y_head
    #print(x_train.transpose().dot(A).dot(y_train))
    w=np.linalg.solve(x_train.transpose().dot(A).dot(x_train)+lam*np.identity(d),x_train.transpose().dot(A).dot(y_train))
    #print(w)
    y_head=test_datum.dot(w)
    #print (y_head)
    return y_head


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO get k folds
    N=x.shape[0]
    d=x.shape[1]
    kf = KFold(N, n_folds=k)
    losses=np.zeros((k,len(taus)))
    i=0
    for trainidx, testidx in kf:
        x_train=x[idx[trainidx],:]
        x_test=x[idx[testidx],:]
        y_train=y[idx[trainidx]]
        y_test=y[idx[trainidx]]
        #print(x_train)
        #print(idx[testidx])
        losses[i,:]=run_on_fold(x_test, y_test, x_train, y_train, taus)
        i+=1
    print(losses)
    print(losses.shape)
    avgloss=np.ones((1,k)).dot(losses)/k
    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    #print(x.shape)
    taus = np.logspace(3,10,200)
    losses = run_k_fold(x,y,taus,k=5)
    #print(losses)
    #plt.plot(losses)
    #print("min loss = {}".format(losses.min()))
    #plt.show()
