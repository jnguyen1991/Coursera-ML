# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:25:13 2018

@author: Jon
"""
'''
ex2
'''
import numpy as np
import matplotlib as mp
import math

fname = r'C:\Users\Jon\Documents\GitHub\Coursera-ML\ex2\ex2data1.txt'

def load(fname):
    '''
    load data in, takes fname as input
    '''
    
    data = np.loadtxt(fname, delimiter=',')
    X = data[:,(0,1)]
    Y = data[:,[2]]
    return data, X, Y

def plotData(X,Y):
    '''
    plots the data using matplotlib
    chooses color based on Y vector
    '''
    mp.pyplot.scatter(X[:,0],X[:,1], c=Y[:,0])
    
def sigmoid(z):
    sig = lambda x:1/(1+math.exp(-x))
    vfunc = np.vectorize(sig)
    g = vfunc(z)
    return g

def costFunction(theta, X, Y):
    m = len(Y)
    print(Y)
    J = 0
    grad = np.zeros(shape(theta))
    return J, grad
    
if __name__ == "__main__":
    data,X,Y = load(fname)
    X = np.hstack( (np.ones( (X[:,0].size,1) ), X) )
    initial_theta = np.zeros((X[0].size,1))
    print(initial_theta)
    #plotData(X,Y)


