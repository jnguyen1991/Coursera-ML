# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:28:05 2018

@author: jnguyen3
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def computeCost(X, Y, theta):
    m = len(Y)
    #use X and multply against theta to find expected value
    #then take the difference with Y to find difference from actuals
    difference = np.dot(X,theta) - Y
    #square distances
    squared = np.square(difference)
    #sum values, and divide by 2*m
    J = sum(squared)/(2*m)
    return J
    
def gradientDescent(X, Y, theta, alpha, num_iters):
    m = len(Y)
    J_history = np.zeros((num_iters, 1))    
    #print theta[0],theta[1]
    for i in range(num_iters):
        temp = np.zeros((len(theta),1))
        for t in range(len(theta)):
            xt = np.transpose(X[:,[t]])
            temp[t] = theta[t] - alpha / m * sum(np.dot(xt,np.dot(X,theta) - Y))
        theta = np.array(temp)
        J_history[i] = computeCost(X, Y, theta)
    return theta, J_history

def featureNormalize(X):
    X_norm = X
    length = len(X.T)
    mu = np.zeros((1,length))
    sigma = np.zeros((1,length))
    
    for i in range(len(X.T)):
        mu[:,i] = np.mean(X[:,[i]])
        sigma[:,i] = np.std(X[:,[i]])
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    return X_norm, mu, sigma

path = r"C:\Users\jnguyen3\Documents\GitHub\Coursera-ML\ex1\ex1data2.txt"
data_nn = np.loadtxt(path, delimiter=",")
data, mu, sigma = featureNormalize(data_nn)
X = data[:,:-1]
Y = data[:,[-1]]
ones = np.ones((len(data),1))
X = np.hstack((ones,X))
theta = np.zeros((len(X.T),1))
iterations = 1500
alpha = 0.01

theta,J_history = gradientDescent(X, Y, theta, alpha, iterations)
