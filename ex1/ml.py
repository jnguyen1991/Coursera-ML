# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:28:05 2018

@author: jnguyen3
"""

import numpy as np
import matplotlib.pyplot as plt

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
        temp0 = theta[0] - alpha / m * sum(np.dot(X,theta) - Y)
        t1 = np.transpose(X[:,[1]])
        temp1 = theta[1] - alpha / m * sum(np.dot(t1,np.dot(X,theta) - Y))
        theta = np.array([temp0, temp1])
        J_history[i] = computeCost(X, Y, theta)
    return theta, J_history

#2 Linear Regression with one variable
#2.1 Plotting the data
path = r"C:\Users\jnguyen3\Documents\GitHub\Coursera-ML\ex1\ex1data1.txt"
data = np.loadtxt(path, delimiter=",")
#print(data)
X = data[:,[0]]
Y = data[:,[1]]
m = len(Y)

plt.plot(X, Y, 'rx', MarkerSize = 10)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')



#2.2 Gradient Descent
#2.2.1 Update Equations
X0 = np.ones((m,1))

X = np.hstack((X0,X))
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

computeCost(X, Y, theta)

theta,J_history = gradientDescent(X, Y, theta, alpha, iterations)

x = np.linspace(4,23,100)
y = theta[0] + theta[1]*x
plt.plot(x,y)

plt.show()

#2.4 Visualizing J(theta)
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i],theta1_vals[j]]
        J_vals[i,j] = computeCost(X, Y, t)

J_vals = J_vals.T

print J_vals

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

X, Y = np.meshgrid(theta0_vals, theta1_vals)


#CS = ax1.contour(

