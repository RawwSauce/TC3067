# -*- coding: utf-8 -*-

### Rene Gagnon
### A01757655

import numpy as np
import matplotlib.pyplot as plt  


def gradientDescent(X, y, theta, alpha, m, numIterations):
    
    temp = np.zeros((2,1))
    x = np.delete(X, 0, 1).T
    y = y.T
    
    for i in range(0,numIterations):
        temp[0,0] = theta[0,0] - (alpha * (1/m) * np.sum((theta[0,0] + (theta [1,0] * x)) - y))
        temp[1,0] = theta[1,0] - (alpha * (1/m) * np.sum(((theta[0,0] + (theta [1,0] * x)) - y) * x))
        theta = temp
        

    return theta

# Defining parameters
theta = np.ones((2,1))
alpha = 0.01
numIterations = 1000

# Reading data from file defining parameters
X = np.loadtxt("housesData.txt", delimiter = ",")
m, col = X.shape
y = np.delete(X, 0, 1)
X = np.c_[np.ones(m),np.delete(X, 1, 1)]

print("Initial theta: ", theta)

# Get theta using gradient descent and make prediction
theta = gradientDescent(X, y, theta, alpha, m, numIterations)
test_X = np.array([[5],[22.5]])
test_X_b = np.c_[np.ones((2, 1)), test_X]
prediction = test_X_b.dot(theta)
print("Final theta: ", theta)

# Plot results
plt.plot(np.delete(X,0,1), y, "bo")
plt.plot(test_X, prediction, "r--")
plt.show()