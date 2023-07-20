import numpy as np

# This file contains the actvations functions and functions to calculate its derivatives

def relu(vals):
    vals[vals < 0] = 0
    return vals

def relu_derivative(vals):
    vals[vals > 0] = 1
    return vals

def sigmoid(vals):
    vals = 1 / (1+ np.exp(-vals))
    return vals

def sigmoid_derivative(vals):
    return vals*(1-vals)

def softmax(vals):
    max = np.max(vals)
    expsum = np.sum(np.exp(vals-max))
    return np.exp(vals-max) / expsum