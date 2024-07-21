import numpy as np
from scipy import signal
from Activations.activations import *
from Settings.settings import optimizer

#This file contains the class for the convolutional layer

class Conv2D:
    def __init__(self, kernelNumber, kernelSize, combinations, activation: Activations, outputSize):
        self.kernelNumber = kernelNumber
        self.kernelSize = kernelSize
        self.kernels = self.initializeKernels(kernelNumber, kernelSize)
        self.kernelsGradients = np.zeros((kernelNumber, kernelSize, kernelSize))
        self.kernelsGradientsHistory = np.zeros((kernelNumber, kernelSize, kernelSize))

        self.outputSize = outputSize

        self.isOutputPooled = True

        self.combinations = combinations
        self.activation = activation
        self.momentum = 0.9

        #FOR ADAM
        self.numIterations = 0
        self.optim = optimizer()

    
    def initializeKernels(self, kernelNumber, kernelSize):
        return np.random.randn(kernelNumber, kernelSize, kernelSize)

    #function to compute the output of the layer
    def forward(self, inputs):
        self.inputs = np.copy(inputs)

        self.outputs = np.zeros((self.kernelNumber, self.outputSize, self.outputSize))

        sumOfAllInputs = np.sum(inputs, axis=0)
        for idx_k, kernel in enumerate(self.kernels):
            inputSum = np.copy(sumOfAllInputs)
            for input_idx in self.combinations[idx_k]:
                inputSum -= inputs[input_idx]             
            self.outputs[idx_k] += signal.correlate2d(inputSum, kernel, "valid")

        self.outputs = self.activation.forward(self.outputs) #pass the outputs in the activation function
        return self.outputs                                  # and then return


    #function to update the gradients f the layer based on gradient descent
    def updateGradients(self, nodeValues):
        nodeValues = nodeValues.reshape(self.kernelNumber, self.outputs.shape[1], self.outputs.shape[2])

        activationDerivatives = np.copy(self.activation.derivative(self.outputs))

        nodeValuesWithActDerivative = nodeValues * activationDerivatives

        inputsGradients = np.zeros(self.inputs.shape)
        sumOfAllInputs = np.sum(self.inputs, axis=0)
        for idx_k, kernel in enumerate(self.kernels):
            inputSum = np.copy(sumOfAllInputs)
            for input_idx in self.combinations[idx_k]:
                inputSum -= self.inputs[input_idx]
            self.kernelsGradients[idx_k] += signal.correlate2d(inputSum, nodeValuesWithActDerivative[idx_k], "valid")
            inputGradient = signal.convolve2d(nodeValuesWithActDerivative[idx_k], self.kernels[idx_k], "full")
            inputsGradients += inputGradient
            for input_idx in self.combinations[idx_k]:
                inputsGradients[input_idx] -= inputGradient

        return inputsGradients
        
    #updates the weight for the next iteration
    def applyGradients(self, learnRate):
        self.numIterations += 1
        dKernels = self.optim.optimizeConv(kGradients=self.kernelsGradients, learnRate=learnRate, t=self.numIterations)
        self.kernels -= dKernels


    def resetGradients(self):
        self.kernelsGradients = np.zeros((self.kernelNumber, self.kernelSize, self.kernelSize))


