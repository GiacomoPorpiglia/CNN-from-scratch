import numpy as np
from scipy import signal
import skimage.measure
from numba import jit
from numba import cuda

from Adam import Adam

e = 2.718281828
cuda.select_device(0)

@jit(nopython=True)
def restoreDimensionsBeforePoolingJIT(nodeValues, pooledOutputs, activation):

    restoredOutput = np.zeros((nodeValues.shape[0], nodeValues.shape[1]*2, nodeValues.shape[2]*2))
    if activation == "RELU":
        activationDerivatives = pooledOutputs
    elif activation == "SIGMOID":
        activationDerivatives = pooledOutputs*(1-pooledOutputs)

    for i in range(nodeValues.shape[0]):
        for j in range(nodeValues.shape[1]):
            for k in range(nodeValues.shape[2]):
                restoredOutput[i][j*2][k*2] = nodeValues[i][j][k]*activationDerivatives[i][j][k]
                restoredOutput[i][j*2][k*2+1] = nodeValues[i][j][k]*activationDerivatives[i][j][k]
                restoredOutput[i][j*2+1][k*2] = nodeValues[i][j][k]*activationDerivatives[i][j][k]
                restoredOutput[i][j*2+1][k*2+1] = nodeValues[i][j][k]*activationDerivatives[i][j][k]
    return restoredOutput


class Convolutional:
    def __init__(self, kernelNumber, kernelSize, combinations, activation, outputSize):
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
        self.optim = Adam()

    
    def initializeKernels(self, kernelNumber, kernelSize):
        return np.random.randn(kernelNumber, kernelSize, kernelSize)

    def maxPool(self, outputs):
        pooledOutputs = skimage.measure.block_reduce(outputs, (1, 2, 2), np.max)
        return pooledOutputs
    def meanPool(self, outputs):
        pooledOutputs = skimage.measure.block_reduce(outputs, (1, 2, 2), np.mean)
        return pooledOutputs
    
    def forward(self, inputs):
        self.inputs = np.copy(inputs)

        self.outputs = np.zeros((self.kernelNumber, self.outputSize, self.outputSize))

        sumOfAllInputs = np.sum(inputs, axis=0)
        for idx_k, kernel in enumerate(self.kernels):
            inputSum = np.copy(sumOfAllInputs)
            for input_idx in self.combinations[idx_k]:
                inputSum -= inputs[input_idx]             
            self.outputs[idx_k] += signal.correlate2d(inputSum, kernel, "valid")

        if self.outputs.shape[1]%2 == 0:
            self.isOutputPooled = True
            self.pooledOutputs = self.meanPool(self.outputs)

            if self.activation == "RELU":
                self.pooledOutputs[self.pooledOutputs < 0 ] = 0
            elif self.activation == "SIGMOID":
                self.pooledOutputs = 1 / (1+ np.power(e, -self.pooledOutputs))#  e**(-self.pooledOutputs))
            return self.pooledOutputs
        else:
            self.isOutputPooled = False
            if self.activation == "RELU":
                self.outputs[self.outputs < 0 ] = 0
            elif self.activation == "SIGMOID":
                self.outputs = 1 / (1+ e**(-self.outputs))
            return self.outputs


    def restoreDimensionsBeforePooling(self, nodeValues):
        restoredOutput = restoreDimensionsBeforePoolingJIT(nodeValues, self.pooledOutputs, self.activation)
        return restoredOutput


    def updateGradients(self, nodeValues):
        if self.isOutputPooled:        
            nodeValues = nodeValues.reshape(self.kernelNumber, self.pooledOutputs.shape[1], self.pooledOutputs.shape[2])
            unpooledNVWithActDerivative = self.restoreDimensionsBeforePooling(nodeValues)
        else:
            if self.activation=="RELU":
                unpooledNVWithActDerivative = nodeValues.reshape(self.kernelNumber, self.outputs.shape[1], self.outputs.shape[2])
            elif self.activation == "SIGMOID":
                unpooledNVWithActDerivative = nodeValues.reshape(self.kernelNumber, self.outputs.shape[1], self.outputs.shape[2]) * (self.outputs*(1-self.outputs))

        inputsGradients = np.zeros(self.inputs.shape)
        sumOfAllInputs = np.sum(self.inputs, axis=0)
        for idx_k, kernel in enumerate(self.kernels):
            inputSum = np.copy(sumOfAllInputs)
            for input_idx in self.combinations[idx_k]:
                inputSum -= self.inputs[input_idx]
            self.kernelsGradients[idx_k] += signal.correlate2d(inputSum, unpooledNVWithActDerivative[idx_k], "valid")
            inputGradient = signal.convolve2d(unpooledNVWithActDerivative[idx_k], self.kernels[idx_k], "full")
            inputsGradients += inputGradient
            for input_idx in self.combinations[idx_k]:
                inputsGradients[input_idx] -= inputGradient

        return inputsGradients
        
    
    def applyGradients(self, learnRate):
        #ADAM
        self.numIterations += 1
        dKernels = self.optim.optimizeConv(kGradients=self.kernelsGradients, learnRate=learnRate, t=self.numIterations)
        self.kernels -= dKernels


    def resetGradients(self):
        self.kernelsGradients = np.zeros((self.kernelNumber, self.kernelSize, self.kernelSize))


