import numpy as np
import skimage.measure
from Activations.activations import *

class Pool2D:
    def __init__(self, inputNumber, inputDim, activation: Activations):
        self.inputNumber = inputNumber
        self.inputDim = inputDim
        self.activation = activation
    
    def forward(self, input):
        self.input = np.copy(input)
        output = np.zeros((input.shape[0], input.shape[1]//2, input.shape[2]//2))
        output = self.activation.forward(input)
        self.output = np.copy(output)
        return output

    def updateGradients(self, nodeValues):
        nodeValues = nodeValues.reshape(self.inputNumber, self.inputDim//2, self.inputDim//2)

        restoredOutput = self.activation.restore(nodeValues)

        return restoredOutput