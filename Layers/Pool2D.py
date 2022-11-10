import numpy as np
import skimage.measure
from numba import jit
from numba import cuda
cuda.select_device(0)

@jit(nopython=True)
def backwardJIT(nodeValues):
    restoredOutput = np.zeros((nodeValues.shape[0], nodeValues.shape[1]*2, nodeValues.shape[2]*2))
    for i in range(nodeValues.shape[0]):
        for j in range(nodeValues.shape[1]):
            for k in range(nodeValues.shape[2]):
                restoredOutput[i][j*2][k*2] = nodeValues[i][j][k]
                restoredOutput[i][j*2][k*2+1] = nodeValues[i][j][k]
                restoredOutput[i][j*2+1][k*2] = nodeValues[i][j][k]
                restoredOutput[i][j*2+1][k*2+1] = nodeValues[i][j][k]
    return restoredOutput


class Pool2D:
    def __init__(self, inputNumber, inputDim, mode):
        self.inputNumber = inputNumber
        self.inputDim = inputDim
        self.mode = mode
    
    def forward(self, input):
        output = np.zeros((input.shape[0], input.shape[1]//2, input.shape[2]//2))
        if self.mode == "mean":
            output = skimage.measure.block_reduce(input, (1, 2, 2), np.mean)
        elif self.mode == "max":
            output = skimage.measure.block_reduce(input, (1, 2, 2), np.max)
        self.output = np.copy(output)
        return output

    def updateGradients(self, nodeValues):
        nodeValues = nodeValues.reshape(self.inputNumber, self.inputDim//2, self.inputDim//2)
        unpooledOutput = backwardJIT(nodeValues)

        return unpooledOutput