import numpy as np
import skimage.measure

class Pool2D:
    def __init__(self, inputNumber, inputDim, mode):
        self.inputNumber = inputNumber
        self.inputDim = inputDim
        self.mode = mode
    
    def forward(self, input):
        self.input = np.copy(input)
        output = np.zeros((input.shape[0], input.shape[1]//2, input.shape[2]//2))
        if self.mode == "MEAN":
            output = skimage.measure.block_reduce(input, (1, 2, 2), np.mean)
        elif self.mode == "MAX":
            output = skimage.measure.block_reduce(input, (1, 2, 2), np.max)
        self.output = np.copy(output)
        return output

    def updateGradients(self, nodeValues):
        nodeValues = nodeValues.reshape(self.inputNumber, self.inputDim//2, self.inputDim//2)
        if self.mode == "MEAN":
            restoredOutput = np.zeros((nodeValues.shape[0], nodeValues.shape[1]*2, nodeValues.shape[2]*2))
            for i in range(nodeValues.shape[0]):
                for j in range(nodeValues.shape[1]):
                    for k in range(nodeValues.shape[2]):
                        restoredOutput[i][j*2][k*2] = nodeValues[i][j][k]
                        restoredOutput[i][j*2][k*2+1] = nodeValues[i][j][k]
                        restoredOutput[i][j*2+1][k*2] = nodeValues[i][j][k]
                        restoredOutput[i][j*2+1][k*2+1] = nodeValues[i][j][k]
        elif self.mode == "MAX":
            #pass the input to see what was the greatest value for each 2x2 square, since the value has to be assigned to that locatino only (the other 3 squares will have 0 as value)
            restoredOutput = np.zeros((nodeValues.shape[0], nodeValues.shape[1]*2, nodeValues.shape[2]*2))
            for i in range(nodeValues.shape[0]):
                for j in range(nodeValues.shape[1]):
                    for k in range(nodeValues.shape[2]):
                        #np.argmax not working for some reason
                        firstIsGreater = ([self.input[i][j][k] >= self.input[i][j][k+1] and self.input[i][j][k] >= self.input[i][j+1][k] and self.input[i][j][k] >= self.input[i][j+1][k+1]])

                        secondIsGreater = ([self.input[i][j][k+1] >= self.input[i][j][k] and self.input[i][j][k+1] >= self.input[i][j+1][k] and self.input[i][j][k+1] >= self.input[i][j+1][k+1]])

                        thirdIsGreater = ([self.input[i][j+1][k] >= self.input[i][j][k+1] and self.input[i][j+1][k] >= self.input[i][j][k] and self.input[i][j+1][k] >= self.input[i][j+1][k+1]])

                        if firstIsGreater:
                            restoredOutput[i][j*2][k*2] = nodeValues[i][j][k]
                        elif secondIsGreater:
                            restoredOutput[i][j*2][k*2+1] = nodeValues[i][j][k]
                        elif thirdIsGreater:
                            restoredOutput[i][j*2+1][k*2] = nodeValues[i][j][k]
                        else:
                            restoredOutput[i][j*2+1][k*2+1] = nodeValues[i][j][k]


        return restoredOutput