import numpy as np
import skimage.measure

# This file contains the actvations functions and functions to calculate its derivatives
class Activations:
    class Relu:
        def forward(self, input):
            input[input < 0] = 0
            return input
        
        def derivative(self, input):
            input[input > 0] = 1
            return input

    class Sigmoid:
        def forward(self, input):
            input = 1 / (1+ np.exp(-input))
            return input
        
        def derivative(self, input):
            return input*(1-input)


    class Softmax:
        def forward(self, input):
            max = np.max(input)
            expsum = np.sum(np.exp(input-max))
            return np.exp(input-max) / expsum
        
        def nodeValuesWithCrossEntropy(self, expected_output, output):
            nodeValues = np.empty_like(output)
            for nodeValueIdx in range(len(nodeValues)):
                sum = 0
                for j in range(output.shape[0]):
                    if nodeValueIdx == j:
                        sum -= (1-output[nodeValueIdx]) * (expected_output[nodeValueIdx])
                    else:
                        sum -= -output[nodeValueIdx] * (expected_output[j])
                nodeValues[nodeValueIdx] = sum
            return nodeValues


    class Mean:
        def forward(self, input):
            return skimage.measure.block_reduce(input, (1, 2, 2), np.mean)
        
        def restore(self, nodeValues):
            
            restoredOutput = np.zeros((nodeValues.shape[0], nodeValues.shape[1]*2, nodeValues.shape[2]*2))
            for i in range(nodeValues.shape[0]):
                for j in range(nodeValues.shape[1]):
                    for k in range(nodeValues.shape[2]):
                        restoredOutput[i][j*2][k*2] = nodeValues[i][j][k]
                        restoredOutput[i][j*2][k*2+1] = nodeValues[i][j][k]
                        restoredOutput[i][j*2+1][k*2] = nodeValues[i][j][k]
                        restoredOutput[i][j*2+1][k*2+1] = nodeValues[i][j][k]
            return restoredOutput


    class Max:
        def forward(self, input):
            return skimage.measure.block_reduce(input, (1, 2, 2), np.max)
        
        def restore(self, nodeValues):
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