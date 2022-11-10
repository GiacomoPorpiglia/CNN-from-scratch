import numpy as np
from math import sqrt
from numba import jit
from numba import cuda
from Adam import Adam
from Activations import activations

cuda.select_device(0)

@jit(nopython=True)
def calculateOutputLayerNodeValuesJIT(nodeValues, output, expected_output):
    for nodeValueIdx in range(len(nodeValues)):
        sum = 0
        for j in range(output.shape[0]):
            if nodeValueIdx == j:
                sum -= (1-output[nodeValueIdx])*(expected_output[nodeValueIdx])
            else:
                sum -= -output[nodeValueIdx] *(expected_output[j])
        nodeValues[nodeValueIdx] = sum
    return nodeValues


@jit(nopython=True)
def calculaeteHiddenLayerNodeValuesJIT(oldLayerWeights, oldNodeValues, activationDerivatives):
    return np.dot((oldLayerWeights), oldNodeValues) * activationDerivatives


class LayerDense:
    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.5*np.random.randn(n_inputs, n_neurons)
        self.biases = 0.5*np.random.randn(n_neurons)

        self.activation = activation

        self.output = np.zeros(self.n_neurons)

        self.costGradientW = self.initializeWeights()
        self.costGradientB = np.zeros(n_neurons)

        self.costGradientWHistory = np.zeros((n_inputs, n_neurons))
        self.costGradientBHistory = np.zeros(n_neurons)

        self.nodeValues = np.zeros(self.n_neurons)
        self.inputs = np.zeros(self.n_inputs)
        self.weightedInputs = np.zeros(self.n_neurons)

        self.optim = Adam()

        #FOR ADAM
        self.numIterations = 0

    #This function initializes the weights of the layer, according to He weight initialization based on Gaussin distribution
    def initializeWeights(self):
        #He weight initialization
        std = sqrt(2 / self.n_inputs)
        weights = np.random.randn(self.n_inputs, self.n_neurons) * std
        return weights

    #this function updates the gradients using the node values of the layer
    #for the weight gradients, it multiplies the node value of the output node by the input of the input node (according to the partial derivative calculation of backpropagation)
    #for the bias gradients, since the derivative of the weighted input in respect to the bias is 1, it multiplies the node value of the output node by 1
    def updateGradients(self):
        #updates costGradientW and costGradientB
        self.costGradientW += np.dot(self.inputs[:,None], self.nodeValues[None,:])
        self.costGradientB += 1*self.nodeValues


    #this function applies the gradients to the weights and biases of each layer, according to the learn rate --- it subtracts the corresponding gradient to each weight and bias, multiplied by the learn rate 
    def applyGradients(self, learnRate):
        #updates weights and biases by adding the gradients multiplied by the learn rate
        #ADAM
        self.numIterations+=1
        dWeights, dBiases = self.optim.optimizeFC(wGradients=self.costGradientW, bGradients=self.costGradientB, learnRate=learnRate, t=self.numIterations)
        self.weights -= dWeights
        self.biases -= dBiases


    #this function rests all the gradients to 0, after the network applied them
    def resetGradients(self):
        self.costGradientW = np.zeros((self.n_inputs, self.n_neurons))
        self.costGradientB = np.zeros(self.n_neurons)

    #this function calculates the node values for the output layer
    def calculateOutputLayerNodeValues(self, expected_output):
        #calculates the node values for the output layer (with softmax and cross entropy) 
        if self.activation == "SOFTMAX":
            self.nodeValues = calculateOutputLayerNodeValuesJIT(self.nodeValues, self.output, expected_output)
            # for nodeValueIdx in range(len(self.nodeValues)):
            #     sum = 0
            #     for j in range(self.output.shape[0]):
            #         if nodeValueIdx == j:
            #             sum -= (1-self.output[nodeValueIdx])*(expected_output[nodeValueIdx])
            #         else:
            #             sum -= -self.output[nodeValueIdx] *(expected_output[j])
            #     self.nodeValues[nodeValueIdx] = sum
        
        return self.nodeValues


    #this function calculates the node values for the hidden layers, based on the activation function of each.
    #First, it calculates the activation function derivatives, and then it  uses them to caluclate the node values, based on the backpropagation method
    def calculateHiddenLayerNodeValues(self, oldLayer, oldNodeValues):

        if self.activation == "SIGMOID":
            activationDerivatives = np.copy(activations.sigmoid_derivative(self.output))  
        elif self.activation == "RELU":
            activationDerivatives = np.copy(activations.relu_derivative(self.output))

        self.nodeValues = calculaeteHiddenLayerNodeValuesJIT(oldLayer.weights, oldNodeValues, activationDerivatives)
        #self.nodeValues = np.dot((oldLayer.weights), oldNodeValues) * activationDerivatives
        return self.nodeValues


    #this function calculates the output of the layer, by taking the inputs from the previous layer
    def forward(self, inputs):
        self.inputs = inputs
        self.weightedInputs = np.zeros(self.n_neurons)

        #calculate the weighted inputs with dot product
        self.weightedInputs = self.biases + np.dot(inputs, self.weights)

        #pass the weighted input into activation function to get output values
        if self.activation == "SIGMOID":
            self.output = activations.sigmoid(self.weightedInputs)
        elif self.activation == "RELU":
            self.output = activations.relu(self.weightedInputs)
        elif self.activation == "SOFTMAX":
            self.output = activations.softmax(self.weightedInputs)
        return self.output