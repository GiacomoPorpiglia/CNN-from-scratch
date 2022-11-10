import numpy as np
from Layers.LayerDense import LayerDense
from Layers.Conv2D import Conv2D
from Layers.Pool2D import Pool2D
import matplotlib.pyplot as plt
from combinationMaps import *

class NeuralNetwork:

    def __init__(self, layerSizes, convSizes, networkToLoadPath):

        self.denseLayers = []
        self.convLayers = []

        self.networkToLoadPath = networkToLoadPath
        self.layerSizes = layerSizes
        for idx in range(len(layerSizes)-1):
            if idx+1 == len(layerSizes)-1:
                self.denseLayers.append(LayerDense(layerSizes[idx], layerSizes[idx+1], "SOFTMAX")) #Options: SOFTMAX
            else:
                self.denseLayers.append(LayerDense(layerSizes[idx], layerSizes[idx+1], "SIGMOID")) #Options: SIGMOID / RELU 

        self.convSizes = convSizes

        outputSize = 28-convSizes[0][1]+1
        self.convLayers.append(Conv2D(convSizes[0][0], convSizes[0][1], combinationMap0, "SIGMOID", outputSize)) #Options: SIGMOID / RELU 
        self.convLayers.append(Pool2D(convSizes[0][0], outputSize, "MEAN")) #Options: MEAN / MAX

        outputSize = int(outputSize/2 -convSizes[1][1] + 1)
        self.convLayers.append(Conv2D(convSizes[1][0], convSizes[1][1], combinationMap1, "SIGMOID", outputSize)) #Options: SIGMOID / RELU 
        self.convLayers.append(Pool2D(convSizes[1][0], outputSize, "MEAN")) #Options: MEAN / MAX

        self.costSum = 0
        self.rightAnswers = 0
        self.wrongAnswers = 0

        self.testRightAnswers = 0
        self.testWrongAnswers = 0
        self.testAccuracyData = []

        self.trainAccuracyData = []
        self.costData = []

        self.networkToLoadPath = networkToLoadPath

        if self.networkToLoadPath != '\n':
            self.load()


    def calculateOutputs(self, inputs):
        finalOutput = inputs.reshape(1, 28, 28)
        for convLayer in self.convLayers:
            finalOutput = convLayer.forward(finalOutput)
        
        finalOutput = finalOutput.reshape(self.denseLayers[0].n_inputs)

        for layer in self.denseLayers:
            finalOutput = layer.forward(finalOutput)
        return finalOutput


    def cost(self, outputs, expected_output):
        cost = 0
        for idx, outputVal in enumerate(outputs):
            cost += self.nodeCost(outputVal, expected_output[idx])
        return cost


    def nodeCost(self, pred_y, correct_y):
        return -correct_y*np.log(pred_y+1e-8)


    def updateAllGradients(self, data, expected_output):

        outputs = self.calculateOutputs(data)
        if(np.argmax(outputs) == np.argmax(expected_output)):
            self.rightAnswers+=1
        else:
            self.wrongAnswers+=1
              
        self.costSum += self.cost(outputs, expected_output)
        outputLayer = self.denseLayers[-1]
        nodeValues = outputLayer.calculateOutputLayerNodeValues(expected_output)
        
        outputLayer.updateGradients()


        for hiddenLayer_idx in reversed(range(len(self.denseLayers)-1)):
            hiddenLayer = self.denseLayers[hiddenLayer_idx]
            nodeValues = hiddenLayer.calculateHiddenLayerNodeValues(self.denseLayers[hiddenLayer_idx+1], nodeValues)
            hiddenLayer.updateGradients()

        #CNN backpropagation
        inputFCLayerNodeValues = np.dot((self.denseLayers[0].weights), self.denseLayers[0].nodeValues)
        nodeValues = self.convLayers[-1].updateGradients(inputFCLayerNodeValues)
        for idx in range(len(self.convLayers)-2, -1, -1):
            nodeValues = self.convLayers[idx].updateGradients(nodeValues)


    def learn(self, batch_data, expected_outputs, learnRate):
        for data_idx, data in enumerate(batch_data):
            self.updateAllGradients(data, expected_outputs[data_idx])

        #store and print the train accuracy and cost of the learning data to see if the network is improving     
        #-----------------Apply and reset the gradients--------------------

        for layer in self.denseLayers:
            layer.applyGradients(learnRate)
            layer.resetGradients()
        
        for convLayer in self.convLayers:
            if isinstance(convLayer, Conv2D):
                convLayer.applyGradients(learnRate)
                convLayer.resetGradients()


    def test(self, data, expected_outputs):
        self.testRightAnswers = self.testWrongAnswers = 0
        for idx, batch in enumerate(data):
            outputs = self.calculateOutputs(batch)
            if(np.argmax(outputs) == np.argmax(expected_outputs[idx])):
                self.testRightAnswers+=1
            else:
                self.testWrongAnswers+=1


    def viewtest(self, data):
        outputs = self.calculateOutputs(data)
        return outputs
    
    def selftest(self, data):
        outputs = self.calculateOutputs(data)
        answer = np.argmax(outputs)
        accuracy = outputs[answer] * 100
        return answer, accuracy


    def load(self):
        for idx, layer in enumerate(self.denseLayers):
            try:
                layer.weights = np.load(self.networkToLoadPath + '/ff_weights' + str(idx+1) + '.npy')
                layer.biases = np.load(self.networkToLoadPath + '/ff_biases' + str(idx+1) + '.npy')
            except IOError:
                pass

        numOfPoolLayers = 0
        for idx, layer in enumerate(self.convLayers):
            if isinstance(layer, Conv2D):
                try:
                    layer.kernels = np.load(self.networkToLoadPath + '/conv_kernels' + str(idx+1-numOfPoolLayers) + '.npy')
                except IOError:
                    pass
            else:
                numOfPoolLayers += 1
            
    def save(self):      
        for idx, layer in enumerate(self.denseLayers):
            try:
                np.save(self.networkToLoadPath + '/ff_weights' + str(idx+1) + '.npy', layer.weights)
                np.save(self.networkToLoadPath + '/ff_biases' + str(idx+1) + '.npy', layer.biases)
            except IOError:
                pass 
            
        numOfPoolLayers = 0
        for idx, layer in enumerate(self.convLayers):
            
            if isinstance(layer, Conv2D):
                try:
                    np.save(self.networkToLoadPath + '/conv_kernels' + str(idx+1-numOfPoolLayers) + '.npy', layer.kernels)
                except IOError:
                    pass
            else:
                numOfPoolLayers += 1

    def run(self, mode, data, labels, learnRate=0.01):

        expected_outputs = np.zeros((len(data), 10))

        for idx, label in enumerate(labels):
                if label==0:
                    expected_outputs[idx] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                elif label==1:
                    expected_outputs[idx] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                elif label == 2:
                    expected_outputs[idx] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                elif label == 3:
                    expected_outputs[idx] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                elif label== 4:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
                elif label == 5:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                elif label == 6:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                elif label== 7:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                elif label== 8:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                elif label == 9:
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                elif label == -1: #value passed with selftest mode(because we don't know the expected output of the user's drawing)
                    expected_outputs[idx] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0])

        if mode == 'train':
            self.learn(data, expected_outputs, learnRate)
            
        elif mode == 'test':
            self.test(data, expected_outputs)
            
        elif mode == 'viewtest':
            output = self.viewtest(data[0])
            answer = np.argmax(output)
            accuracy = np.max(output) * 100

            print("Answer:", answer, "Accuracy:", accuracy)

            image_plt = data.reshape(1, 28, 28, 1)
            image = np.asarray(image_plt[0] * 255).squeeze()
            plt.title(f"Label: {np.argmax(expected_outputs[0]) if expected_outputs[0, 0] != -1 else 'none'}, answer: {answer}", color= 'green' if answer == np.argmax(expected_outputs[0]) else 'red')
            plt.imshow(image, cmap='Greys_r')
            plt.show()

        elif mode == 'selftest': #if it is drawn by the user, we don't wanna show it with plt, we just want to print the answers (which is already done previously in the test function)
                answer, accuracy = self.selftest(data[0])
                return answer, accuracy



