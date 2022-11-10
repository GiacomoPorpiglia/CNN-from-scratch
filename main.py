import numpy as np
import random
from NeuralNetwork import NeuralNetwork
from draw import createDrawCanvas
from loadSamples import *
from randomizeImage import *

def train(network, image_size, images_set, labels_set, counter, mode, learnRate):
    batch_size = images_set.shape[0]
    images_set = images_set.reshape(batch_size, image_size, image_size)

    #add randomness for better generalization
    for image_idx, image in enumerate(images_set):
        images_set[image_idx] = rotate_image(image)
        images_set[image_idx] = zoom_image(images_set[image_idx], random.uniform(0.9, 1.1))
        images_set[image_idx] = translate_image(images_set[image_idx], random.randint(-1, 1), random.randint(-1, 1))

    # each image now is represented as a 1d array (so if the image-size is 28, the array will have lenght of 28*28, so 784 )
    images_set = images_set.reshape(batch_size, image_size*image_size)

    #add noise and normalize the values, so pass from a range [0, 255] to [0, 1]
    for image_idx, image in enumerate(images_set):
        images_set[image_idx] = clear_and_normalize(images_set[image_idx])
        #images_set[image_idx] = add_noise(images_set[image_idx])
    images_set = images_set.reshape(batch_size, image_size, image_size)

    #run the data through the network
    network.run(mode, images_set, labels_set, learnRate)
   
def test(network, image_size, images_set, labels_set, mode):
    batch_size = images_set.shape[0]

    images_set = images_set.reshape(batch_size, image_size*image_size)
    
    for image_idx, image in enumerate(images_set):
        images_set[image_idx] = clear_and_normalize(images_set[image_idx])

    #run the image through the network  
    network.run(mode, images_set, labels_set)

def viewtest(network, image_size, images_set, labels_set, mode):
    batch_size = 1

    images_set = images_set.reshape(batch_size, image_size*image_size)

    for image_idx, image in enumerate(images_set):
        images_set[image_idx] = clear_and_normalize(images_set[image_idx])

    #run the image through the network  
    network.run(mode, images_set, labels_set)


def main():

    mode = input("Type 'train' if you want to train your model\nType 'test' if you want to run a test (Note: To run the test, you must have trained the network before)\nType 'viewtest' if you want to see the image and the output of the network\nType 'selftest' if you want to draw the digits yourself\n")
    
    #if the user diesn't write one of the 3 modes, the program will just stop
    if mode != "train" and mode != "test" and mode != "viewtest" and mode != "selftest":
        return
    
    networkToLoadPath = input("Do you want to load an existing network? If so, type the path of the folder where the network is (relative or absolute). If the network isn't found in that path, no network will be loaded.\nPath: ")

    if mode == "train":

        runTestWhileTraining = input("Do you want to run the network on the test data while training, to get infos about the test accuracy %? [Y/n]: ")
        if runTestWhileTraining == "Y" or runTestWhileTraining == "y":
            runTestWhileTraining = True
        else:
            runTestWhileTraining = False

    #--------------------Neural Network Loop--------------------------

    network = NeuralNetwork([256, 120, 84, 10], [[6, 5], [16, 5]], networkToLoadPath)

    image_size = 28 #28x28 pixels
    batchCounter = 0

    learnRate = 0.01
    
    if mode == 'train':
        batch_size = 128
        test_batch_size = 2000
        trainImages = loadImages('train')
        trainLabels = loadLabels('train')
        if runTestWhileTraining:
            testImages = loadImages('test')
            testLabels = loadLabels('test')
        while True:
            batchCounter+=1
            epochProgress = batchCounter*batch_size/60000
            print(f"Epoch number {int(epochProgress)+1}, Progress: {round((epochProgress-int(epochProgress))*100, 3)}%", end="\r")

            images_train_set, labels_train_set = selectImagesAndLabels(batch_size, trainImages, trainLabels)
            
            train(network, image_size, images_train_set, labels_train_set, batchCounter, 'train', learnRate)
            
            #every half epoch, run test and get results, and write train ad test accuracy + cost average to file
            if runTestWhileTraining and batchCounter % int(30000/batch_size) == 0:
                images_test_set, labels_test_set = selectImagesAndLabels(test_batch_size, testImages, testLabels)
                test(network, image_size, images_test_set, labels_test_set, 'test')

                #WRITE THE TRAINING DATA TO FILE
                try:
                    with open(networkToLoadPath + '/testData.txt', 'a') as f:
                        f.write(str(round((network.testRightAnswers/test_batch_size)*100, 3)) + ', ')
                    with open(networkToLoadPath + '/trainData.txt', 'a') as f:
                        f.write(str(round((network.rightAnswers/(network.rightAnswers+network.wrongAnswers))*100, 3)) + ', ')
                    with open(networkToLoadPath + '/costData.txt', 'a') as f:
                        f.write(str(round((network.costSum/(network.rightAnswers+network.wrongAnswers)), 3)) + ', ')
                except:
                    pass
                network.rightAnswers  = network.wrongAnswers = network.costSum = 0
                network.save()


    elif mode == 'test':
        testImages = loadImages('test')
        testLabels = loadLabels('test')
        batch_size = 10000
        while True:
            images_test_set, labels_test_set = selectImagesAndLabels(batch_size, testImages, testLabels)
            test(network, image_size, images_test_set, labels_test_set, 'test')
            print("Accuracy:", network.testRightAnswers/(batch_size) *100, "%")
            print(network.testRightAnswers, network.testWrongAnswers)
            network.testRightAnswers = network.testWrongAnswers = 0
    elif mode == 'viewtest':
        testImages = loadImages('test')
        testLabels = loadLabels('test')
        while True:
            images_test_set, labels_test_set = selectImagesAndLabels(1, testImages, testLabels)
            viewtest(network, image_size, images_test_set, labels_test_set, 'viewtest')
    elif mode == 'selftest':
        #with selftest mode, a canvas will open for the user to draw the number, then by pressing enter the data will be passed thorough the network, and its answer will be calculated
            createDrawCanvas(network)



if __name__=="__main__":
    main()
