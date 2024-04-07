import numpy as np
import random
from NeuralNetwork import NeuralNetwork
from drawCanvas import drawCanvas
from loadSamples import *
from randomizeImage import *
from Settings.settings import *
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Required ...... Specify the mode: You can choose between train, test, selftest and viewtest.", required=True,action='store')
parser.add_argument('--path', help="Required ...... Specify the path you want to load an existing network from, or in case of train where you want the new network to be stored.", required=True, action='store')
parser.add_argument('--epochs', help="Optional ...... Default set to 20, specifies the number of epochs you want to train the model for.", action='store')
args = parser.parse_args()


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
    print("Running test...")
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


def main(learnRate, batch_size, LDNSize, CNNSize):

    mode = args.mode
    #if the user doesn't write one of the 4 modes, the program will just stop
    if mode != "train" and mode != "test" and mode != "viewtest" and mode != "selftest":
        print("Please make sure you type one of the above options. Please try again.")
        return
    
    networkToLoadPath = args.path

    if networkToLoadPath=="" and mode != "train":
        print("No path was specified. Initializing a random, untrained network...")
    elif networkToLoadPath=="" and mode == "train":
        print("You must specify the path were you want the network to be saved. Try again.")
        return
    
    network_file_path = Path(networkToLoadPath)
    if not network_file_path.exists():
        print("The specified path doesn't exist. Make sure you typed in the correct path.")
        return
    #--------------------Neural Network Loop--------------------------

    network = NeuralNetwork(LDNSize, CNNSize, networkToLoadPath)

    image_size = 28 #28x28 pixels
    batchCounter = 0
    if mode == 'train':
        trainImages = loadImages('train', image_size)
        trainLabels = loadLabels('train')

        test_batch_size = 10000
        testImages = loadImages('test', image_size)
        testLabels = loadLabels('test')
        maxAccuracy = 0

        maxEpochs = 20 if args.epochs is None else int(args.epochs)
        currentEpoch = 0
        while currentEpoch < maxEpochs:
            batchCounter+=1
            epochProgress = batchCounter*batch_size/60000

            print(f"Epoch number {int(epochProgress)+1}, Progress: {round((epochProgress-int(epochProgress))*100, 2):.2f}%", end="\r")

            images_train_set, labels_train_set = selectImagesAndLabels(batch_size, image_size, trainImages, trainLabels)
            train(network, image_size, images_train_set, labels_train_set, batchCounter, 'train', learnRate)
            
            #every epoch, run test and get results, and write train, test accuracy and cost average to file
            if epochProgress-int(epochProgress) + batch_size/60000 >= 1:
                print(f"\nEpoch {int(currentEpoch)+1} completed")
                currentEpoch+=1
                images_test_set, labels_test_set = selectImagesAndLabels(test_batch_size, image_size, testImages, testLabels)
                test(network, image_size, images_test_set, labels_test_set, 'test')

                trainAccuracy = round((network.rightAnswers/(network.rightAnswers+network.wrongAnswers))*100, 3)
                testAccuracy = round((network.testRightAnswers/test_batch_size)*100, 3)
                costAverage = round((network.costSum/(network.rightAnswers+network.wrongAnswers)), 3)
                #WRITE THE TRAINING DATA TO FILE
                try:
                    with open(networkToLoadPath + '/testData.txt', 'a') as f:
                        f.write(str(testAccuracy) + ', ')
                    with open(networkToLoadPath + '/trainData.txt', 'a') as f:
                        f.write(str(trainAccuracy) + ', ')
                    with open(networkToLoadPath + '/costData.txt', 'a') as f:
                        f.write(str(costAverage) + ', ')
                except:
                    pass
                network.rightAnswers  = network.wrongAnswers = network.costSum = 0
                if testAccuracy > maxAccuracy:
                    maxAccuracy = testAccuracy
                    network.save()
                    print(f"Network saved at epoch {int(epochProgress)+1} with test accuracy: {testAccuracy}")
                else:
                    print("\n")
                


    elif mode == 'test':
        testImages = loadImages('test', image_size)
        testLabels = loadLabels('test')
        batch_size = 10000
        while True:
            images_test_set, labels_test_set = selectImagesAndLabels(batch_size, image_size, testImages, testLabels)
            test(network, image_size, images_test_set, labels_test_set, 'test')
            print("Accuracy:", network.testRightAnswers/(batch_size) *100, "%")
            print("Right answers: ", network.testRightAnswers, "  Wrong answers: ", network.testWrongAnswers)
            network.testRightAnswers = network.testWrongAnswers = 0
    elif mode == 'viewtest':
        testImages = loadImages('test', image_size)
        testLabels = loadLabels('test')
        while True:
            images_test_set, labels_test_set = selectImagesAndLabels(1, image_size, testImages, testLabels)
            viewtest(network, image_size, images_test_set, labels_test_set, 'viewtest')
    elif mode == 'selftest':
        #with selftest mode, a canvas will open for the user to draw the number, then by pressing enter the data will be passed thorough the network, and its answer will be calculated
        drawCanvas(network)




if __name__=="__main__":
    main(learnRate, batch_size,LDNSize, CNNSize)
