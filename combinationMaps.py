#This file contains combinationMaps, meaning the arrays describing, for the convolutional layers, to which kernels of the next layer they have to forward their output

#The reversed combination maps contain the values that are not contained in che combination map.
#For instance, the first convolutional layer has as its only input the original image, meaning the input with index [0], as said in the reversedCombinationMap0. Similarly, the first kernel of the second layer will take inputs from the outputs [0, 1, 2] of the first layer.

reversedCombinationMap0 = [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]


combinationMap0 = [
            [],
            [],
            [],
            [],
            [],
            [],
        ]


reversedCombinationMap1 = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 0],
            [5, 0, 1],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 0],
            [4, 5, 0, 1],
            [5, 0, 1, 2],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [2, 3, 5, 0],
            [0, 1, 2, 3, 4, 5]
        ]


combinationMap1 = [
            [3, 4, 5],
            [4, 5, 0],
            [5, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [4, 5],
            [5, 0],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [5, 2],
            [0, 3],
            [1, 4],
            [],       
        ]

