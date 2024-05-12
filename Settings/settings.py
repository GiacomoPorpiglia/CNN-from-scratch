from Optimizers.Adam import Adam
from Optimizers.Momentum import Momentum
learnRate = 0.01
batch_size = 128

num_of_images_in_train_set = 60000
num_of_images_in_test_set  = 10000

LDNSize = [256, 120, 84, 10] #Size of the layer-dense part of the network (the first number is not the size of the first layer, it is the number of inputs)
CNNSize = [[6, 5], [16, 5]] #each array contains [number of kernels, kernel_size]. The first layer is made of 6 kernels, each of 5x5 #NOTE: If you want to change the dimensiions of the layers, you will also need to change the combinations in combinationsMaps.py file
optimizer = Adam #You can change it to Momentum, but I have experienced that Adam woks much better and has a quicker convergence.