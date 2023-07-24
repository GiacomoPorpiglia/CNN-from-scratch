# Convolutional Neural Network for Image Classification From Scratch (Only NumPy and SciPy)

Hello everyone! The project's goal was to write a Neural Network from scratch, without the help of any libraries like PyTorch, Keras, TensorFlow ecc... <br />
But why bother, you may ask. The main reason is that, since it was my first approach to neural networks, I didn't want to fast-forward using libraries(PyTorch, TensorFlow, Keras, ecc...) straightaway, becaues I think that in roder to have a deep and full understanding of how something works, you have to do it the hard way, first. <br />
So, to give you the general idea of what this project is, it is a convolutional neural network for the classification of the MNIST hand-written digits dataset (Of course, with some small changes, it can be used for classifying many different datasets).
The model I used is LeNet-5, a very popular model for this problem ([this is a great article](https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/) if you want to learn more), with some small tweaks I'll explain later on.
<br />
LeNet is made of:  
  * a convolution layer with 6 kernels (C1)  
  * a pooling layer (I used the mean pool, but I read that max pool gives, for this dataset, very similar results) (C2)  
  * a convolution layer with 16 kernels (the input of each kernel are different combinations of the 6 outputs of the previous layer) (C3)  
  * a pooling layer, like the previous one (C4)  
  * a fully connected layer with 120 neurons (with 16x4x4=256 inputs, obtained by flattening the 16 outputs of C4)  
  * a fully connected layer with 84 neurons  
  * an output layer with 10 outputs, one for each digit  

The **activation function** is **sigmoid**, and for the output layer is **Softmax**, paired with **cross-entropy loss**.  

The only difference with the original LeNet is the number of inputs for the FC layers: in fact, since the size of each image was originally 32x32, the flattened output of C4 had size=16x5x5=400. No other changes were made from the original model.

For the optimization, I experimented multiple approaches:  
  * Gradient Descent, which has the downside of getting stuck very easily at local minima.  
  * Gradient Descent with Momentum, which helps avoiding local minima, but still has some problems in improving the model when it gets to around 95% accuracy. This is because the changes at that point of the training have to be small, and with momenutm is very tricky to get that. I tried lowering the learning rate as the training went on, but with small results.  
  * The real game changer (at least for me), was **Adam optimization**: a widely popular optimizer, which combines momentum and RMSprop. More about it [here](https://optimization.cbe.cornell.edu/index.php?title=Adam)  

## Training

Now let's talk about how I trained the model. Now, my goal wasn't just getting a very high accuracy on the dataset: I wanted the network to **be able to generalize** so that it could recognize efficiently REAL hand-written digits that the user can write on a simple drawing canvas.  
Now, someone may think that, if the model has a very high training accuracy, it will also generalize very well on any given image, but I found that wasn't the case.
In fact I experimented that every number in the dataset is centered and arranged in a specific way (so they fit in a 20x20 box in the center of the image). So, when I drew my own numbers, I often got wrong answers, despite the accuracy being > 97%.  <br />
What I did to avoid this was **distorcing the train images**, giving them some randomness: I shifted them, zoomed them, and rotated them, to help convergence.
The results I got (with a learn rate of 0.01 for a total of 20 epochs) were much better than I hoped:  
  - A **98.55% accuracy** on the training data (undistorted), more exactly 59131 right, 869 wrong
  - A **98.50% accuracy** on the test data, 9850 right, 150 wrong.
    - On this result, I also want to point out something very interesting. As you can read in the [official MNIST dataset website](http://yann.lecun.com/exdb/mnist/), the first 5000 images of the test dataset are supposed to be simpler than the last 5000.
    Instead, I got a 97.88% accuracy on the "easy" ones, and a **99.12% accuracy** on the hard ones!
    I personally don't have an explanation on why is this, but if you have let me know!

![training graph](https://github.com/GiacomoPorpiglia/CNN-from-scratch/blob/master/images/train_graph_98%2C5%25.png)


All the data of the training are in 3 files located in the folder "saved_network_98.5%" (they are "trainData.txt", "testData.txt", and "costData.txt")
As you may notice, the cost seems pretty high for such accuracy, but keep in mind the training was done on the distorted images: in fact, the accuracy on the training images during the training was only around 95%.

## How to use


You can choose between 4 modes: train, test, viewtest and selftest. </br>
With <b>train</b> you can train a new model. Here is an example of execution: </br>
```
python main.py --mode train --path /path/to/network/folder --epochs <numberOfEpochs(default: 20)>
```
with <b>test</b> you can test an existing model with a batch of images and get the accuracy of the model.</br>
```
python main.py --mode test --path /path/to/network/folder
```
With <b>viewtest</b> you can view the images the model is computing, associated with the model answer and confidence relative to it.</br>
```
python main.py --mode viewtest --path /path/to/network/folder
```
With <b>selftest</b> you can draw numbers yourself and feed them to the network, and see if it can recognize what they are!
```
python main.py --mode selftest --path /path/to/network/folder
```

=======
To use the project, you can easily clone the repo or download it.
To install the required dependecies, go to the project folder, open a command prompt and run the command 
```
pip install -r requirements.txt
```
and then run the **main.py** file. <br />

Also, if you'd like to play around using the pre-trained model, you will need to set the --path to the folder "saved_network_98.5%", where the network is saved. <br />

Take a look at the selftest mode: <br /> <br />
<img src="/images/selftest.gif" width="80%" />
<br /> <br />

To change the settings, like learn-rate or layer dimensions, they are all stored in the settings file, in the Settings folder. <br />

It's easier done than said, so I encourage you to try it on your own and have fun with it!
The model is saved in .npy files, each containing kernels/weights/biases for each layer.


## An open issue: GPU optimization

Do you know what's the main problem with such things without any library? GPU optimization!
In fact, this program runs on the CPU, and to train it **for 20 epochs, it took roughly 2 hours!**
I know this is not optimal, but speed wasn't the goal of this project, anyway!
I know there are some ways (like CUDA) to make array computations run on the GPU, but it would take me very long to rewrite the code.
Anyway, I hope you like this project as much as I have enjoyed making it, and let me know what you think!
