# Convolutional Neural Network for Image Classification From Scratch (Only NumPy and SciPy)

Hello everyone! The project's goal was to write a Neural Network from scratch, without the help of any libraries like PyTorch, Keras, TensorFlow ecc... <br />
But why bother, you may ask. The main reason is that, since it was my first approach to neural networks, I didn't want to fast-forward using libraries(PyTorch, TensorFlow, Keras, ecc...) straightaway, becaues I think that in roder to have a deep and full understanding of how something works, you have to do it the hard way, first. <br />
So, to give you the general idea of what this project is, it is a convolutional neural network for the classification of the MNIST hand-written digits dataset (Of course, with some small changes, it can be used for classifying many different datasets).
The model I used is LeNet-5, a very popular model for this problem ([this is a great article](https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/) if you want to learn more), with some small tweaks I'll explain later on.
<br />
LeNet is made of:
    -a convolution layer with 6 kernels (C1)
    -a pooling layer (I used the mean pool, but I read that max pool gives, for this dataset, very similar results) (C2)
    -a convolution layer with 16 kernels (the input of each kernel are different combinations of the 6 outputs of the previous layer) (C3)
    -a pooling layer, like the previous one (C4)
    -a fully connected layer with 120 neurons (with 16x3x3=256 inputs, obtained by flattening the 16 outputs of C4)
    -a fully connected layer with 84 neurons
    -a output layer with 10 outputs, one for each digit

    The activation function is sigmoid, and for the output layer is Softmax, paired with cross-entropy loss.

The only difference with the original LeNet is the number of inputs for the FC layers: in fact, since the size of each image was originally 32x32, the flattened output of C4 had size=16x5x5=400. No other changes were made from the original model.

For the optimization, I experimented multiple approaches:
    -Gradient Descent, which has the downside of getting stuck very easily at local minima.
    -Gradient Descent with Momentum, which helps avoiding local minima, but still has some problems in improving the model when it gets to around 95% accuracy. This is because the changes at that point of the training have to be small, and with momenutm is very tricky to get that. I tried lowering the learning rate as the training went on, but with small results.
    The real game changer (at least for me), was Adam optimization
    -Adam optimization is a widely popular optimizer, which combines momentum and RMSprop. More about it here https://optimization.cbe.cornell.edu/index.php?title=Adam

Now let's talk about how I trained the model. Now, my goal wasn't just getting a very high accuracy on the dataset: I wanted the network to be able to generalize in order to be able to recognize REAL hand-written digits, that the user can write on a simple drawing canvas.
Now, you may think that, if the model has a very high training accuracy, it will also generalize very well on any given image, but I found that wasn't the case.
In fact I found out that every number in the dataset is centered and arranged in a specific way (so they fit in a 20x20 box in the center of the image). So, when I drew my own numbers, I often got wrong answers, despite the accuracy being > 97%.
What I did to avoid this was distorcing the train images, giving them some randomness: I shifted them, zoomed them, rotated them and added some noise, to help convergence.
The results I got are very interesting: 
    -A 98.55% accuracy on the training data (undistorted), more exactly 59131 right, 869 wrong
    -A 98.46% accuracy on the test data, 9846 right, 154 wrong.
        -On this result, I also want to point out something very interesting. As you can read in the official MNIST dataset website http://yann.lecun.com/exdb/mnist/, the first 5000 images of the test dataset are supposed to be simpler than the last 5000.
        Instead, I got a 97.82% accuracy on the "easy" ones, and a 99.1% accuracy on the hard ones!
        I personally don't have an explanation on why is this, but if you have let me know!

All the data of the training are in 3 files located in the folder "saved_network_98.46%" (they are "trainData.txt", "testData.txt", and "costData.txt")
As you may notice, the cost seems pretty high for such accuracy, but keep in mind the training was done on the distorted images: in fact, the accuracy on the training images during the training was only around 95%.


How to use the program

By running the main file, you can choose between several options:
    -train, if you want to train the model. Keep in mind that you have to specify the folder in which you want the model to be saved for future use.
    -test, to see the accuracy of the model on the 10000 test images.
    -viewtest, to see a single image from the test dataset with the network guess.(so you can see where the network fails the most)
    -selftest, to draw your own numbers and test the network with them, which I find very fun!

The model is saved in .npy files, each containing kernels/weights/biases for each layer.

An open issue: GPU optimization
Do you know what's the main problem with doing things without any library? GPU optimization!
In fact, this program runs on the CPU, and to train it for 20 epochs, it took very roughly 2 hours!
I know this is not optimal, but speed wasn't the goal of this project, anyway!
I know there are some ways (like CUDA) to make the program run on the GPU, but it would take me very long to rewrite the code.
Anyway, I hope you like this project as much as I have enjoyde making it, and let me know what you think!
