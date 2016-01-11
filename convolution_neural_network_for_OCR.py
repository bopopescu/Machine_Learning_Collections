"""
This program is for identifying the handwriting based on    
    images using Convolution Neural Network.

The CNN network so far gives an error rate of 0.83 percent on testing data

http://deeplearning.net/tutorial/gettingstarted.html
https://www.microway.com/hpc-tech-tips/keras-theano-deep-learning-frameworks/
https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

The MNIST dataset consists of handwritten digit images and it 
    is divided in 60,000 examples for the training set and 10,000 
    examples for testing. In many papers as well as in this tutorial, 
    the official training set of 60,000 is divided into an actual training 
    set of 50,000 examples and 10,000 validation examples (for selecting 
    hyper-parameters like learning rate and size of the model). All digit 
    images have been size-normalized and centered in a fixed size image of 
    28 x 28 pixels. In the original dataset each pixel of the image is 
    represented by a value between 0 and 255, where 0 is black, 255 is 
    white and anything in between is a different shade of grey.

For convenience we pickled the dataset to make it easier to use in python. 
    It is available for download here. The pickled file represents a tuple of 
    3 lists : the training set, the validation set and the testing set. Each of 
    the three lists is a pair formed from a list of images and a list of class 
    labels for each of the images. An image is represented as numpy 1-dimensional 
    array of 784 (28 x 28) float values between 0 and 1 (0 stands for black, 
    1 for white). The labels are numbers between 0 and 9 indicating which 
    digit the image represents. The code block below shows how to load the dataset.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np, os
np.random.seed(31415)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.regularizers import l2, activity_l2
from sys import platform as _platform

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import gzip
import cPickle

APP_NAME = "mnist"
if _platform == "linux" or _platform == "linux2":
    # linux
    ossys = 'linux'
elif _platform == "darwin":
    # OS X
    ossys = 'mac'
elif _platform == "win32":
    # Windows...
    ossys = 'windows'

# Define basic parameters
batch_size = 128
# number of rounds of trainings in model.fit
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

"""
# The following is to save MNIST data into a pickle file
import pickle
(X_train, y_train), (X_test, y_test) = mnist.load_data()
data = {'MNIST_OCR_Data_(xtrain,ytrain)_(xtest,ytest)': ((X_train, y_train), (X_test, y_test))}
pickle.dump(data, open("mnist.p", "wb" ))
"""

# Load data
f = gzip.open("./test_data/mnist.p.gz", "rb")
mnistdict = cPickle.load(f)
(X_train, y_train), (X_test, y_test) = mnistdict[mnistdict.keys()[0]]
f.close()
nb_classes = len(np.unique(y_train)) #nb_classes = 10 # number of y-lables, 10 digits

# added 1 in the second place for grayscale image, Use 3 for RGB image
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0

if ossys != 'linux':
    # Plot 10 sample graphs
    for i in range(10):
        xd_2 = X_train[i,0,:,:]
        imgplot = plt.imshow(np.array(xd_2), interpolation='nearest', cmap=cm.binary)
        plt.show()
        plt.close()

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
    # for example, label=8 gives [0,0,0,0,0,0,0,1,0,0]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model Training
# Check out the http://keras.io/layers/convolutional/ 
#grapher = Graph() #for plotting the network
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))  # relu is the activation function, Rectified Linear Unit (ReLU)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
#the maxpool is used in some bleeding-edge Theano version
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25)) #dropout (of neurons) feature for regularization

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
# three convolutional layers chained together -- might work for larger images
# full, valid, maxpool then full, valid, maxpool
n_neurons = nb_filters * (img_rows/nb_pool/nb_pool) * (img_cols/nb_pool/nb_pool)

print("There are %i neurons." %n_neurons)
model.add(Dense(128))  # flattens n_neuron connections per neuron in the fully connected layer
                                  # here, the fully connected layer is defined to have 128 neurons
                                  # therefore all n_neurons inputs from the previous layer connecting to each
                                  # of these fully connected neurons (FC layer), reduces it's input to a single
                                  # output signal.  Here's the activation function is given be ReLU.  
model.add(Activation('relu'))
model.add(Dropout(0.5))           # dropout is then applied 

# finally the 128 outputs of the previous FC layer are fully connected to num_classes of neurons, which 
# is activated by a softmax function
model.add( Dense(nb_classes, W_regularizer=l2(0.01) ))
model.add( Activation('softmax') )
# write the neural network model representation to a png image
#grapher.plot(model, 'nn_mnist.png')

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
# model.compile(loss='categorical_crossentropy', optimizer='sgd' or 'adam or 'adadelta')

# Details for using Models http://keras.io/models/
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,
            #alidation_data=(X_test, Y_test),
            show_accuracy=True, verbose=1,
            validation_split=0.15)

# Evaluate Spark model by evaluating the underlying model using Test/Training Data
score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=2)
print('Training accuracy:', score[1])

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])

"""
Train on 60000 samples, validate on 10000 samples
Epoch 0
60000/60000 [==============================] - 79s - loss: 0.2596 - acc: 0.9200 - val_loss: 0.0548 - val_acc: 0.9826
Epoch 1
60000/60000 [==============================] - 79s - loss: 0.0961 - acc: 0.9713 - val_loss: 0.0441 - val_acc: 0.9861
Epoch 2
60000/60000 [==============================] - 79s - loss: 0.0735 - acc: 0.9782 - val_loss: 0.0426 - val_acc: 0.9860
Epoch 3
60000/60000 [==============================] - 79s - loss: 0.0617 - acc: 0.9816 - val_loss: 0.0330 - val_acc: 0.9885
"""
