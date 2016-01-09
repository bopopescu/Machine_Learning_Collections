"""
This program is for identifying the handwriting based on    
    images using Convolution Neural Network. It was run
    on Spark using Docker.

The image data file mnist.p.gz is saved in work directory.

#######################
Execute Spark on Docker, the docker envrionment is CentOS
    docker pull sequenceiq/spark:1.5.1
    sudo docker run -it sequenceiq/spark:1.5.1 bash

Test Spark on Docker
    bash-4.1# cd /usr/local/spark
    bash-4.1# cp conf/spark-env.sh.template conf/spark-env.sh
    # nano conf/spark-env.sh

    bash-4.1# ./sbin/start-master.sh
    bash-4.1# ./sbin/start-slave.sh spark:172.17.0.109:7077 # the ip is the spark master ip

Turn On Browser for your current IP:8080
    # run spark pi.py for a test
    bash-4.1# ./bin/spark-submit examples/src/main/python/pi.py

    15/11/05 02:11:23 INFO scheduler.DAGScheduler: Job 0 finished: reduce at /usr/local/spark-1.5.1-bin-hadoop2.6/examples/src/main/python/pi.py:39, took 1.095643 s

    Pi is roughly 3.148900

    bash-4.1# ./sbin/stop-all.sh

Test run this CNN network
    ../bin/spark-submit convolution_neural_network_on_Spark_for_OCR2.py --master local[*] --driver-memory 3g
        # where local[30] splits tasks into 30 local nodes/executors. This is used in local mode
        # local[*] uses up all possible logical cores
    # No hungup run
    nohup /bin/sh -c '../bin/spark-submit convolution_neural_network_on_Spark_for_OCR2.py --master local[*] --driver-memory 3g' &

To use the Spark MLib Dataframe features, inputs must be reshaped into one-dimensional
"""


from __future__ import absolute_import
from __future__ import print_function
import numpy as np, os
np.random.seed(31415)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.utils import np_utils

from keras.regularizers import l2, activity_l2
from sys import platform as _platform

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from elephas.ml_model import ElephasEstimator
from elephas.ml.adapter import to_data_frame
from elephas import optimizers as elephas_optimizers

from pyspark.mllib.evaluation import MulticlassMetrics

import gzip
import cPickle

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

APP_NAME = "mnist"
MASTER_IP = 'local[24]' #No need to set it here, can be set outside of this program file

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
num_workers = 4 # workers in the spark cluster
batch_size = 128
# number of rounds of trainings in model.fit
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Load data
f = gzip.open("./test_data/mnist.p.gz", "rb")
mnistdict = cPickle.load(f)
(X_train, y_train), (X_test, y_test) = mnistdict[mnistdict.keys()[0]]
f.close()
nb_classes = len(np.unique(y_train)) #nb_classes = 10 # number of y-lables, 10 digits

# added 1 in the second place for grayscale image, Use 3 for RGB image
    # Since it will be used in dataframe below
    # it will have to convert all dimensions to one dimension
#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.reshape(X_train.shape[0], 1 * img_rows * img_cols)
X_test = X_test.reshape(X_test.shape[0], 1 * img_rows *img_cols)

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

inputshape = X_train.shape[1:]

# Model Training
# Check out the http://keras.io/layers/convolutional/ 
#grapher = Graph() #for plotting the network
model = Sequential()
model.add(Convolution1D(nb_filters, nb_conv*nb_conv))
model.add(Activation('relu'))  # relu is the activation function, Rectified Linear Unit (ReLU)
model.add(Convolution1D(nb_filters, nb_conv*nb_conv))
model.add(Activation('relu'))
#the maxpool is used in some bleeding-edge Theano version
model.add(MaxPooling1D(nb_pool * nb_pool))
model.add(Dropout(0.25)) #dropout (of neurons) feature for regularization

model.add(Convolution1D(nb_filters, nb_conv*nb_conv))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filters, nb_conv*nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling1D(nb_pool * nb_pool))
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

## spark
conf = SparkConf().setAppName(APP_NAME) #.setMaster(MASTER_IP)
sc = SparkContext(conf=conf)

# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(sc,model)

# Build RDD from numpy features and labels
df = to_data_frame(sc, X_train, Y_train, categorical=True)
test_df = to_data_frame(sc, X_test, Y_test, categorical=True)

# Initialize Spark ML Estimator
adadelta = elephas_optimizers.Adadelta()
estimator = ElephasEstimator(sc,
                             model,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             optimizer=adadelta,
                             frequency='batch',
                             mode='asynchronous',
                             num_workers=2,
                             verbose=0,
                             validation_split=0.15,
                             categorical=True,
                             nb_classes=nb_classes)

# Fitting a model returns a Transformer
fitted_model = estimator.fit(df)

# Evaluate Spark model by evaluating the underlying model
prediction = fitted_model.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)

prediction_and_label = pnl.map(lambda row: (row.label, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print(metrics.precision())
print(metrics.recall())
