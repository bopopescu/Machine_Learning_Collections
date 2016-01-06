"""
This program is for identifying the handwriting based on    
    images using Convolution Neural Network. It was run
    on Spark using Docker.

The image data file mnist.pkl.gz is saved in work directory.

#######################
Execute Spark on Docker
    docker pull sequenceiq/spark:1.5.1
    sudo docker run -it sequenceiq/spark:1.5.1 bash

Test Spark on Docker
    bash-4.1# cd /usr/local/spark
    bash-4.1# cp conf/spark-env.sh.template conf/spark-env.sh
    bash-4.1# vi conf/spark-env.sh
    export SPARK_LOCAL_IP=172.17.0.109
    export SPARK_MASTER_IP=172.17.0.109

    bash-4.1# ./sbin/start-master.sh
    bash-4.1# ./sbin/start-slave.sh spark:172.17.0.109:7077

Turn On Browser for your current IP:8080
    # run spark pi.py for a test
    bash-4.1# ./bin/spark-submit examples/src/main/python/pi.py

    15/11/05 02:11:23 INFO scheduler.DAGScheduler: Job 0 finished: reduce at /usr/local/spark-1.5.1-bin-hadoop2.6/examples/src/main/python/pi.py:39, took 1.095643 s

    Pi is roughly 3.148900
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

import gzip
import cPickle

APP_NAME = "mnist"
MASTER_IP = 'local[24]'

# Define basic parameters
batch_size = 128
nb_classes = 10
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Load data
f = gzip.open("./test_data/mnist.pkl.gz", "rb")
dd = cPickle.load(f)
(X_train, y_train), (X_test, y_test) = dd

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

## spark
conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER_IP)
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)

# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(sc,model)

# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, validation_split=0.1, num_workers=24)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.get_network().evaluate(X_test, Y_test, show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])
