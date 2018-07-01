from __future__ import print_function

from keras.models import Model
from keras.utils import np_utils
import numpy as np
# from util import *

import matplotlib.pyplot as plt

import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

NUM_OBSERVATIONS = 100

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def create_model():
    x = keras.layers.Input((NUM_OBSERVATIONS,1,1))
    # drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    # drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    # drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(4, activation='softmax')(full)

    model = Model(input=x, output=out)
    return model

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model

fname = 'easy'

x_train, y_train = readucr(fname+'/'+fname+'_train.csv')
x_test, y_test = readucr(fname+'/'+fname+'_test.csv')
nb_classes = len(np.unique(y_test))
batch_size = min(x_train.shape[0]/10, 16)

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)

print(np.unique(y_test))

for i, instance in enumerate(x_test):
    if y_test[i] == 2:
        plt.plot(instance)
plt.show()


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

x_train_mean = x_train.mean()
x_train_std = x_train.std()
print(x_train_mean, x_train_std)
x_train = (x_train - x_train_mean)/(x_train_std)

x_test = (x_test - x_train_mean)/(x_train_std)
x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))

model = load_trained_model(fname+'/'+fname+'.hdf5')
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
y = model.evaluate(x_test, Y_test)
# logger.info('categorical_crossentropy=%f, accuracy=%f' % (scores[0],scores[1]))

print(y)
print(Y_test)