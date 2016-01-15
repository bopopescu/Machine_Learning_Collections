# -*- coding: utf-8 -*-
"""
@author: pm

This is a MNIST Digital Handwriting example in Python by Deep Belief Network.

install nolearn package for deep belief network
$ pip install nolearn

Package usage refers to the source codes
https://github.com/danielfrg/copper/blob/master/copper/ml/gdbn/gdbn.py
"""

# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np, gzip, cPickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import cv2  #from openCV
except ImportError:
    print("No cv2 package is avaible...")

# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
# print "[X] downloading data..."
# dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training
# and testing splits


f = gzip.open("./test_data/mnist.p.gz", "rb")
mnistdict = cPickle.load(f)
(trainX, trainY), (testX, testY) = mnistdict[mnistdict.keys()[0]]
f.close()

img_rows, img_cols = 28, 28

trainX = trainX.reshape(trainX.shape[0], img_rows * img_cols)
testX = testX.reshape(testX.shape[0], img_rows*img_cols)

trainX = trainX.astype("float32")
testX = testX.astype("float32")
trainX /= 255.0
testX /= 255.0

# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)

"""
    help(DBN)
 |  Methods defined here:
 |  
 |  __init__(self, layer_sizes=None, scales=0.05, fan_outs=None, output_act_funct=None, 
    real_valued_vis=True, use_re_lu=True, uniforms=False, learn_rates=0.1, 
    learn_rate_decays=1.0, learn_rate_minimums=0.0, momentum=0.9, l2_costs=0.0001, 
    dropouts=0, nesterov=True, nest_compare=True, rms_lims=None, 
    learn_rates_pretrain=None, momentum_pretrain=None, 
    l2_costs_pretrain=None, nest_compare_pretrain=None, 
    epochs=10, epochs_pretrain=0, loss_funct=None, minibatch_size=64,
     minibatches_per_epoch=None, pretrain_callback=None, fine_tune_callback=None, 
     random_state=None, verbose=0)

     :param output_act_funct: Output activation function.  Instance
                                 of type
                                 :class:`~gdbn.activationFunctions.Sigmoid`,
                                 :class:`~.gdbn.activationFunctions.Linear`,
                                 :class:`~.gdbn.activationFunctions.Softmax`
                                 from the
                                 :mod:`gdbn.activationFunctions`
                                 module.  Defaults to
                                 :class:`~.gdbn.activationFunctions.Softmax`.
"""

dbn = DBN(
    layer_sizes = [trainX.shape[1], 800, 10], # trainX.shape[1] is the input layer, 10 is output layer
                                            # 300 is the hidden layer
    output_act_funct = "Softmax",
    dropouts = 0.0,
    use_re_lu=True,
    l2_costs=0.0001,
    learn_rates = 0.3,
    learn_rate_decays = 0.9,
    epochs = 10,
    loss_funct = None, # if not specified, default is the count of percentage or wrong labels, built in function
    verbose = 1)


##### Below is the trick for changing score function to evaluate the accuracy. The original program
    # does not have other options except for pure compare % of accurate outputs,
    # here one may create a function of his/her own.
import new

def _score(self, X, y):
    outputs = self.predict_proba(X)
    targets = self._onehot(y)  # This is a built-in function to ensure results are like 1,2,3,... as numerical
    mistakes = np.sum(np.not_equal(targets, outputs))
    #return - float(mistakes) / len(y) + 1
    return  1 - 1.0*mistakes/len(y)

dbn.score = new.instancemethod(_score, dbn, dbn.__class__) #update the score function
###########
dbn.fit(trainX, trainY)

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print classification_report(testY, preds)
print 'The accuracy on testing data is:', accuracy_score(testY, preds)

# randomly select a few of the test instances
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
    # classify the digit
    pred = dbn.predict(np.atleast_2d(testX[i]))
    # reshape the feature vector to be a 28x28 pixel image, then change
    # the data type to be an unsigned 8-bit integer
    image = (testX[i] * 255).reshape((img_rows, img_cols)).astype("uint8")
    # show the image and prediction
    print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
    plt.imshow(image, interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()

