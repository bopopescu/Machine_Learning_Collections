import numpy as np
import struct
import cv2
# from constants import *
import pickle as pk
import array
import csv
import time

import os
import sys
import serial
from glob import glob
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import time
from scipy.signal import savgol_filter, lfilter, butter
from scipy.interpolate import interp1d

import rpy2
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects.packages import importr
# Set up our R namespaces
R = rpy2.robjects.r
DTW = importr('dtw')

############################################################
### Constants
############################################################
MAX_TIME = 30.0

############################################################
### IO
############################################################
def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print "Pickle file cannot be opened."
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print 'load_pickle failed once, trying again'
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()

def resampling(data, pts=200):
    old_t = np.linspace(0., 30., 1500)
    new_t = np.linspace(0., 30., num=pts)
    new_data = np.interp(new_t, old_t, data)
    return new_data

def DTWDist(query, template):
    query = resampling(query)
    template = resampling(template)
    alignment = R.dtw(query.tolist(), template.tolist(), keep=True)
    dist = alignment.rx('distance')[0][0]
    disp_to_term('%.4E        '%dist)
    return dist

def accuracy_score(l1, l2):
    return sum([(1 if l1[i]==l2[i] else 0) for i in range(len(l1))]) / float(len(l1))

def readucr(filename, t=MAX_TIME):
    data = np.loadtxt(filename, delimiter = ',')
    print data.shape
    Y = data[:,0]
    X = data[:,1:1+int((data.shape[1]-1)*t/MAX_TIME)]
    return X, Y