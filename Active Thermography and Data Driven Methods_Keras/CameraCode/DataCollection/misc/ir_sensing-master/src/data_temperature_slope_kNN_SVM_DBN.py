# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp

import scipy as scp
import scipy.ndimage as ni
from scipy.signal import butter, lfilter, freqz

import pickle

import unittest
import random

import os, os.path

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b,a,data)
    return y

# Define features
# Input: Zt = (time_list, temp_list)
# Output: Fvec = [data] + [slope]
def feature_vector_diff(Zt,i=0):
    # Generating Fvec
    temp_data = np.array(Zt[1][i:]).flatten().tolist()

    # Calculating Slope
    temp_slope = []
    for j in range(np.size(temp_data)):
        if j <= 1 or j >= (np.size(temp_data)-1):
            temp_slope.append(0)
        else:
            temp_slope.append((temp_data[j+1]-temp_data[j-1])/(2*0.005))

    order = 8 # 5?
    fs = 100.0
    cutoff = 2

    # Filter the data
    temp_slope = butter_lowpass_filter(np.array(temp_slope), cutoff, fs, order).tolist()

    return temp_data, temp_slope