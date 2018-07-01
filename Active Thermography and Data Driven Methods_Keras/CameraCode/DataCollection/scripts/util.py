import numpy as np
import struct
import cv2
from constants import *
import pickle as pk
import array
import csv

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

# import rpy2
# import rpy2.robjects.numpy2ri as rpyn
# from rpy2.robjects.packages import importr
# Set up our R namespaces
# R = rpy2.robjects.r
# DTW = importr('dtw')

from scipy.signal import butter, lfilter, freqz



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


############################################################
### Thermistor Data
############################################################
def thermistor_plot_trial(material, trial_num):
    path = os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'teensy_data.pkl')
    data = load_pickle(path)

    plt.plot(data['time'], data['data'])
    plt.show()

def thermistor_plot_all_trial(material):
    wildcard = os.path.join(DATA_PATH, material, 'trial*', 'teensy_data.pkl')
    files = glob(wildcard)

    for f in files:
        data = load_pickle(f)
        plt.plot(data['time'], data['data'])

    plt.ylim([23,50])
    plt.show()





############################################################
### Thermal Camera Data
############################################################
def get_trial_files(material, trial_num):
    print "Reading File List"
    path = os.path.join(DATA_PATH, material, 'trial%d'%trial_num)
    wildcard = os.path.join(path, '*')
    files = glob(wildcard)
    files = [f for f in files if '.bin' in f]
    print "Sorting"
    files = sorted(files, key=lambda x: int(x.split('/')[-1][:-4]))
    print "Done"

    return files

def load_trial_data(material, trial_num, subtract_min, use_min_pixel=False):
    files = get_trial_files(material, trial_num)

    print "Creating buffer"
    timestamp = []
    darr = []
    for i,f in enumerate(files):
        disp_to_term("Loading files %d     " % i)
        t, img = extract_binary(f)
        timestamp.append(t)
        darr.append(img)
    print "\nDone"

    timestamp = np.array(timestamp).astype(float)
    timestamp /= 1000.

    darr = np.array(darr)
    if subtract_min:
        print "Subtracting Min.."

        if not use_min_pixel:
            min_frame = np.min(darr, axis=0)
            for data in darr:
                data -= min_frame
        else:
            min_pixel = np.min(darr)
            darr -= min_pixel

    # minval = np.min(darr)
    # maxval = np.max(darr)
    # print "Load Complete! Min: %.2f; Max: %.2f" % (minval, maxval)

    return timestamp, darr

def extract_binary(filename):
    arr = array.array('H')
    arr.fromfile(open(filename, 'rb'), os.path.getsize(filename)/arr.itemsize)
    return int(filename.split('/')[-1][:-4]), np.array(arr, dtype='float64').reshape((lHeight, lWidth))[:,0::2]

def extract_trial(material, trial_num):
    files = get_trial_files(material, trial_num)

    data = []
    for i, f in enumerate(files):
        timestamp, img = extract_binary(f)
        print i, timestamp
        data.append((timestamp, img))

    times, imgs = zip(*data)

    save_pickle({'time':times, 'data':imgs}, os.path.join(path, 'cam.pkl'))

def play_trial(material, trial_num, step=0.000001, subtract_min=True, jump=50):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    minval = np.min(darr)
    maxval = np.max(darr)

    plt.ion()
    plt.imshow(darr[0], clim=[minval, maxval])
    plt.colorbar()
    for i, img in enumerate(darr):
        if i % jump == 0:
            print np.mean(img)
            plt.imshow(img, clim=[minval, maxval])
            plt.title(timestamp[i])
            plt.pause(step)



def display_binary(fin):
    time, data = extract_binary(fin)
    plt.imshow(data)
    plt.show()

def bin2mat(fin, fout):
    time, data = extract_binary(fin, fout)
    save_pickle(data, fout)




############################################################
### TEENSY
############################################################
fB,fA = butter(2, 0.1, analog=False)

def setup_serial(dev_name, baudrate):
    try:
        serial_dev = serial.Serial(dev_name)
        if(serial_dev is None):
            raise RuntimeError("Serial port %s not found!\n" % (dev_name))

        serial_dev.baudrate = baudrate
        serial_dev.parity = 'N'
        serial_dev.stopbits = 1
        serial_dev.write_timeout = .1
        serial_dev.timeout= 1.

        serial_dev.flushOutput()
        serial_dev.flushInput()
        return serial_dev

    except serial.serialutil.SerialException as e:
        print "Serial port %s not found!\n" % (dev_name)
        return []

def send_string(serial_dev, message):
    try:
        serial_dev.write(message)
        serial_dev.flushOutput()
    except serial.serialutil.SerialException as e:
        print "Error sending string"

def get_adc_data(serial_dev, num_adc_inputs):

    ln = serial_dev.readline()

    if not ln:
        print 'not received'
        return []

    #serial_dev.flushInput()
    try:
        l = map(int, ln.split(','))
    except ValueError:
        serial_dev.flush()
        l = []
    if len(l) != num_adc_inputs:
        serial_dev.flush()
        l = []
    else:
        return l

    # print 'passed'
    return l

def temperature(raw_data,Vsupp,Rref):
    raw_data = np.array(raw_data)
    # Vref = 3.3
    # Vin = raw_data/4095.0*Vref

    T1 = 288.15
    B = 3406
    R1 = 14827
    Vin[Vin <= 0] = .001
    RT = Rref*(raw_data - 1)
    RT[RT <= 0] = .001
    TC = (T1*B/np.log(R1/RT))/(B/np.log(R1/RT) - T1) - 273.15
    return TC.tolist()

def start_heating(temp_dev):
    send_string(temp_dev, 'G')

def stop_heating(temp_dev):
    send_string(temp_dev, 'X')