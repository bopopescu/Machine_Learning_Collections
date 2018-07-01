import numpy as np
import struct
import cv2
from constants import *
import pickle as pk
import array
import csv
from collections import Counter
from itertools import product

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
from scipy.ndimage.filters import convolve as convolveim

import scipy.stats as st

import rpy2
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects.packages import importr
# Set up our R namespaces
R = rpy2.robjects.r
DTW = importr('dtw')

from scipy.signal import butter, lfilter, freqz

import keras
from keras.models import Model
from keras.utils import np_utils

import itertools


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
### Time Series Tool Kit
############################################################
def normalize(data):
    temp = data.copy()
    if len(set(temp)) <= 1:
        return np.zeros_like(temp)
    temp -= np.min(temp)
    temp /= np.max(temp)
    return temp

def normalize2(data):
    temp = data.copy()
    if len(set(temp)) <= 1:
        return np.zeros_like(temp)
    temp -= np.mean(temp)
    temp /= np.std(temp)
    return temp

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b,a,data)
    return y

def slope(data, timestamp=None, smoothing=False):
    # Calculating Slope
    temp = []
    for j in range(np.size(data)):
        if j <= 1 or j >= (np.size(data)-1):
            temp.append(0.)
        else:
            if timestamp is None:
                temp.append((data[j+1]-data[j-1])/(2.))
            else:
                temp.append((data[j+1]-data[j-1])/(timestamp[j+1]-timestamp[j-1]))

    # Filter the data
    if smoothing:
        temp = butter_lowpass_filter(np.array(temp), cutoff, fs, order)

    return temp

def moving_average(a, n=3):
    ret = np.convolve(a, [1./n]*n, 'same')
    return ret

def padded_moving_average(a, n=3):
    if n % 2 == 0:
        n += 1
    a = np.array([a[0]]*((n-1)/2) + a.tolist() + [a[-1]]*((n-1)/2))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def inverse_model_function(timestamp, data):
    ret = []

    for i,d in enumerate(data):
        x = data[i]
        if timestamp[i] < HEAT_INTERVAL:
            ret.append(x**2.)
        else:
            ret.append( (x**2. + HEAT_INTERVAL)**2 / (4. * x**2.) )

    return np.array(ret)

def quick_normalized_model(timestamps, normalization=True):
    template = []
    for t in timestamps:
        if t < HEAT_INTERVAL:
            template.append(np.sqrt(t))
        else:
            template.append(np.sqrt(t) - np.sqrt(t - HEAT_INTERVAL))
    template = np.array(template)
    if normalization:
        template = normalize(template)
    return template

def resampling(t, data, pts):
    new_t = np.linspace(0., OBSERVATION_INTERVAL, num=pts)
    new_data = np.interp(new_t, t, data)
    return new_data

def DTWDist(query, template):
    alignment = R.dtw(query.tolist(), template.tolist(), keep=True)
    dist = alignment.rx('distance')[0][0]
    return dist

def euclideanDist(query, template):
    temp = query - template
    return np.sum(temp*temp)


def weightedeuclideanDist(timestamp, query, template):
    temp = query - template
    sq = temp*temp
    for i, d in enumerate(timestamp):
        if d <= HEAT_INTERVAL:
            sq[i] *= TRIAL_INTERVAL/HEAT_INTERVAL - 1.
        else:
            break
    return np.sum(sq)

# DTW distance between normalized pixel time series and normalized simulated time series
def normalized_DTW_score(timestamps, query):
    template = quick_normalized_model(timestamps)

    # normalize query
    query = normalize(query)

    return DTWDist(query, template)

def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = np.expand_dims(kernel, axis=0)
    return kernel

def conv(darr, k_size, stride, gaussian=False, sigma=None):
    if gaussian:
        kernel = gkern(k_size, sigma)
    else:
        kernel = np.ones((1,k_size,k_size)) / (k_size ** 2.)
    kstart = int(np.floor((k_size-1)/2.))
    kend = -int(np.ceil((k_size-1)/2.))

    darr = convolveim(darr, kernel, mode='constant')
    darr = darr[:, kstart:kend:stride, kstart:kend:stride]
    return darr


############################################################
### Thermal Camera Data
############################################################
def get_trials(material):
    path = os.path.join(DATA_PATH, material)
    return sorted([int(d[5:]) for d in os.listdir(path) if 'trial' in d])


def rename_trials(material):
    path = os.path.join(DATA_PATH, material)
    dirs = get_trials(material)
    for i, d in enumerate(dirs):
        while 'trial%d'%i in os.listdir(path) and i != d:
            print 'Waiting for last dir to finish..'
            time.sleep(0.1)
        os.rename(os.path.join(path, 'trial%d'%d),
            os.path.join(path, 'trial%d'%i))

def check_existing_trials(material, base_path):
    path = os.path.join(base_path, material)
    if not os.path.exists(path):
        os.makedirs(path)
    ts = [int(f[:-4].split('l')[-1]) for f in os.listdir(path)]
    return ts

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

def load_trial_data(material, trial_num, subtract_min, normalization=False, use_min_pixel=False, window=None, t_limit=OBSERVATION_INTERVAL):
    files = get_trial_files(material, trial_num)

    print "Creating buffer"
    timestamp = []
    darr = []
    for i,f in enumerate(files):
        disp_to_term("Loading files %d     " % i)
        t, img = extract_binary(f)
        timestamp.append(t)
        darr.append(img)
        if t > t_limit * 1000.:
            break
    print "\nDone"

    if 0. not in timestamp:
        timestamp = [0.] + timestamp
        darr = [darr[0]] + darr

    timestamp = np.array(timestamp).astype(float)
    timestamp /= 1000.

    darr = np.array(darr)
    if subtract_min:
        print "Subtracting Min.."

        if not use_min_pixel:
            min_frame = np.min(darr, axis=0)
            darr -= min_frame
        else:
            min_pixel = np.min(darr)
            darr -= min_pixel

    if normalization:
        print "Normalizing.."

        if not use_min_pixel:
            max_frame = np.max(darr, axis=0)
            darr /= max_frame
        else:
            max_pixel = np.max(darr)
            darr /= max_pixel

    if window is not None:
        ylow, yhigh, xlow, xhigh = window
        darr = darr[:, ylow:yhigh, xlow:xhigh]

    # minval = np.min(darr)
    # maxval = np.max(darr)
    # print "Load Complete! Min: %.2f; Max: %.2f" % (minval, maxval)

    return timestamp, darr


def extract_binary(filename):
    arr = array.array('H')
    arr.fromfile(open(filename, 'rb'), os.path.getsize(filename)/arr.itemsize)
    return int(filename.split('/')[-1][:-4]), np.array(arr, dtype='float64').reshape((lHeight, lWidth))

def extract_trial(material, trial_num):
    files = get_trial_files(material, trial_num)

    data = []
    for i, f in enumerate(files):
        timestamp, img = extract_binary(f)
        print i, timestamp
        data.append((timestamp, img))

    times, imgs = zip(*data)

    save_pickle({'time':times, 'data':imgs}, os.path.join(path, 'cam.pkl'))

def play_trial(material, trial_num, step=0.000001, subtract_min=True, normalization=True, window=None, jump=5):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min, normalization, window=window)
    minval = np.min(darr)
    maxval = np.max(darr)

    # ylow, yhigh, xlow, xhigh = vdo_window
    # darr[:,ylow,xlow] = 6000
    # darr[:,yhigh,xhigh] = 6000

    plt.ion()
    plt.imshow(darr[0], clim=[minval, maxval])
    plt.colorbar()
    for i, img in enumerate(darr):
        if i % jump == 0:
            print np.mean(img)
            plt.imshow(img, clim=[minval, maxval])
            plt.title(timestamp[i])
            plt.pause(step)

def per_pixel_reading_plot(material, trial_num, subtract_min=False, diff=-1):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    for i in range(lHeight):
        for j in range(lWidth):
            disp_to_term("Ploting pixel (%d,%d)            " % (i,j))
            if diff < 0 or (np.max(darr[:,i,j]) - np.min(darr[:,i,j])) >= diff:
                plt.plot(timestamp, darr[:,i,j], linewidth=0.2)

    print '\nDone'
    plt.show()

def pixel_variance_distribution(material, trial_num, subtract_min=False):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    var = []
    for i in range(lHeight):
        for j in range(lWidth):
            diff = np.max(darr[:,i,j]) - np.min(darr[:,i,j])
            var.append(diff)
            disp_to_term("Ploting pixel (%d,%d) with diff %.2f           " % (i,j,diff))

    plt.hist(var, bins=1000)
    print '\nDone'
    plt.show()

def pixel_dtwDist_distribution(material, trial_num, top=500):

    print material, trial_num
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    dtwmat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))

    dtw_list = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                dtw_list.append(dtwmat[i][j])

    dtw_list = sorted(dtw_list)[:top]

    plt.hist(dtw_list, bins=max(10, top/50))
    print '\nDone'
    plt.show()

def top_variance_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=500):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    varmat = load_pickle(os.path.join(VAR_PATH, material, 'trial%d.pkl'%(trial_num)))
    template = quick_normalized_model(timestamp)

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                pixels.append((varmat[i][j], i, j))
                disp_to_term("Pixel (%d,%d) with variance %.2f           " % (i,j,varmat[i][j]))
            else:
                disp_to_term("Pixel (%d,%d) not in window           " % (i,j))

    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
    print "Done"

    dist_list = []
    for ind, item in enumerate(pixels):
        var, i, j = item
        dist = euclideanDist(normalize(darr[:,i,j]), template)
        dist_list.append(dist)
        disp_to_term("Drawing pixel #%d/%d (%d,%d) with dtwdist %.3E      " % (ind, num_pixels, i, j, dist))

        temp = darr[:,i,j]
        if normalization:
            temp = normalize(temp)

        plt.plot(timestamp, temp, linewidth=0.2)
        plt.title(material)

    print "Average DTWDist: ", np.mean(dist_list)
    print '\nDone'
    plt.show()

def top_DTWDist_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=500):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    dtwmat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))
    template = quick_normalized_model(timestamp)

    if normalization:
        plt.plot(timestamp, template, 'r', linewidth=1)

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                pixels.append((dtwmat[i][j], i, j))
                disp_to_term("Pixel (%d,%d) with dtwdist %.2f           " % (i,j,dtwmat[i][j]))
            else:
                disp_to_term("Pixel (%d,%d) not in window           " % (i,j))

    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0])[:num_pixels]
    print "Done"
    print pixels[0], pixels[-1]

    dist_list = []
    for ind, item in enumerate(pixels):
        var, i, j = item
        dist = euclideanDist(normalize(darr[:,i,j]), template)
        dist_list.append(dist)
        disp_to_term("Drawing pixel #%d/%d (%d,%d) with var %d       " % (ind, num_pixels, i, j, var))

        temp = darr[:,i,j]
        if normalization:
            temp = normalize(temp)

        plt.plot(timestamp, temp, linewidth=0.2)
        plt.title(material)

    print "Average DTWDist: ", np.mean(dist_list)
    print '\nDone'
    plt.show()


def top_eucDist_pixels_plot(material, trial_num, normalization=True, smoothing=True, subtract_min=False, num_pixels=500):

    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    eucmat = load_pickle(os.path.join(EUC_PATH, material, 'trial%d.pkl'%(trial_num)))
    template = quick_normalized_model(timestamp)

    if normalization:
        if smoothing:
            # plt.plot(timestamp, butter_lowpass_filter(template, cutoff, fs, order), 'r', linewidth=1)
            plt.plot(timestamp, padded_moving_average(template, 19), 'r', linewidth=1)
        else:
            plt.plot(timestamp, template, 'r', linewidth=1)

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                pixels.append((eucmat[i][j], i, j))
                disp_to_term("Pixel (%d,%d) with dtwdist %.2f           " % (i,j,eucmat[i][j]))
            else:
                disp_to_term("Pixel (%d,%d) not in window           " % (i,j))

    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0])[:num_pixels]
    print "Done"
    print pixels[0], pixels[-1]

    dist_list = []
    for ind, item in enumerate(pixels):
        var, i, j = item
        # dist = weightedeuclideanDist(timestamp, normalize(darr[:,i,j]), template)
        dist = np.correlate(normalize(darr[:,i,j]), template)
        dist_list.append(dist)
        disp_to_term("Drawing pixel #%d/%d (%d,%d) with euc %d       " % (ind, num_pixels, i, j, var))

        temp = darr[:,i,j]
        if normalization:
            temp = normalize(temp)
        if smoothing:
            # temp = butter_lowpass_filter(temp, cutoff, fs, order)
            temp = padded_moving_average(temp, n=19)

        plt.plot(timestamp, temp, linewidth=0.2)
        plt.title(material)

    print "Average eucDist: ", np.mean(dist_list)
    print '\nDone'
    plt.show()

# def rank_by_dist_then_variance_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=1000, heating_time=1.0, euclidean=False, downsampling=500):
#     timestamp, darr = load_trial_data(material, trial_num, subtract_min)

#     template = None
#     if downsampling > 0:
#         template = quick_normalized_model(np.linspace(0., TRIAL_INTERVAL, num=downsampling), heating_time)
#     else:
#         template = quick_normalized_model(timestamp, heating_time)

#     pixels = []
#     for i in range(lHeight):
#         for j in range(lWidth):
#             diff = np.max(darr[:,i,j]) - np.min(darr[:,i,j])
#             dist = float('inf')
#             if euclidean:
#                 dist = euclideanDist(normalize(darr[:,i,j]), template)
#             else:
#                 temp = darr[:,i,j].copy()
#                 if downsampling > 0:
#                     temp = resampling(timestamp, temp, pts=downsampling)
#                 dist = DTWDist(normalize(temp), template)
#             pixels.append((diff, dist, i, j))
#             disp_to_term("Pixel (%d,%d) with diff %.2E dist %.2E          " % (i,j,diff,dist))

#     print "\nSorting by dist.."
#     pixels = sorted(pixels, key=lambda x: x[1])[:num_pixels*2]
#     print "Done"

#     print "\nSorting by diff.."
#     pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
#     print "Done"

#     for ind, item in enumerate(pixels):
#         var, dist, i, j = item
#         disp_to_term("Drawing pixel #%d/%d (%d,%d) with var %d dist %.2E      " % (ind, num_pixels, i, j, var, dist))

#         temp = darr[:,i,j]
#         if normalization:
#             temp = normalize(temp)

#         plt.plot(timestamp, temp, linewidth=0.2)
#         plt.title(material)

#     print '\nDone'
#     plt.show()

def create_video_from_trial(material, trial_num, base_path=VIDEO_PATH, use_min_pixel=False):

    timestamp, darr = load_trial_data(material, trial_num, True, use_min_pixel)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.mov'%(trial_num))

    maxval = np.max(darr)
    darr *= 255
    darr /= maxval
    darr = np.uint8(darr)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(len(timestamp) / OBSERVATION_INTERVAL)
    out = cv2.VideoWriter(savefile, fourcc, fps, (lWidth,lHeight))

    for i, img in enumerate(darr):
        disp_to_term("Rendering frame %d      " % i)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.write(img_bgr)

    print "\nDone\n"
    out.release()

def generate_DWT_Distance_matrix(material, trial_num, normalization=True, subtract_min=False, euclidean=False, downsampling=500, base_path=DTW_PATH):
    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    template = None
    if downsampling > 0:
        template = quick_normalized_model(np.linspace(0., OBSERVATION_INTERVAL, num=downsampling))
    else:
        template = quick_normalized_model(timestamp)

    result = np.zeros((256,324))

    for i in range(lHeight):
        for j in range(lWidth):
            dist = float('inf')
            if euclidean:
                dist = euclideanDist(normalize(darr[:,i,j]), template)
            else:
                temp = darr[:,i,j].copy()
                if downsampling > 0:
                    temp = resampling(timestamp, temp, pts=downsampling)

                dist = DTWDist(normalize(temp), template)

                result[i][j] = dist
            disp_to_term("Pixel (%d,%d) with dist %.2E          " % (i,j,dist))

    save_pickle(result, savefile)
    print '\nDone'

def generate_variance_matrix(material, trial_num, subtract_min=False, base_path=VAR_PATH):
    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = np.zeros((256,324))

    for i in range(lHeight):
        for j in range(lWidth):
            variance = np.var(darr[:,i,j], dtype=np.float64)

            result[i][j] = variance
            disp_to_term("Pixel (%d,%d) with var %.2E          " % (i,j,variance))

    save_pickle(result, savefile)
    print '\nDone'

def generate_ChebyshevDistance_matrix(material, trial_num, subtract_min=False, base_path=CHE_PATH):
    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = darr.max(axis=0) - darr.min(axis=0)

    # result = np.zeros((256,324))

    # for i in range(lHeight):
    #     for j in range(lWidth):
    #         cheb = np.max(darr[:,i,j]) - np.min(darr[:,i,j])

    #         result[i][j] = cheb
    #         disp_to_term("Pixel (%d,%d) with var %.2E          " % (i,j,cheb))

    save_pickle(result, savefile)
    print '\nDone'

def generate_euclidean_Dist_matrix(material, trial_num, subtract_min=False, base_path=EUC_PATH):
    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    template = quick_normalized_model(timestamp)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = np.zeros((256,324))

    for i in range(lHeight):
        for j in range(lWidth):
            euc = euclideanDist(normalize(darr[:,i,j]), template)

            result[i][j] = euc
            disp_to_term("Pixel (%d,%d) with euc %.2E          " % (i,j,euc))

    save_pickle(result, savefile)
    print '\nDone'


def in_window_average_eucDist():
    for material in materials:
        for trial_num in range(NUM_TRIALS):
            print material, trial_num
            window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
            eucmat = load_pickle(os.path.join(EUC_PATH, material, 'trial%d.pkl'%(trial_num)))

            euc_list = []
            for i in range(lHeight):
                for j in range(lWidth):
                    if inside_window(window, (i,j)):
                        euc_list.append(eucmat[i][j])

            print "Average eucDist: ", np.mean(euc_list)

def in_window_average_dtwDist(top_num=250):
    for material in materials:
        for trial_num in range(NUM_TRIALS):
            print material, trial_num
            window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
            dtwmat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))

            dtw_list = []
            for i in range(lHeight):
                for j in range(lWidth):
                    if inside_window(window, (i,j)):
                        dtw_list.append(dtwmat[i][j])

            dtw_list = sorted(dtw_list)[:top_num]
            print dtw_list[0], dtw_list[-1]
            print "Average dtwDist: ", np.mean(dtw_list)

def display_binary(fin):
    time, data = extract_binary(fin)
    plt.imshow(data)
    plt.show()

def bin2mat(fin, fout):
    time, data = extract_binary(fin, fout)
    save_pickle(data, fout)


############################################################
### Classification
############################################################
def create_per_pixel_dataset(materials, base_name, normalization=True, subtract_min=False, num_pixels=500):

    for material in materials:
        for trial_num in get_trials(material):
            print "Processing: %s, trial%d" % (material, trial_num)

            timestamp, darr = load_trial_data(material, trial_num, subtract_min)
            chemat = load_pickle(os.path.join(CHE_PATH, material, 'trial%d.pkl'%(trial_num)))
            window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))

            pixels = []
            for i in range(lHeight):
                for j in range(lWidth):
                    if inside_window(window, (i,j)):
                        pixels.append((chemat[i][j], i, j))

            print "\nSorting.."
            pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
            print "Done"

            np.random.shuffle(pixels)
            training, test = pixels[:num_pixels*3/4], pixels[num_pixels*3/4:]

            with open('%s_train.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in training:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, NUM_OBSERVATIONS)
                    writer.writerow([classes.index(material.split('_')[0])] + temp.tolist())

            with open('%s_test.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in test:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, NUM_OBSERVATIONS)
                    writer.writerow([classes.index(material.split('_')[0])] + temp.tolist())

            print '\nDone'


def create_difficult_dataset(base_name='difficult', normalization=False, subtract_min=False, num_pixels=500):

    for material in difficult:
        for trial_num in range(NUM_TRIALS):
            print "Processing: %s, trial%d" % (material, trial_num)

            timestamp, darr = load_trial_data(material, trial_num, subtract_min)
            window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
            dtwmat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))

            pixels = []
            for i in range(lHeight):
                for j in range(lWidth):
                    if inside_window(window, (i,j)):
                        pixels.append((dtwmat[i][j], i, j))
                        disp_to_term("Pixel (%d,%d) with dtwdist %.2f           " % (i,j,dtwmat[i][j]))
                    else:
                        disp_to_term("Pixel (%d,%d) not in window           " % (i,j))

            print "\nSorting.."
            pixels = sorted(pixels, key=lambda x: x[0])[:num_pixels]
            print "Done"
            print pixels[0], pixels[-1]

            np.random.shuffle(pixels)
            training, test = pixels[:num_pixels/2], pixels[num_pixels/2:]

            with open('%s_train.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in training:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp)
                    writer.writerow([materials.index(material)] + temp.tolist())

            with open('%s_test.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in test:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp)
                    writer.writerow([materials.index(material)] + temp.tolist())

            print 'Done\n'

def create_informative_dataset(materials, base_name, normalization=True, subtract_min=False, binary=True):
    for material in materials:
        for trial_num in get_trials(material):
            print "Processing: %s, trial%d" % (material, trial_num)

            timestamp, darr = load_trial_data(material, trial_num, subtract_min)
            chemat = load_pickle(os.path.join(CHE_PATH, material, 'trial%d.pkl'%(trial_num)))
            window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))

            pixels = []
            noninformative = []
            for i in range(lHeight):
                for j in range(lWidth):
                    if inside_window(window, (i,j)):
                        pixels.append((chemat[i][j], i, j))
                    else:
                        noninformative.append((chemat[i][j], i, j))

            num_pixels = len(pixels)
            total_noninf = len(noninformative)
            print "\nSorting.."
            noninformative = sorted(noninformative, key=lambda x: x[0], reverse=True)
            # linspace sampling from non informative
            selected = np.linspace(0., total_noninf-1, num=num_pixels).astype('int')
            print "Done"
            print num_pixels, len(selected)

            np.random.shuffle(pixels)
            np.random.shuffle(selected)
            training, test = pixels[:num_pixels/2], pixels[num_pixels/2:]
            training_n, test_n = selected[:num_pixels/2], selected[num_pixels/2:]

            with open('%s_train.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in training:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, 100)
                    writer.writerow([1 if binary else materials.index(material)] + temp.tolist())

                for ind in training_n:
                    diff, i, j = noninformative[ind]
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, 100)
                    writer.writerow([0 if binary else len(materials)] + temp.tolist())


            with open('%s_test.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for diff, i, j in test:
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, 100)
                    writer.writerow([1 if binary else materials.index(material)] + temp.tolist())

                for ind in test_n:
                    diff, i, j = noninformative[ind]
                    temp = darr[:,i,j]
                    if normalization:
                        temp = normalize(temp)
                    temp = resampling(timestamp, temp, 100)
                    writer.writerow([0 if binary else len(materials)] + temp.tolist())

            print '\nDone'





def create_regional_dataset(materials, base_name, normalization=False):

    for material in materials:
        # trials = np.array(get_trials(material))
        # np.random.shuffle(trials)
        # train, test = np.split(trials, 2)
        train = range(50)
        test = range(50,100)

        for trial_num in train:
            print "Processing: %s, trial%d, train" % (material, trial_num)

            darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

            darr = conv(darr, 15, 4, True, 0.125)

            _, h, w = darr.shape

            with open('%s_train.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for i in range(h):
                    for j in range(w):
                        temp = darr[:,i,j]
                        temp = resampling(timestamp, temp, 200)
                        if normalization:
                            temp = normalize2(temp)
                        writer.writerow([materials.index(material)] + temp.tolist())

        for trial_num in test:
            print "Processing: %s, trial%d, test" % (material, trial_num)

            darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

            darr = conv(darr, 15, 4, True, 0.125)

            _, h, w = darr.shape

            with open('%s_test.csv'%base_name, 'ab') as csvfile:
                writer = csv.writer(csvfile)

                for i in range(h):
                    for j in range(w):
                        temp = darr[:,i,j]
                        temp = resampling(timestamp, temp, 200)
                        if normalization:
                            temp = normalize2(temp)
                        writer.writerow([materials.index(material)] + temp.tolist())


        print '\nDone'



# # Extract Informative Pixels from difficult materials
# def create_dataset_difficult(material, trial_num, normalization=True, subtract_min=False, num_pixels=1000, heating_time=1.0, euclidean=False, downsampling=500):
#     timestamp, darr = load_trial_data(material, trial_num, subtract_min)

#     template = None
#     if downsampling > 0:
#         template = quick_normalized_model(np.linspace(0., TRIAL_INTERVAL, num=downsampling), heating_time)
#     else:
#         template = quick_normalized_model(timestamp, heating_time)

#     pixels = []
#     for i in range(lHeight):
#         for j in range(lWidth):
#             diff = np.max(darr[:,i,j]) - np.min(darr[:,i,j])
#             dist = float('inf')
#             if euclidean:
#                 dist = euclideanDist(normalize(darr[:,i,j]), template)
#             else:
#                 temp = darr[:,i,j].copy()
#                 if downsampling > 0:
#                     temp = resampling(timestamp, temp, pts=downsampling)
#                 dist = DTWDist(normalize(temp), template)
#             pixels.append((diff, dist, i, j))
#             disp_to_term("Pixel (%d,%d) with diff %.2E dist %.2E          " % (i,j,diff,dist))

#     print "\nSorting by dist.."
#     pixels = sorted(pixels, key=lambda x: x[1])[:num_pixels]
#     print "Done"

#     # np.random.shuffle(pixels)
#     # training, test = pixels[:num_pixels/2], pixels[num_pixels/2:]

#     with open('%s_%d.csv'%(material, trial_num), 'wb') as csvfile:
#         writer = csv.writer(csvfile)

#         for diff, dist, i, j in pixels:
#             temp = darr[:,i,j]
#             if normalization:
#                 temp = normalize(temp)
#             temp = resampling(timestamp, temp)
#             writer.writerow([materials.index(material)] + [i,j] + temp.tolist())

#     print '\nDone'


############################################################
### Analyze Extracted Pixels
############################################################
def inside_window(window, pt):
    path = mpltPath.Path(window)
    return path.contains_point(pt)

# def visualize_extraxted_pixels(material, trial_num, path=''):
#     data = np.genfromtxt(os.path.join(path, '%s_%d.csv'%(material, trial_num)), delimiter=',')
#     pts = 1500 # data.shape[1] - 1
#     time = np.linspace(0., TRIAL_INTERVAL, pts)

#     print "Load Complete"
#     for entry in data:
#         plt.plot(time, entry[-pts:], linewidth=0.2)
#         plt.title(material)

#     print '\nDone'
#     plt.show()

# def visualize_extracted_pixel_locations(material, trial_num, path=''):
#     data = np.genfromtxt(os.path.join(path, '%s_%d.csv'%(material, trial_num)), delimiter=',')
#     mask = np.zeros((256,324))

#     print "Load Complete"
#     for entry in data:
#         x, y = int(entry[1]), int(entry[2])
#         mask[x][y] = 255

#     plt.title('%s_%d' % (material, trial_num))
#     plt.matshow(mask)
#     print '\nDone'
#     plt.show()

def visualize_mat(material, trial_num, path):

    data = load_pickle(os.path.join(path, material, 'trial%d.pkl'%(trial_num)))

    plt.matshow(data)
    plt.colorbar()
    plt.title(material)

    print '\nDone'
    plt.show()

def create_window_using_frame_after(material, trial_num, frame=950, path=WINDOW_PATH):
    global coords
    coords = []

    files = get_trial_files(material, trial_num)

    savepath = os.path.join(path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savefile = os.path.join(savepath, 'trial%d.pkl'%trial_num)

    f = None
    for filename in files:
        if int(filename.split('/')[-1][:-4]) > frame:
            f = filename
            break

    _,data = extract_binary(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data)
    ax.set_title('%s_%d' % (material, trial_num))
    fig.colorbar(cax)

    def onclick(event):
        global ix, iy, coords
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)

        coords.append((ix, iy))

        if len(coords) == 4:
            save_pickle(coords, savefile)
            fig.canvas.mpl_disconnect(cid)

        print coords
        return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    print '\nDone'
    plt.show()


############################################################
### Video Processing
############################################################
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
    out = keras.layers.Dense(NUM_CLASSES, activation='softmax')(full)

    model = Model(input=x, output=out)
    return model

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model



def create_feature_extractor():
    trained_model_layers = load_trained_model().layers

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

    out = keras.layers.pooling.GlobalAveragePooling2D()(conv3)

    model = Model(input=x, output=out)
    return model

def render_trial(material, trial_num, model, base_path):
    timestamp, darr = load_trial_data(material, trial_num, False)

    savepath = os.path.join(base_path, material)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = np.zeros((NUM_CLASSES,256,324))

    start = time.time()

    darr -= np.min(darr, axis=0)
    darr /= np.max(darr, axis=0)

    darr_mean = 0.562614819045
    darr_std = 0.214789864592
    darr = (darr - darr_mean)/(darr_std)
    X = darr.reshape((darr.shape[0],-1)).T

    f = interp1d(timestamp, X, axis=1)
    xp = np.linspace(0., OBSERVATION_INTERVAL, num=NUM_OBSERVATIONS)
    X = f(xp)

    # plt.plot(xp, X[(lHeight/2)*lWidth+lWidth/2, :])
    # plt.show()

    X = X.reshape(X.shape + (1,1,)) # FCN
    print 'Test set shape: ', X.shape

    ys = model.predict(X)
    print 'Result shape: ', ys.shape

    cs = np.argmax(ys, axis=1).reshape((lHeight, lWidth)) # FCN
    # cs = ys.astype('int').reshape((lHeight, lWidth)) # SVM
    print cs.shape

    # for i in range(lHeight):
    #     for j in range(lWidth):
    #         temp = darr[:,i,j].copy()
    #         temp = normalize(temp)
    #         temp = resampling(timestamp, temp, NUM_OBSERVATIONS)

    #         temp = temp.reshape((1, NUM_OBSERVATIONS, 1, 1)) # FCN
    #         # temp = temp.reshape(1, -1) # SVM

    #         c = model.predict(temp)

    #         y = np.argmax(c) # FCN
    #         # y = int(c[0]) # SVM

    #         disp_to_term("Pixel (%d,%d) with class %s          " % (i,j,materials[y]))
    #         result[:,i,j] = c

    print '\n\nTime Elapsed: %.4f' % (time.time() - start)

    plt.matshow(cs)
    plt.colorbar()
    plt.show()
    save_pickle(ys, savefile)
    print '\nDone'

def classify_video(material, trial_num, model, num_pixels=250):
    timestamp, darr = load_trial_data(material, trial_num, False)
    chemat = load_pickle(os.path.join(CHE_PATH, material, 'trial%d.pkl'%(trial_num)))
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                pixels.append((chemat[i][j], i, j))

    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
    print "Done"

    X = []
    for var,ii,jj in pixels:
        temp = darr[:,ii,jj].copy()
        temp = normalize(temp)
        temp = resampling(timestamp, temp, NUM_OBSERVATIONS)
        X.append(temp)
    #     plt.plot(temp)
    # plt.show()
    X = np.array(X)
    # X_mean = 0.562614819045
    # X_std = 0.214789864592
    # # print X_mean, X_std
    # X = (X - X_mean)/(X_std)
    X = X.reshape(X.shape + (1,1,))
    print X.shape

    ys = model.predict(X)
    print 'Result shape: ', ys.shape

    cs = np.argmax(ys, axis=1).astype('int')

    print('Voting Result: ', Counter([classes[i] for i in cs]))

def classify_regions(material, trial_num, model, labels):
    k_size = 17
    kernel = np.ones((1,k_size,k_size)) / (k_size ** 2.)

    # print "Processing: %s, trial%d" % (material, trial_num)

    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

    darr = convolveim(darr, kernel, mode='constant')
    darr = darr[:, k_size//2:-(k_size//2), k_size//2:-(k_size//2)]

    _, h, w = darr.shape

    X = []
    for i in range(h):
        for j in range(w):
            temp = darr[:,i,j]
            temp = resampling(timestamp, temp, 200)
            temp = normalize2(temp)
            X.append(temp)

    X = np.array(X)
    X = X.reshape(X.shape + (1,1,))
    # print X.shape

    ys = model.predict(X)
    # print 'Result shape: ', ys.shape

    cs = np.argmax(ys, axis=1).astype('int')

    ctr = Counter([labels[i] for i in cs])
    # print('Voting Result: ', ctr)
    return ctr.most_common(1)[0][0]


def knn_classify_regions(material, trial_num, sample_trial_num):

    refs = []
    for m in dist_30:
        temp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%sample_trial_num, 'vdo.npy'))
        temp = temp.mean(axis=(1,2))
        timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%sample_trial_num, 'timestamp.npy'))
        temp = resampling(timestamp, temp, 200)
        temp = normalize2(temp)
        refs.append(temp)

    k_size = 17
    kernel = np.ones((1,k_size,k_size)) / (k_size ** 2.)

    print "Processing: %s, trial%d" % (material, trial_num)

    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

    darr = convolveim(darr, kernel, mode='constant')
    darr = darr[:, k_size//2:-(k_size//2), k_size//2:-(k_size//2)]

    _, h, w = darr.shape

    labels = []
    for i in range(h):
        for j in range(w):
            temp = darr[:,i,j]
            temp = resampling(timestamp, temp, 200)
            temp = normalize2(temp)

            label = None
            err = float('inf')
            for ind, ref in enumerate(refs):
                mse = np.sum((ref - temp)**2)
                if mse < err:
                    label = classes[ind]
                    err = mse
            labels.append(label)

    print('Voting Result: ', Counter(labels))


def display_rendered_img(material, trial_num, base_path):
    savepath = os.path.join(base_path, material)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = load_pickle(savefile)
    print result.shape
    mat = np.argmax(result, axis=1).reshape((lHeight, lWidth))

    plt.matshow(mat)
    plt.colorbar()
    plt.title('%s_%d' % (material, trial_num))
    plt.show()

def display_class_activations(material, trial_num, base_path):
    savepath = os.path.join(base_path, material)
    savefile = os.path.join(savepath, 'trial%d.pkl'%(trial_num))

    result = load_pickle(savefile)
    print result.shape
    for i in range(result.shape[-1]):
        # plt.figure()
        mat = result[:,i].reshape((lHeight, lWidth))
        plt.matshow(mat)
        plt.colorbar()
        plt.title('%s_%d_%d' % (material, trial_num, i))

    plt.show()

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