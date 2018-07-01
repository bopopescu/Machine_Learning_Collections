#!/usr/bin/env python

# SVM+PCA for Temperature Data
from progressbar import *
import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
from math import floor
import pickle
import optparse
import unittest
import random
import os, sys
import util
from util import *
import itertools
from data_temperature_slope_kNN_SVM_DBN import *

from sklearn import decomposition
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import scipy.signal as sgn

data_path = "../1k_Trial"
instances = 100
exp = 20
num_folds = 3
total_time = [2.0]
MAX_TIME = max(total_time)

def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def create_binary_dataset(fmat1, fmat2, label1, label2, t):
    data = {'data':[], 'target':[]}
    a = 0
    for fvec in fmat1:
        temp, slope = fvec
        # if temp[0] > 35 or temp[0] < 30:
        #     # print temp[0]
        #     a += 1
        #     continue
        # found = False

        # for s in slope:
        #     if s > 0:
        #         a += 1
        #         found = True
        #         break
        # if found:
        #     continue

        length = len(temp)
        temp = temp[:t*200]
        slope = slope[:t*200]
        slope[-1] = 0
        data['data'].append(temp + slope)
        data['target'].append(str(label1))

    # print a
    a = 0
    for fvec in fmat2:
        temp, slope = fvec
        # if temp[0] > 35 or temp[0] < 30:
        #     print temp[0]
        #     a += 1
        #     continue
        # found = False

        # for s in slope:
        #     if s > 0:
        #         a += 1
        #         found = True
        #         break
        # if found:
        #     continue

        length = len(temp)
        temp = temp[:t*200]
        slope = slope[:t*200]
        slope[-1] = 0
        data['data'].append(temp + slope)
        data['target'].append(str(label2))
    # print a

    # util.rest()
    # print data
    return data

def run_pca(dataset):
    # pca = decomposition.PCA(n_components=10)
    pca = decomposition.PCA()
    pca.fit(dataset['data'])
    reduced_mat = pca.transform(dataset['data'])
    dataset['data'] = reduced_mat
    return dataset


def run_crossvalidation_new(data_dict, folds):
    # data_dict = run_pca(data_dict)
    skf = StratifiedKFold(data_dict['target'], n_folds=folds)
    svc = svm.SVC(kernel='linear')
    #svc = svm.SVC(kernel='rbf')
    #svc = svm.SVC(kernel='poly', degree=3)
    scores = cross_val_score(svc, data_dict['data'], data_dict['target'], cv=skf, scoring='f1_weighted')
    # print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2)
    return scores.mean()

def load_trial_data(ind_obj1, ind_obj2, fn):
    fmat1 = util.load_pickle(fn + '/' + str(ind_obj1) + '.pkl')
    fmat2 = util.load_pickle(fn + '/' + str(ind_obj2) + '.pkl')

    return fmat1, fmat2

def transform_erfc(vec, T_amb, t_m0=25):
    out = []
    for i in range(len(vec)):
        trial = vec[i]
        t_m0 = T_amb[i]
        temp, slope = trial
        temp = [(t - t_m0) / (temp[0] - t_m0) for t in temp]
        slope = [s / (temp[0] - t_m0) for s in slope]
        out += [(temp, slope)]
    return out

def _add_slope(d):
    materials = os.listdir(d)

    for mat in materials:
        T_ambs = []

        if mat.endswith('.pkl') or mat.startswith('.'):
            continue
        print mat
        data = []
        for f in os.listdir(d + '/' + mat):
            if f.startswith('.'):
                continue
            for _f in os.listdir(d + '/' + mat + '/' + f):
                if _f.startswith('.'):
                    continue
                vec = load_pickle(d + '/' + mat + '/' + f + '/' + _f)
                vec = [[v[0] for v in vec if v[0] >= 0], [v[2] for v in vec if v[0] >= 0]]
                # print vec, '\n\n\n'
                vec = feature_vector_diff(vec)
                data += [vec]
                T_ambs += [float(_f.split('_')[-1][:-4])]

        save_pickle((T_ambs, data), d + '/' + mat + '.pkl')

def add_slope(d):
    materials = os.listdir(d)

    for mat in materials:
        T_ambs = []

        if mat.endswith('.pkl') or mat.startswith('.'):
            continue
        print mat
        data = []
        for f in os.listdir(d + '/' + mat):
            if f.startswith('.'):
                continue
            
            _vec = load_pickle(d + '/' + mat + '/' + f)
            _vec = [[i[0], i[1]-300] for i in _vec if i[0] >= 300][:600]

            vec = feature_vector_diff(_vec)
            data += [vec]

        save_pickle(data, d + '/' + mat + '.pkl')

def to_csv(fmat1, label1, t, data):
    a = 0
    for fvec in fmat1:
        temp, slope = fvec
        # print len(temp)
        # print len(slope)
        if max(temp) > 36:
            continue

        length = len(temp)
        temp = temp[:400]
        slope = slope[:400]
        slope[-1] = 0
        data['data'].append(temp + slope)
        data['target'].append(str(label1))

    # print a

    # util.rest()
    # print data
    return data


if __name__ == '__main__':
    # mat1 = sys.argv[1]
    # mat2 = sys.argv[2]
    exp_dir = '../0710_data'
    svm_out = '../0710_out'
    add_slope(exp_dir)

    materials = [m[:-4] for m in os.listdir('../0710_data') if m.endswith('.pkl')]
    print materials
    accum = 0
    buff = ''
    for pair in list(itertools.combinations(materials, 2)):
        mat1, mat2 = pair
    
        # print "Loading Data"
        vec1 = util.load_pickle(exp_dir + '/' + mat1 + '.pkl')
        vec2 = util.load_pickle(exp_dir + '/' + mat2 + '.pkl')

        # vec1 = transform_erfc(vec1, T_amb1)
        # vec2 = transform_erfc(vec2, T_amb2)
        # print len(vec1), len(vec2)

        data_dic = create_binary_dataset(vec1, vec2, mat1, mat2, 2)
        # print "created"
        score = run_crossvalidation_new(data_dic, num_folds)
        if not os.path.exists(svm_out):
            os.makedirs(svm_out)
        buff += '%s, %s, %f\n' % (mat1, mat2, score)
        print '%s, %s, %s' % (mat1, mat2, str(score))
    print accum
    with open(svm_out + "/out.csv", "w") as text_file:
        text_file.write(buff)
    # # mat1 = sys.argv[1]
    # # mat2 = sys.argv[2]
    # exp_dir = '../1k_Trial'
    # svm_out = '../1k_exp_out'
    # add_slope(exp_dir)

    # materials = [m[:-4] for m in os.listdir('../1k_Trial') if m.endswith('.pkl')]
    # print materials
    # data = {'data':[], 'target':[]}

    # accum = 0
    # for m in materials:
    
    #     # print "Loading Data"
    #     T_amb1, vec1 = util.load_pickle(exp_dir + '/' + m + '.pkl')

    #     # vec1 = transform_erfc(vec1, T_amb1)
    #     # vec2 = transform_erfc(vec2, T_amb2)
    #     # print len(vec1), len(vec2)

    #     data = to_csv(vec1, m, 2, data)
    # print len(data['data'][0])
    # new_df = pd.DataFrame(data)
    # path = '../temp_ground'# edge_list_dir + "/" + suffix
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # new_df.to_csv(os.path.join('../temp_ground/experiment_data.csv'))

