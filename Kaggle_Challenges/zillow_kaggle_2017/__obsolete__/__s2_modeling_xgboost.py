# -*- coding: utf-8 -*-
"""
2017 Zillow Kaggle Challenge

https://www.kaggle.com/c/zillow-prize-1

@author: MP
"""

import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import multiprocessing

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb

import logging
import gc


##### Common Functions #####
def pandas_fast_merge(left, right, how="inner"):
    # faster pandas merge, better than pd.merge
    common_cols = np.intersect1d(left.columns.values, right.columns.values)
    left.set_index(list(common_cols), inplace=True)
    right.set_index(list(common_cols), inplace=True)
    df = left.join(right, how=how)
    df.reset_index(inplace=True)
    left.reset_index(inplace=True)
    right.reset_index(inplace=True)
    return df


def show_sizeof(x, level=0):
    print("\t" * level, x.__class__, sys.getsizeof(x), x)
    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                show_sizeof(xx, level + 1)
        else:
            for xx in x:
                show_sizeof(xx, level + 1)


def show_sizeof2(x):
    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)
    return sizeof_fmt(sys.getsizeof(x))


# save and load sparse matrix
def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices, \
                indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), \
                        shape = loader['shape'])


## Global Parameter
submission_df = pd.read_csv("./source_data/sample_submission.csv")
submission_df["rowid"] = range(submission_df.shape[0])

# add filemode="w" to overwrite
logging.basicConfig(filename="log.log", level=logging.INFO)
#logging.debug("This is a debug message")
logging.info("Logging starts...")
#logging.error("An error has happened!")


######## STEP 01 Load in Data from previous steps ###############
full = pd.read_pickle("full_step01.p")
non_predictors = ["parcelid", "logerror", "split", "transactiondate", "rowidx"]
# The following are to shrink the size the files in order to save the memory
full = full[non_predictors]

if False:
    full_matrix = np.load('full_char_svd_step01.npy')
    full_numerical_var_matrix = np.load('full_numerical_var_matrix_step01.npy')
    full_matrix = np.hstack((full_matrix, full_numerical_var_matrix))
    np.save('full_matrix_step01', full_matrix)
if False:
    # This is for the case when one wants to use the sparse matrix for the model building
    sparse1 = load_sparse_csr("full_char_sparse_step01.npz")
    full_num_ = np.load('full_numerical_var_matrix_step01.npy')
    full_num_ = csr_matrix(full_num_)
    full_matrix_sparse = hstack((sparse1, full_num_), format='csr')
    del sparse1, full_num_
    save_sparse_csr("full_matrix_sparse", full_matrix_sparse)

full_matrix = np.load('full_matrix_step01.npy')
# to load the .npy file in R, just library(RcppCNPy) ; npyLoad("E:/zillow_kaggle_2017/zzz.npy")
gc.collect()

######## STEP 02 Model Fitting: Get the propensity of train, test ###############
# This step is due to the finding that the train and tets splits have different
# predictors distribution, hence causing the model fitting on the train to be
# biased

xgb_params = {
            'eta': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.2,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'silent': 0,
            'alpha': 1,
            'lambda': 1,
            "nthread": multiprocessing.cpu_count()-1,
            "min_child_weight": 5
        }

dtrain = xgb.DMatrix(full_matrix,
                    label=1*(full['split']=='test'),
                    #feature_names = train_X.columns.values,
                    missing = np.nan)
res = xgb.cv(xgb_params, dtrain, num_boost_round=300, nfold=4, seed = 10011,
                callbacks=[xgb.callback.print_evaluation(period=50, show_stdv=True),
                            xgb.callback.early_stop(20)])

####### LGB Version
dtrain = lgb.Dataset(full_matrix, label=1*(full['split']=='test'))
watchlist = [dtrain]
lgb_params = {
            'learning_rate': 0.1,
            "boosting_type": 'gbdt',
            'num_leaves': 15,
            "early_stopping_round": 20,
            'bagging_fraction': 0.8,
            "bagging_freq": 1,
            'feature_fraction': 0.2,
            'objective': 'binary',
            'metric': 'auc',
            'lambda_l2': 1,
            'lambda_l2': 1,
            "nthread": multiprocessing.cpu_count()-1,
            "min_data_in_leaf": 5,
            "verbose": 1
        }
res = lgb.train(lgb_params, dtrain, num_boost_round=300, watchlist, seed = 10011)


