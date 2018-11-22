# -*- coding: utf-8 -*-
"""
2017 Zillow Kaggle Challenge

https://www.kaggle.com/c/zillow-prize-1

@author: MP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
from functools import partial



# Set global parameters
data_dir = os.path.join(os.getcwd(), "./source_data")
propertcsv = "properties_2016.csv"
traincsv = "train_2016_v2.csv"
submissioncsv = "sample_submission.csv"

full_data = "full.csv"


########################################################################
################################  FUNCTIONS  ###########################
########################################################################

def convert_column_to_object(df, char_columns):
    # to conver the columns in the char_columns into object(i.e. string)
    for col in char_columns:
        if col not in df.columns.values:
            continue
        elif df[col].dtypes != object:
            df[col] = np.array(df[col].values, dtype=np.str)
        else:
            continue
    return df


class XGBOOSTQUANTILE(BaseEstimator, RegressorMixin):
    def __init__(self, quant_alpha,quant_delta,quant_thres,quant_var,
                n_estimators = 100, max_depth = 3, reg_alpha = 5.,
                reg_lambda = 1.0, gamma = 0.5):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var
        #xgboost parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_alpha= reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        #keep xgboost estimator in memory
        self.clf = None

    @staticmethod
    def quantile_loss(y_true, y_pred,_alpha,_delta,_threshold,_var):
        x = y_true - y_pred
        grad = (x<(_alpha-1.0)*_delta)*(1.0-_alpha)- ((x>=(_alpha-1.0)*_delta)&
                                (x<_alpha*_delta) )*x/_delta-_alpha*(x>_alpha*_delta)
        hess = ((x>=(_alpha-1.0)*_delta)& (x<_alpha*_delta) )/_delta
        _len = np.array([y_true]).size
        var = (2*np.random.randint(2, size=_len)-1.0)*_var
        grad = (np.abs(x)<_threshold )*grad - (np.abs(x)>=_threshold )*var
        hess = (np.abs(x)<_threshold )*hess + (np.abs(x)>=_threshold )
        return grad, hess

    def fit(self, X, y):
        self.clf = XGBRegressor(
                        objective=partial(quantile_loss,
                        _alpha = self.quant_alpha,
                        _delta = self.quant_delta,
                        _threshold = self.quant_thres,
                        _var = self.quant_var),
                        n_estimators = self.n_estimators,
                        max_depth = self.max_depth,
                        reg_alpha =self.reg_alpha,
                        reg_lambda = self.reg_lambda,
                        gamma = self.gamma )
        self.clf.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        score = (self.quant_alpha-1.0)*(y-y_pred)*(y<y_pred)+self.quant_alpha*(y-y_pred)* (y>=y_pred)
        score = 1./np.sum(score)
        return score


############################################################################################
################################ STEP 01 Load in Data and Merges ###########################
############################################################################################

# Load Data
full = pd.read_csv(full_data)

char_vars = pd.read_csv("./char_vars.csv")['Features'].values
num_vars = pd.read_csv("./num_vars.csv")['Features'].values
pure_char_vars = pd.read_csv("./char_vars.csv")["Pure_Character"].values
pure_char_vars = pure_char_vars[list(map(lambda x: type(x) == str, pure_char_vars))]
char_vars = np.intersect1d(char_vars, full.columns.values)
num_vars = np.intersect1d(num_vars, full.columns.values)
pure_char_vars = np.intersect1d(pure_char_vars, full.columns.values)

chars2change = np.intersect1d(full.columns.values[full.dtypes !='object'], pure_char_vars)
# convert some char vars to char even if values appear numerical
full = convert_column_to_object(full, chars2change)
# add some missing variables back
full["numberofstories"] = full["numberofstories_log"].apply(np.exp)
full["yardbuildingsqft17_log"] = np.log(full["yardbuildingsqft17"])
full["finishedsquarefeet6_log"] = np.log(full["finishedsquarefeet6"])


# Common Setting
non_predictors = ["parcelid", "transactiondate", "logerror", "split", "random", "foldid",
                    "logerror_pred", "rowidx"]
predictors = np.unique(np.hstack((char_vars, num_vars)))
month_vars = ["transactionMonth", "transactionYear"]

full.to_csv("full.csv", index=None)

# Print the string
"'"+"', '".join(non_predictors)+"'"
# 'parcelid', 'transactiondate', 'logerror', 'split', 'random', 'foldid', 'logerror_pred', 'rowidx'
"'"+"', '".join(predictors)+"'"
# 'airconditioningtypeid', 'bathroomcnt', 'bathroomcnt_log', 'bedroomcnt', 'bedroomcnt_log',
# 'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr', 'calculatedbathnbr_log',
# 'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet_log', 'censustractandblock',
# 'censustractandblock_12to12', 'censustractandblock_1to12', 'censustractandblock_1to4',
# 'censustractandblock_1to8', 'censustractandblock_5to11', 'finishedfloor1squarefeet',
# 'finishedfloor1squarefeet_log', 'finishedsquarefeet12', 'finishedsquarefeet12_log',
# 'finishedsquarefeet15', 'finishedsquarefeet15_log', 'finishedsquarefeet50', 'finishedsquarefeet50_log',
# 'finishedsquarefeet6', 'fullbathcnt', 'fullbathcnt_log', 'garagetotalsqft', 'garagetotalsqft_log',
# 'hashottuborspa', 'heatingorsystemtypeid', 'landtaxvaluedollarcnt', 'landtaxvaluedollarcnt_log',
# 'latitude', 'latitude_log', 'longitude', 'longitude_log', 'lotsizesquarefeet', 'lotsizesquarefeet_log',
# 'numberofstories_log', 'poolcnt', 'poolcnt_log', 'pooltypeid7', 'propertycountylandusecode',
# 'propertycountylandusecode_1to2', 'propertylandusetypeid', 'propertyzoningdesc', 'propertyzoningdesc_1to3',
# 'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10', 'rawcensustractandblock', 'regionidcity',
# 'regionidneighborhood', 'regionidzip', 'roomcnt', 'roomcnt_log', 'structuretaxvaluedollarcnt',
# 'structuretaxvaluedollarcnt_log', 'taxamount', 'taxamount_log', 'taxdelinquencyflag',
# 'taxdelinquencyyear', 'taxdelinquencyyear_log', 'taxvaluedollarcnt', 'taxvaluedollarcnt_log',
# 'threequarterbathnbr', 'threequarterbathnbr_log', 'transactionMonth', 'transactionYear',
# 'unitcnt', 'unitcnt_log', 'yardbuildingsqft17', 'yearbuilt', 'yearbuilt_log'
"'"+"', '".join(pure_char_vars)+"'"
# 'airconditioningtypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'censustractandblock',
# 'censustractandblock_12to12', 'censustractandblock_1to12', 'censustractandblock_1to4',
# 'censustractandblock_1to8', 'censustractandblock_5to11', 'hashottuborspa', 'heatingorsystemtypeid',
# 'pooltypeid7', 'propertycountylandusecode', 'propertycountylandusecode_1to2', 'propertylandusetypeid',
# 'propertyzoningdesc', 'propertyzoningdesc_1to3', 'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10',
# 'rawcensustractandblock', 'regionidcity', 'regionidneighborhood', 'regionidzip', 'taxdelinquencyflag',
# 'transactionMonth', 'transactionYear'
"'"+"', '".join(num_vars)+"'"
# 'bathroomcnt', 'bathroomcnt_log', 'bedroomcnt', 'bedroomcnt_log', 'calculatedbathnbr',
# 'calculatedbathnbr_log', 'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet_log',
# 'finishedfloor1squarefeet', 'finishedfloor1squarefeet_log', 'finishedsquarefeet12',
# 'finishedsquarefeet12_log', 'finishedsquarefeet15', 'finishedsquarefeet15_log', 'finishedsquarefeet50',
# 'finishedsquarefeet50_log', 'finishedsquarefeet6', 'fullbathcnt', 'fullbathcnt_log', 'garagetotalsqft',
# 'garagetotalsqft_log', 'landtaxvaluedollarcnt', 'landtaxvaluedollarcnt_log', 'latitude_log',
# 'longitude_log', 'lotsizesquarefeet', 'lotsizesquarefeet_log', 'numberofstories_log', 'poolcnt',
# 'poolcnt_log', 'roomcnt', 'roomcnt_log', 'structuretaxvaluedollarcnt', 'structuretaxvaluedollarcnt_log',
# 'taxamount', 'taxamount_log', 'taxdelinquencyyear', 'taxdelinquencyyear_log', 'taxvaluedollarcnt',
# 'taxvaluedollarcnt_log', 'threequarterbathnbr', 'threequarterbathnbr_log', 'unitcnt', 'unitcnt_log',
# 'yardbuildingsqft17', 'yearbuilt', 'yearbuilt_log'
"'"+"', '".join(char_vars)+"'"
# 'airconditioningtypeid', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid',
# 'calculatedbathnbr', 'censustractandblock', 'censustractandblock_12to12', 'censustractandblock_1to12',
# 'censustractandblock_1to4', 'censustractandblock_1to8', 'censustractandblock_5to11', 'fullbathcnt',
# 'hashottuborspa', 'heatingorsystemtypeid', 'latitude', 'longitude', 'poolcnt', 'pooltypeid7',
# 'propertycountylandusecode', 'propertycountylandusecode_1to2', 'propertylandusetypeid',
# 'propertyzoningdesc', 'propertyzoningdesc_1to3', 'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10',
# 'rawcensustractandblock', 'regionidcity', 'regionidneighborhood', 'regionidzip', 'roomcnt',
# 'taxdelinquencyflag', 'taxdelinquencyyear', 'threequarterbathnbr', 'transactionMonth', 'transactionYear',
# 'unitcnt', 'yearbuilt'

