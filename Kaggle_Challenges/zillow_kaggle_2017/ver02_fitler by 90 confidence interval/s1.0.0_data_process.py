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
import matplotlib

from datetime import datetime
from scipy import stats
from scipy.stats import kendalltau
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from scipy.stats import itemfreq
import pylab, warnings

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.decomposition import TruncatedSVD, SparsePCA, FastICA

import gc

matplotlib.style.use('ggplot')


# Set global parameters
data_dir = os.path.join(os.getcwd(), "./source_data")
propertcsv = "properties_2016.csv"
traincsv = "train_2016_v2.csv"
submissioncsv = "sample_submission.csv"



##### Common Functions #####
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


def log2(x):
    min_ = np.nanmin(x)
    if min_ >= 1: min_ = 1
    x2_ = x - min_ + 1
    return np.log(x2_)


def pandas_fast_merge(left, right, how="inner"):
    # faster pandas merge, better than pd.merge
    common_cols = np.intersect1d(left.columns.values, right.columns.values)
    left.set_index(list(common_cols), inplace=True)
    right.set_index(list(common_cols), inplace=True)
    df = left.join(right, how=how)
    df.reset_index(inplace=True)
    left.reset_index(inplace=True)
    right.reset_index(inplace=True)
    print("pandas_fast_merge has danger of re-ordering the rows!")
    return df


def add_log_to_num_columns(df, num_columns):
    # to add log to the numerical columns
    for col in num_columns:
        if col not in df.columns.values:
            continue
        elif df[col].dtypes not in [float, int]:
            continue
        else:
            df[col+'_log'] = log2(df[col].values)
    return df


class StandardScaler2(object):
    """
    The class for scaler that handles the missing values
    It needs the numpy package to be loaded first as np
    x: is the input matrix
    """
    def __init__(self):
        super(StandardScaler2, self).__init__()
    def fit(self, x):
        self.means = list(map(lambda i: np.nanmean(x[:, i]), range(x.shape[1])))
        self.stds = list(map(lambda i: np.nanstd(x[:, i]), range(x.shape[1])))
    def transform(self, newmatrix):
        return (newmatrix - np.array(self.means)) / np.array(self.stds)


def convert_str_to_date(vec):
    # convert vec like "2015-03-02" to date, where vec is the numpy array
    vec2 = pd.to_datetime(vec, format='%Y-%m-%d', errors='ignore')
    vec2_year = np.array(vec2.year.values, dtype=np.str)
    vec2_month = np.array(vec2.month.values, dtype=np.str)
    #vec2_day = np.array(vec2.day.values, dtype=np.str)
    # add manual day there
    days = ["15"] * len(vec2)
    vec3 = np.array(["{}-{}-{}".format(a_, b_.zfill(2), c_.zfill(2)) \
                    for a_, b_, c_ in zip(vec2_year, vec2_month, days)])
    # return the string date, year month
    return vec3, vec2_year, vec2_month


def converttestchar2date(x):
    return x[:4]+"-"+x[4:6]+"-15"

converttestchar2date = np.vectorize(converttestchar2date)


def order_df_columns(df, columns):
    # this is to reorder the df columns by putting the columns as the first few
    # columns to display
    columns = np.intersect1d(np.array(columns), df.columns.values)
    cols = np.hstack((np.array(columns), np.setdiff1d(df.columns.values, columns))).tolist()
    df = df[cols]
    return df


def substring_columns(df, column, substring_range=[0,4]):
    # this is to create the substring of the column for the, say, first 4 characters, [0,4]
    df["zzzrowidx"] = range(df.shape[0])
    df[column] = np.array(df[column].values, dtype=np.str)
    df_small = df[[column]].drop_duplicates(subset=[column])
    def convert_(x):
        return x[substring_range[0]:substring_range[1]]
    convert_ = np.vectorize(convert_)
    df_small[column+"_"+str(substring_range[0]+1)+"to"+str(substring_range[1])] = convert_( \
                        df_small[column].values)
    #df = pd.merge(df, df_small, on=column, how="left")
    df = pandas_fast_merge(df, df_small, how = "left")
    df = df.sort_values(['zzzrowidx'], ascending=[True])
    df.drop(["zzzrowidx"], axis=1, inplace=True)
    return df


def convert_pd_to_sparse_matrix(df):
    char_vars = list(df.columns.values[df.dtypes==object])
    counter = 0
    sprase_columns = []
    for col in char_vars:
        if counter == 0: print("-"*20)
        df2 = df[[col]].copy()
        if np.sum(df2[col].isnull()) > 0:
            df2.loc[df2[col].isnull(), col] = "NAN"
            print("Variable {} is done with filling NA with 'NAN' values...".format(col))
        print("Processing {}-th variable out of {} in the list of string fields".format(counter+1, len(char_vars)))
        if counter == 0:
            x_ = pd.get_dummies(df2[[col]], prefix=[col], sparse=True)
            sparse_matrix = csr_matrix(x_)
            print("Variable {} is done with sparse matrix creation...".format(col))
            sprase_columns += list(x_.columns.values)
            del x_
        else:
            try:
                x_ = pd.get_dummies(df2[[col]], prefix=[col], sparse=True)
                sparse_matrix = hstack((sparse_matrix, \
                                        csr_matrix(x_)), \
                                        format='csr')
                print("Variable {} is done with sparse matrix creation...".format(col))
                sprase_columns += list(x_.columns.values)
                del x_
            except MemoryError as e:
                print("*** Variable {} has too many levels, failed to create sparse matrix ***".format(col))
        counter += 1
        print("-"*20)
        gc.collect()
    return sparse_matrix, sprase_columns


def convert_pd_to_sparse_matrix2(df):
    # This is a much faster verison
    char_vars = list(df.columns.values[df.dtypes==object])
    counter = 0
    sprase_columns = []
    for col in char_vars:
        if counter == 0: print("-"*20)
        df2 = df[[col]].copy()
        if np.sum(df2[col].isnull()) > 0:
            df2.loc[df2[col].isnull(), col] = "NAN"
            print("Variable {} is done with filling NA with 'NAN' values...".format(col))
        print("Processing {}-th variable out of {} in the list of string fields".format(counter+1, len(char_vars)))
        if counter == 0:
            v = DictVectorizer(sparse=True)
            sparse_matrix = v.fit_transform(df2[[col]].to_dict(orient = 'records'))
            print("Variable {} is done with sparse matrix creation...".format(col))
        else:
            try:
                v = DictVectorizer(sparse=True)
                sparse_matrix_ = v.fit_transform(df2[[col]].to_dict(orient = 'records'))
                sparse_matrix = hstack((sparse_matrix, sparse_matrix_), format='csr')
                print("Variable {} is done with sparse matrix creation...".format(col))
                del sparse_matrix_
            except MemoryError as e:
                print("*** Variable {} has too many levels, failed to create sparse matrix ***".format(col))
        counter += 1
        print("-"*20)
        gc.collect()
    return sparse_matrix


def convert_pd_to_sparse_matrix3(df):
    # This is a much faster verison
    # the transformation v needs to be saved in the real production environment in order to
    # apply transform for new data
    char_vars = list(df.columns.values[df.dtypes==object])
    counter = 0
    sprase_columns = []
    for col in char_vars:
        if np.sum(df[col].isnull()) > 0:
            df.loc[df[col].isnull(), col] = "NAN"
            print("Variable {} is done with filling NA with 'NAN' values...".format(col))
    v = DictVectorizer(sparse=True)
    sparse_matrix = v.fit_transform(df[char_vars].to_dict(orient = 'records'))
    return sparse_matrix

# save and load sparse matrix
def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices, \
                indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), \
                        shape = loader['shape'])


############################################################################################
################################ STEP 01 Load in Data and Merges ###########################
############################################################################################

# Load Data and EDA
property_df = pd.read_csv(os.path.join(data_dir, propertcsv))
train_df = pd.read_csv(os.path.join(data_dir, traincsv))
submission_df = pd.read_csv(os.path.join(data_dir, submissioncsv))


## Step 01: Melt the submission file into the shape of the train
submission_df["rowid"] = range(submission_df.shape[0])
test_df = pd.melt(submission_df,id_vars=["ParcelId", "rowid"])
test_df.columns = ["parcelid", "rowid", "transactiondate", "logerror"]
test_df["logerror"] = np.nan
test_df["transactiondate"] = converttestchar2date(test_df["transactiondate"].astype("object"))
test_df = test_df.sort_values(['rowid', 'transactiondate'], ascending=[True, True])
test_df.drop(["rowid"], axis=1, inplace = True)
test_df = test_df[["parcelid", "logerror", "transactiondate"]]

full = pd.concat([train_df, test_df])
full["rowidx"] = range(full.shape[0])
full["split"] = "train"
full.loc[np.isnan(full['logerror']), ["split"]] = "test"

itemfreq(full.split.values)


### Load in Train Data
full = pandas_fast_merge(full, property_df, how = "left")
full = full.sort_values(["rowidx"], ascending=[True])
char_columns = pd.read_csv(os.path.join("char_vars_in_originaldata.csv"))["Features"].values
num_columns = pd.read_csv(os.path.join("num_vars_in_originaldata.csv"))["Features"].values
num_columns_to_log = pd.read_csv(os.path.join("num_vars_to_log.csv"))["Features"].values
variable_importance_orig_columns = pd.read_csv(os.path.join("Var_Importance.csv"))["Feature"].values
non_predictors = ["parcelid", "logerror", "split", "transactiondate", "rowidx"]

del test_df, train_df
gc.collect()


### Conver char variable to the object type
full = convert_column_to_object(full, char_columns)
full = add_log_to_num_columns(full, num_columns_to_log)


### Substring some char columns
char_columns_ = ["censustractandblock", "censustractandblock", "propertyzoningdesc", \
                    "propertyzoningdesc", "propertycountylandusecode",
                    "censustractandblock",
                    "propertyzoningdesc", "censustractandblock", "censustractandblock"]
ranges_ = ([0,12], [0,8], [0,3], [0,4], [0,2], [0,4], [4,10], [4,11], [11,12])
for i in range(len(char_columns_)):
    char_ = char_columns_[i]
    range_ = ranges_[i]
    print("Processing variable {} for range {}".format(char_, range_))
    full = substring_columns(full, column = char_, substring_range = range_)
    gc.collect()


## Special Processing on Some Vars
#full["transactionDayofYear"] = pd.to_datetime(full.transactiondate, format="%Y-%m-%d").dt.dayofyear
#full["transactionDayofYear2"] = np.sin(full["transactionDayofYear"] * np.pi / 365.25)
#full["transactionDayofYear3"] = np.cos(full["transactionDayofYear"] * np.pi / 365.25)
#full["transactionDayofYear"] = np.log(pd.to_datetime(full.transactiondate, format="%Y-%m-%d").dt.year + \
#                                full["transactionDayofYear"].values / 365.25)
full["transactionYear"] = pd.to_datetime(full.transactiondate, format="%Y-%m-%d").dt.year
full["numberofstories_log"] = np.log(np.float64(full["numberofstories"].values) + 1.0)
full["yearbuilt_log"] = np.log(np.float64(full["yearbuilt"].values))
full["transactionMonth"] = pd.to_datetime(full.transactiondate, format="%Y-%m-%d").dt.month
#full["transactionMonth2"] = np.sin(full["transactionMonth"] * np.pi / 12.0)
#full["transactionMonth3"] = np.cos(full["transactionMonth"] * np.pi / 12.0)
full['rawcensustractandblock'] = np.float64(full['rawcensustractandblock'].values)
full['censustractandblock'] = np.float64(full['censustractandblock'].values)
full['censustractandblock_1to12'] = np.float64(full['censustractandblock_1to12'].values)
full = convert_column_to_object(full, ["transactionMonth"])

# check if any predictors missing
print("Some important variables still not in full dataframe are {}".format(list(np.setdiff1d(variable_importance_orig_columns, full.columns.values))))

predictors = np.intersect1d(full.columns.values,
                            list(variable_importance_orig_columns)+["censustractandblock_5to11", "censustractandblock_12to12", "transactionYear"])
full = full[list(non_predictors) + list(predictors)]

gc.collect()
if False:
    full.to_csv("full.csv", index=None)
    full.to_pickle("full_step01.p")
    full = pd.read_pickle("full_step01.p")






##############################################################################################################
##############################################################################################################
##############################################################################################################
#   REST OF THE FOLLOWING IS NOT USED!!!
##############################################################################################################
##############################################################################################################
##############################################################################################################
######## STEP 02 SparseMatrix Creation ###############
gc.collect()
char_vars = list(np.intersect1d(np.array(predictors), full.columns.values[full.dtypes == object]))
# check the char varaiables that have more than 5000 levels
for col in char_vars:
    n_ = len(full[col].unique())
    if n_ > 1500: print("Warnings: Variable {} has {} distinct levels!!".format(col, n_))
    # Warnings: Variable censustractandblock has 96772 distinct levels!!
    # Warnings: Variable censustractandblock_1to12 has 8666 distinct levels!!
    # Warnings: Variable propertyzoningdesc has 5639 distinct levels!!
    # Warnings: Variable propertyzoningdesc_1to4 has 1687 distinct levels!!
    # Warnings: Variable rawcensustractandblock has 99394 distinct levels!!
char_vars = list(np.setdiff1d(np.array(char_vars), \
                np.array(["propertyzoningdesc", \
                          "censustractandblock", \
                          "censustractandblock_1to12", \
                          "rawcensustractandblock"])))
# some variables have too many levels listed above and need to be excluded
full_char_sparse = convert_pd_to_sparse_matrix3(full[char_vars])
if False:
    save_sparse_csr("full_char_sparse_step01", full_char_sparse)
    gc.collect()
# pd.DataFrame({"sparse_features":full_char_sparse_columns}).to_csv("s1_sparse_feature_names.csv", index=False)

## SVD Creation
svd_ncomponents = 300
svd = TruncatedSVD(n_components=min(svd_ncomponents, min(full_char_sparse.shape)-1), \
                   algorithm='arpack', \
                   random_state=10011)
# svd.fit(full_char_sparse)
full_char_svd = np.asmatrix(svd.fit_transform(full_char_sparse))  # convert to dense matrix after SVD
print("Total Variance Explained by SVD-{} is ".format(svd_ncomponents) + str(svd.explained_variance_ratio_.sum()))
# total variance is explained at 84.3% for 300SVD
# Scale the svd matrix
scaler = StandardScaler()
scaler.fit(full_char_svd)
full_char_svd = np.asmatrix(scaler.transform(full_char_svd))
full_char_svd = full_char_svd.astype(np.float32)  # save memory
if False:
    np.save('full_char_svd_step01', full_char_svd)
    full_char_svd = np.load('full_char_svd_step01.npy')


######## STEP 03 Matrix for the Numerical Variables ###############
num_vars = list(np.setdiff1d(np.array(predictors), np.array(list(char_vars) + ["propertyzoningdesc"])))
full_numerical_var_matrix = full[num_vars]
# two special treatments on two variables since their starndard deviations are 0
# due to its massive amount of misisng values nan
full_numerical_var_matrix["poolcnt"] = full_numerical_var_matrix["poolcnt"].fillna(0)
full_numerical_var_matrix["poolcnt_log"] = full_numerical_var_matrix["poolcnt_log"].fillna(0.1)
full_numerical_var_matrix = np.asmatrix(full_numerical_var_matrix.values).astype(np.float64)
scaler4 = StandardScaler2()
scaler4.fit(full_numerical_var_matrix)
full_numerical_var_matrix = scaler4.transform(full_numerical_var_matrix)
full_numerical_var_matrix = full_numerical_var_matrix.astype(np.float32)
if False:
    np.save('full_numerical_var_matrix_step01', full_numerical_var_matrix)
    full_numerical_var_matrix = np.load('full_numerical_var_matrix_step01.npy')

print(char_vars)
print(num_vars)


#########################################################################
###############  Create the Score File for Submission   #################
#########################################################################
submission_file = full.loc[full.split=="test", ["parcelid", "transactiondate", "logerror"]]
submission_file = submission_file.sort_values(["parcelid", "transactiondate"], ascending=[True, True])
submission_file = submission_file.set_index(['parcelid','transactiondate']).unstack().reset_index()
submission_file.columns = [submission_file.columns.droplevel(1)[0]] \
                        + list(submission_file.columns.droplevel(0)[1:])
submission_file = submission_file.rename(columns={"parcelid": "ParcelId"})
# use the submission_df file that was created above that has the rowid to assign order
submission_file = pandas_fast_merge(submission_file, submission_df[["ParcelId", "rowid"]])
submission_file = submission_file.sort_values(["rowid"], ascending=[True])
submission_file.drop(["rowid"], axis=1, inplace=True)
submission_file.rename(columns={"2016-10-15": "201610", \
                                "2016-11-15": "201611", \
                                "2016-12-15": "201612", \
                                "2017-10-15": "201710", \
                                "2017-11-15": "201711", \
                                "2017-12-15": "201712", \
                                }, \
                        inplace=True)
subfilename = 'submission_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
submission_file.to_csv(subfilename, index=False)



###### LEGACY CODE
if False:
    # char and numerical predictors
    char_vars = ['airconditioningtypeid',
                'buildingclasstypeid',
                'buildingqualitytypeid',
                'censustractandblock_1to4',
                'censustractandblock_1to8',
                'hashottuborspa',
                'heatingorsystemtypeid',
                'pooltypeid7',
                'propertycountylandusecode',
                'propertycountylandusecode_1to2',
                'propertylandusetypeid',
                'propertyzoningdesc_1to3',
                'propertyzoningdesc_1to4',
                'propertyzoningdesc_5to10',
                'regionidcity',
                'regionidneighborhood',
                'regionidzip',
                'taxdelinquencyflag',
                'transactionMonth',
                'yearbuilt']
    num_vars = ['bathroomcnt',
                'bathroomcnt_log',
                'bedroomcnt',
                'bedroomcnt_log',
                'calculatedbathnbr',
                'calculatedbathnbr_log',
                'calculatedfinishedsquarefeet',
                'calculatedfinishedsquarefeet_log',
                'censustractandblock',
                'censustractandblock_1to12',
                'finishedfloor1squarefeet',
                'finishedfloor1squarefeet_log',
                'finishedsquarefeet12',
                'finishedsquarefeet12_log',
                'finishedsquarefeet15',
                'finishedsquarefeet15_log',
                'finishedsquarefeet50',
                'finishedsquarefeet50_log',
                'finishedsquarefeet6',
                'fullbathcnt',
                'fullbathcnt_log',
                'garagetotalsqft',
                'garagetotalsqft_log',
                'landtaxvaluedollarcnt',
                'landtaxvaluedollarcnt_log',
                'latitude',
                'latitude_log',
                'longitude',
                'longitude_log',
                'lotsizesquarefeet',
                'lotsizesquarefeet_log',
                'numberofstories_log',
                'poolcnt',
                'poolcnt_log',
                #'propertyzoningdesc',
                'rawcensustractandblock',
                'roomcnt',
                'roomcnt_log',
                'structuretaxvaluedollarcnt',
                'structuretaxvaluedollarcnt_log',
                'taxamount',
                'taxamount_log',
                'taxdelinquencyyear',
                'taxdelinquencyyear_log',
                'taxvaluedollarcnt',
                'taxvaluedollarcnt_log',
                'threequarterbathnbr',
                'threequarterbathnbr_log',
                'transactionDayofYear',
                'transactionDayofYear2',
                'transactionDayofYear3',
                'transactionMonth2',
                'transactionMonth3',
                'unitcnt',
                'unitcnt_log',
                'yardbuildingsqft17',
                'yearbuilt_log']

if False:
    # ICA Components Creation
    fastica = FastICA(n_components=svd_ncomponents, random_state=10011)
    full_char_ica700 = np.asmatrix(fastica.fit_transform(full_char_svd))
    scaler3 = StandardScaler()
    scaler3.fit(full_char_ica700)
    full_char_ica700 = np.asmatrix(scaler.transform(full_char_ica700))
    full_char_ica700 = full_char_ica700.astype(np.float32)  # save memory
    if False:
        np.save('full_char_ica700_step01', full_char_ica700)
        full_char_ica700 = np.load('full_char_ica700_step01.npy')

if False:
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

