# -*- coding: utf-8 -*-
"""
@author: pm
This is a Day-ahead regression task using H2O Deep Learning

The details of H2O usage are on its booklets from its website
"""

import os, numpy as np
import pandas as pd
import sklearn
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


try:
    # shutdown the H2O instance running at localhost:54321 In case there is one running
    h2o.shutdown(prompt=False)
except:
    pass

# Start H2O cluster with all available cores (default)
h2o.init()
# h2o.init(ip = "127.0.0.1", port = 54321) # default setting

## Load Data by numpy
data_np = np.genfromtxt('test_data/Data_for_ShortTerm_Forecast.csv', 
                        delimiter=",", skip_header=0, dtype="|S5")

## Load Data by Pandas
data_pd = pd.read_csv('test_data/Data_for_ShortTerm_Forecast.csv')
# select train partition of data
train_pd = data_pd[data_pd['Partition']=='Train']
del train_pd['Partition']
# get_dummies convert Monday, Friday values of Column Weekday to 0,0,...,1, dummry
    # 7 columns
weekday_colmn = pd.get_dummies(train_pd['Weekday'])
del train_pd['Weekday']
train_pd = pd.concat([train_pd, weekday_colmn], axis=1)
## Same Manipulation for Test Data
test_pd = data_pd[data_pd['Partition']=='Test']
del test_pd['Partition']
weekday_colmn = pd.get_dummies(test_pd['Weekday'])
del test_pd['Weekday']
test_pd = pd.concat([test_pd, weekday_colmn], axis=1)

train_pd.to_csv('train.csv')
test_pd.to_csv('test.csv')

# Load Data into H2O Data Frame
train = h2o.import_file(os.path.join(os.getcwd(),"train.csv"),sep=",", col_names=0)
test = h2o.import_file(os.path.join(os.getcwd(),"test.csv"),sep=",", col_names=0)
# Remove temp data
os.remove('train.csv')
os.remove('test.csv')

train.describe()
# temp = train.as_data_frame() # check out what h2o data frame is like in dataframe
####### H2O Deep Learning for Regression #####

# Specify the response and predictor columns
all_col_names = train.names
#y = "C785" # the last column, C785
y = all_col_names[1] #since all_col_names[0] is list of indices
x = train.names[2:]

##############################################################################
# Train Deep Learning model and validate on test set
model = H2ODeepLearningEstimator(distribution="laplace",
                classification=False,
                activation="RectifierWithDropout",
                hidden=[200,200, 200, 200, 200, 200],
                diagnostics=True,
                input_dropout_ratio=0.2, #regularization
                sparse=False, # since the majority data values are 0
                l1=1e-5, # L1 regularization value
                epochs=10, # number of epochs to run
                nfolds = 5,
                variable_importances=True) # show variable importance

model.train(x=x, y=y, training_frame=train, validation_frame=test)

# View specified parameters of the Deep Learning model
model.params

# Examine the performance of the trained model
model # display all performance metrics

model.model_performance(train=True) # training metrics
model.mse(valid=True) # get Mean Squared Error only


