# -*- coding: utf-8 -*-
"""
@author: pm

This is a Day-ahead regression task using H2O Deep Learning

The Weighted MAPE value on testing is 0.017739

The details of H2O usage are on its booklets from its website.

More details can be found from source codes on GitHub
https://github.com/h2oai/h2o-3/blob/master/h2o-py/h2o/h2o.py
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

data_file = 'test_data/Data_for_ShortTerm_Forecast.csv'

## Load Data by numpy
data_np = np.genfromtxt(data_file, delimiter=",", skip_header=0, dtype="|S5")

## Load Data by Pandas
data_pd = pd.read_csv(data_file)
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

train_pd.to_csv('train_.csv')
test_pd.to_csv('test_.csv')

# Load Data into H2O Data Frame
train = h2o.import_file(os.path.join(os.getcwd(),"train_.csv"),sep=",", col_names=0)
test = h2o.import_file(os.path.join(os.getcwd(),"test_.csv"),sep=",", col_names=0)
# Remove temp data
os.remove('train_.csv')
os.remove('test_.csv')

train.describe()
# temp = train.as_data_frame() # check out what h2o data frame is like in dataframe
####### H2O Deep Learning for Regression #####

# Specify the response and predictor columns
all_col_names = train.names
#y = "C785" # the last column, C785
y = all_col_names[1] #since all_col_names[0] is list of indices
x = train.names[2:]
ncols = len(x)

##############################################################################
# Train Deep Learning model and validate on test set
# H2O Deep Learning supports Poisson, Gamma, Tweedie and Laplace distributions. It also supports Absolute and Huber loss 

model = H2ODeepLearningEstimator(
                model_id="regression_model_dayahead",
                distribution="laplace",
                loss = "Absolute",
                activation="Rectifier", #other options are Rectifier, Tanh, TanhWithDropout, RectifierWithDropout
                hidden=[ncols, ncols, ncols, ncols, ncols, ncols, ncols, ncols],
                adaptive_rate = True, # use ADADELTA for learning rate
                input_dropout_ratio=0.05, #regularization
                sparse=False, # since the majority data values are 0
                l1=1e-4, # L1 regularization value
                epochs=10000, # number of epochs to run
                nfolds = 7,
                variable_importances=True) # show variable importance

model.train(x=x, y=y, training_frame=train, validation_frame=test)

## Check the weighted Mape value
def wMape(a,b):
    # a,b are predicted and original y values
    # b is the original y values
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = np.abs(a-b)
    return np.sum(diff) / (np.sum(b) + 1e-5)

pred = model.predict(test)
test_pred =pred.as_data_frame()[0][1:]
test_y = test[y].as_data_frame()[0][1:]
print "Weighted MAPE value on Testing is %f" %wMape(test_pred, test_y)

pred = model.predict(train)
train_pred =pred.as_data_frame()[0][1:]
train_y = train[y].as_data_frame()[0][1:]
print "Weighted MAPE value on Training is %f" %wMape(train_pred, train_y)
# View specified parameters of the Deep Learning model
model.params

# Examine the performance of the trained model
model # display all performance metrics
model.model_performance(train=True) # training metrics
model.model_performance(valid=True) # training metrics
model.mse(valid=True) # get Mean Squared Error only
model.mse(train=True)

# Show top 20 variable importance
model.varimp()[:20]
###############


# The following is to use grid-search to find the best parameters #
##############################################################################
hidden_opt = [[ncols, ncols, ncols],[2*ncols, 2*ncols, 2*ncols, 2*ncols]] #hidden layer structures to test
l1_opt = [1e-4,1e-3, 1e-5] # l1 regularization test

hyper_parameters = {"hidden":hidden_opt, "l1":l1_opt} # tells model to use hidden layers and l1 regularization

from h2o.grid.grid_search import H2OGridSearch

model_grid = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)

model_grid.train(x=x, y=y, distribution="laplace", epochs=1000, 
                loss = "Absolute", activation="Rectifier",
                training_frame=train, validation_frame=test, #stopping_rounds=10,
                stopping_tolerance=0.001, stopping_metric="MSE")

# print model grid search results
model_grid

mse_min = 1e30
for model in model_grid:
    mse = model.mse()
    if mse < mse_min:
        mse_min = mse+0
        best_model = model
    print model.model_id + " mse: " + str(model.mse())

print "Smallest MSE from candidnate models is %i" %best_model.mse()
print "hidden:",best_model.full_parameters['hidden']['actual_value'], " ; l1",best_model.full_parameters['l1']['actual_value']

