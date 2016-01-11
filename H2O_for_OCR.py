# -*- coding: utf-8 -*-
"""
@author: pm

This is a MNIST Digital Handwriting example in Python

The details of H2O usage are on its booklets from its website.
Installation of H2O
$ /root/anaconda/bin/pip install requests
$ /root/anaconda/bin/pip install tabulate
$ /root/anaconda/bin/pip uninstall h2o
$ /root/anaconda/bin/pip install http://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/8/Python/h2o-3.6.0.8-py2.py3-none-any.whl

"""

import os
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

# both gz files are CSV files with 785 values on each row, where
    # 784 of them are 28X28 digital images, which can be transformed
    # by the numpy.reshape function easily
    # The last row is the prediction 1,2,3...0


# Load Data from S3
#train = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz")
#test = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz")

# Load Data
train = h2o.import_file(os.path.join(os.getcwd(),"test_data/mnist_csv_files/train.csv.gz"),sep=",")
test = h2o.import_file(os.path.join(os.getcwd(),"test_data/mnist_csv_files/test.csv.gz"),sep=",")

# type(train) shows it is H2O DataFrame format
train.describe()
test.describe()

# To convert train test H2O dataframe to array-like list, use
# test_list = test.as_data_frame()
#     The data will look like [['C1', '0', '0', ......],
#                               ['C2', '0', '0', ......]]
#         where the first C1, C2 are the column names


# Specify the response and predictor columns

all_col_names = train.names
#y = "C785" # the last column, C785
y = all_col_names[-1]
x = train.names[0:784]
ncols = len(x)

# Encode the response column as categorical for multinomial classification
    # for Categorical type of problems, change them into factors, as in R
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


##############################################################################
# Train Deep Learning model and validate on test set
model = H2ODeepLearningEstimator(distribution="multinomial",
            activation="Rectifier", # or RectifierWithDropout
            hidden=[10*ncols, 10*ncols, 10*ncols],
            input_dropout_ratio=0.2, #regularization
            sparse=True, # since the majority data values are 0
            l1=1e-5, # L1 regularization value
            epochs=10000, # number of epochs to run
            variable_importances=True) # show variable importance

model.train(x=x, y=y, training_frame=train, validation_frame=test)

# View specified parameters of the Deep Learning model
model.params

# Examine the performance of the trained model
model # display all performance metrics

model.model_performance(train=True) # training metrics
model.mse(valid=True) # get Mean Squared Error only

# Classify the test set (predict class labels)
# This also returns the probability for each class
pred = model.predict(test)
test_pred =pred.as_data_frame()[0][1:]
test_y = test[y].as_data_frame()[0][1:]
print "Testing accuracy is %f" %(sum(map(lambda t: t[0] != t[1],zip(test_pred,test_y)))*1.0/len(test_y))

# Take a look at the predictions of 10 predictions
pred.head()

# Show prediction and original test data Y values in list
# pred_list = pred.as_data_frame()[0]
# test_list = test.as_data_frame()[-1]

# the model object can not be pickled dump to a file because it is an
    # instance method

# Show top 20 variable importance
model.varimp()[:20]
model_path = h2o.save_model(model, './mnist_dp_model/') # one can save it to local, s3, hdfs
h2o.loadModel(model_path) #model is saved in a folder specifies in model_path
#############


##############################################################################
# Perform 5-fold cross-validation on training_frame
model_cv = H2ODeepLearningEstimator(distribution="multinomial",
                activation="RectifierWithDropout",
                hidden=[32,32,32],
                input_dropout_ratio=0.2,
                sparse=True,
                l1=1e-5,
                epochs=10,
                nfolds=5)

model_cv.train(x=x,y=y,training_frame=train)
model.model_performance(valid=True) # validation metrics
model_cv.mse(xval=True) # Cross-validated MSE
##############


##############################################################################
# Perform a grid-search for best parameter settings
hidden_opt = [[32,32],[32,16,8],[100]] #hidden layer structures to test
l1_opt = [1e-4,1e-3] # l1 regularization test

hyper_parameters = {"hidden":hidden_opt, "l1":l1_opt} # tells model to use hidden layers and l1 regularization

from h2o.grid.grid_search import H2OGridSearch

model_grid = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)

model_grid.train(x=x, y=y, distribution="multinomial", epochs=1000,
                training_frame=train, validation_frame=test, score_interval=2,stopping_rounds=3,
                stopping_tolerance=0.05, stopping_metric="misclassification")

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

##############################################################################
"""
Output is like

ModelMetricsMultinomial: deeplearning
** Reported on train data. **

MSE: 0.2027608619
R^2: 0.975774131703
LogLoss: 0.619453512243

Confusion Matrix: vertical: actual; across: predicted

0    1     2    3     4    5    6     7     8     9     Error      Rate
---  ----  ---  ----  ---  ---  ----  ----  ----  ----  ---------  -------------
976  0     3    3     0    22   8     1     10    1     0.046875   48 / 1,024
0    1043  3    46    0    8    0     2     6     0     0.0586643  65 / 1,108
0    2     831  93    0    5    4     3     23    7     0.141529   137 / 968
0    0     14   989   0    18   0     3     16    1     0.049952   52 / 1,041
0    4     0    6     10   21   34    15    54    814   0.989562   948 / 958
0    0     2    31    0    820  14    0     11    0     0.0660592  58 / 878
0    1     0    1     0    23   956   0     1     0     0.0264766  26 / 982
1    2     2    63    0    0    0     943   13    8     0.0862403  89 / 1,032
0    0     1    25    0    19   1     0     898   0     0.0487288  46 / 944
1    2     0    26    0    13   4     245   98    576   0.403109   389 / 965
978  1054  856  1283  10   949  1021  1212  1130  1407  0.187677   1,858 / 9,900

Top-10 Hit Ratios:
k    hit_ratio
---  -----------
1    0.812323
2    0.93303
3    0.958384
4    0.970202
5    0.978687
6    0.986869
7    0.992626
8    0.996768
9    0.999091
10   1
"""
