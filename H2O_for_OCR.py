# -*- coding: utf-8 -*-
"""
@author: pm

This is a MNIST Digital Handwriting example in Python
"""

import os
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# Start H2O cluster with all available cores (default)
h2o.init()

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

# Specify the response and predictor columns

all_col_names = train.names
#y = "C785" # the last column, C785
y = all_col_names[-1]
x = train.names[0:784]

# Encode the response column as categorical for multinomial classification
    # for Categorical type of problems, change them into factors, as in R
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


##########################
# Train Deep Learning model and validate on test set
model = H2ODeepLearningEstimator(distribution="multinomial",
            activation="RectifierWithDropout",
            hidden=[200,128,32, 10],
            input_dropout_ratio=0.25, #regularization
            sparse=True, # since the majority data values are 0
            l1=1e-5, # L1 regularization value
            epochs=10) # number of epochs to run

model.train(x=x, y=y, training_frame=train, validation_frame=test)
#############

##########################
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


# View specified parameters of the Deep Learning model
model.params

# Examine the performance of the trained model
model # display all performance metrics

model.model_performance(train=True) # training metrics
model.model_performance(valid=True) # validation metrics


# Get MSE only
model.mse(valid=True)

# Cross-validated MSE
model_cv.mse(xval=True)


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
