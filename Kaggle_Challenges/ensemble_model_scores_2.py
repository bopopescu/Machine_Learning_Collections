import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

print(train.head())

### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, test_size=0.05, random_state=1234)
for train_index, test_index in sss:
    break

train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]

### building the classifiers
clfs = []

rfc = RandomForestClassifier(n_estimators=50, random_state=4141, n_jobs=-1)
rfc.fit(train_x, train_y)
print('RFC LogLoss {score}'.format(score=log_loss(test_y, rfc.predict_proba(test_x))))
clfs.append(rfc)

### usually you'd use xgboost and neural nets here

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
print('LogisticRegression LogLoss {score}'.format(score=log_loss(test_y, logreg.predict_proba(test_x))))
clfs.append(logreg)

rfc2 = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
rfc2.fit(train_x, train_y)
print('RFC2 LogLoss {score}'.format(score=log_loss(test_y, rfc2.predict_proba(test_x))))
clfs.append(rfc2)


### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(test_y, final_prediction)

#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))


#############################################################################################
# Method 2
#############################################################################################

from scipy.optimize import minimize
##This function is the one that will be minimized with respect to w, the model weights
def fun(w,probs,y_true):
sum = 0
for i in range(len(probs)):
sum+= probs[i]*w[i]
return logloss_mc(y_true,sum)
## First save your model probabilities like so:
probs=[p1,p2,p3,p4]
## w0 is the initial guess for the minimum of function 'fun'
## This initial guess is that all weights are equal
w0 =np.ones(len(probs))/(len(probs))
## This sets the bounds on the weights, between 0 and 1
bnds = tuple((0,1) for w in w0)
## This sets the constraints on the weights, they must sum to 1
## Or, in other words, 1 - sum(w) = 0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
## Calls the minimize function from scipy.optimize
## -----------------------------------------------
## fun is the function defined above
## w0 is the initial estimate of weights
## (probs, y_true) are the additional arguments passed to 'fun'; probs are the probabilities,
## y_true is the expected output for your training set
## method = 'SLSQP' is a least squares method for minimizing the function;
## There are other methods available, but I don't know enough about the theory to make recommendations ## ---------------------------------------------------
weights = minimize(fun,x0,(probs,y_true),method='SLSQP',bounds=bnds,constraints=cons).w
## As a sanity check, make sure the weights do in fact sum to 1
print("Weights sum to %0.4f:" % weights.sum())
## Print out the weights
print(weights)
## This will combine the model probabilities using the optimized weights
y_prob = 0
for i in range(len(probs)):
    y_prob += probs[i]*weights[i]
