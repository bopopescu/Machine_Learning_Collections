load("PoissonExample.RData")

library(glmnet)
library(hydroGOF)

fit = glmnet(x, y, family = "poisson", standardize = TRUE, intercept=TRUE);
cvfit = cv.glmnet(x, y, family = "poisson", standardize = TRUE, intercept=TRUE, nfolds=10);
plot(cvfit);
opt.lam = c(cvfit$lambda.min, cvfit$lambda.1se);
coef(cvfit, s = opt.lam);
pred = predict(cvfit, newx = x, type = "response", s = c(opt.lam));
pred = cbind(y, pred);
colnames(pred) = c("y", "pred_lambda_min", "pred_lambda_lse");
rmse_lambda_min = rmse(pred[,1], pred[,2]);
rmse_lambda_lse = rmse(pred[,1], pred[,3]);

###########
# shorter x matrix for demo 
########### 

x = x[,c(1:2)];
fit = glmnet(x, y, family = "poisson");
cvfit = cv.glmnet(x, y, family = "poisson", nfolds=10);
plot(cvfit);
opt.lam = c(cvfit$lambda.min, cvfit$lambda.1se);
coef(cvfit, s = opt.lam);
pred = predict(cvfit, newx = x, type = "response", s = c(cvfit$lambda.min));
pred = cbind(y, pred);
colnames(pred) = c("y", "pred_lambda_min");
rmse_lambda_min = rmse(pred[,1], pred[,2]);
coef_lambda_min = coef(cvfit, s = c(cvfit$lambda.min));
coef_lambda_min;
coef_lambda_min = as.matrix(coef_lambda_min);
x_new = cbind(1,x);
pred_new = exp(x_new %*% coef_lambda_min) 

# This demostrates that glmnet uses link function log automatically # when loss distribution is poisson

################################################
# xgboost sample
################################################
library(glmnet)
library(Matrix)
require(xgboost)
require(Matrix)
require(data.table)
require(gbm)

data(PoissonExample)

trn <- sample(seq_len(500),size=350)
trn <- sort(trn)
tst <- sort(setdiff(seq_len(500), trn))


### Elastic Net ####
fit1 <- cv.glmnet(x[trn,], y[trn], nfolds = 80, family='poisson')
print(fit1$lambda.min)
rmse <- sqrt(mean((y[tst] - predict(fit1, x[tst, ],s=fit1$lambda.min, type="response"))**2))

### XGBoost ####
sparse_matrix <- sparse.model.matrix(~., data = as.data.frame(x))[,-1]
colnames(sparse_matrix) <- colnames(as.data.frame(x))
head(sparse_matrix)

train <- list(data=sparse_matrix[trn, ], label=y[trn])
test <- list(data=sparse_matrix[tst, ], label=y[tst])
dtrain <- xgb.DMatrix(data=sparse_matrix[trn, ], label = y[trn])
dtest <- xgb.DMatrix(data=sparse_matrix[tst, ], label = y[tst])

param <- list(objective = "count:poisson", booster = "gblinear", #default bootster is gbtree, for classification
              nthread = 4, alpha = 1, lambda = 1e-3,  max.depth = 5,
              nfold=3,  eta=0.01,  #eta is like gbm learning rate
              silent=1, eval_metric="rmse")
# normally, you do not need to set eta (step_size)
# XGBoost uses a parallel coordinate descent algorithm (shotgun),
# there could be affection on convergence with parallelization on certain cases
# setting eta to be smaller value, e.g 0.5 can make the optimization more stable

##
# the rest of settings are the same
##
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 10000
bst <- xgb.train(param, data=dtrain, nrounds=num_round, watchlist=watchlist,
                early.stop.round=10,
                print.every.n=100)

#xgb.dump(bst, "dump.raw.txt", with.stats = T)
# variable importance only works in gbtree
#print("Most important features (look at column Gain):")
#imp_matrix <- xgb.importance(feature_names = colnames(train$data), model = bst)
#print(imp_matrix)
pred <- predict(bst, dtest)
rmse2 <- sqrt(mean((getinfo(dtest, "label") - round(pred))**2))

if (FALSE) {
    # user define objective function, given prediction, return gradient and second order gradient
    # this is loglikelihood loss
    logregobj <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      preds <- 1/(1 + exp(-preds))
      grad <- preds - labels
      hess <- preds * (1 - preds)
      return(list(grad = grad, hess = hess))
    }
    # user defined evaluation function, return a pair metric_name, result
    # NOTE: when you do customized loss function, the default prediction value is margin
    # this may make buildin evalution metric not function properly
    # for example, we are doing logistic loss, the prediction is score before logistic transformation
    # the buildin evaluation error assumes input is after logistic transformation
    # Take this in mind when you use the customization, and maybe you need write customized evaluation function
    evalerror <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
      return(list(metric = "error", value = err))
    }
    param <- list(max_depth=2, eta=1, nthread = 2, silent=1,
                  objective=logregobj, eval_metric=evalerror)
}
### GBM ####
dt <- cbind(data.frame(x), y=y)
formula <- as.formula(paste0("y ~ ", paste0(colnames(data.frame(x)), collapse="+")))
fit3 <- gbm(formula,
                data=dt[trn, ],
                distribution = "poisson", n.trees=10000, n.minobsinnode = 2, interaction.depth = 10,
                cv.folds=10)
best.iter <- gbm.perf(fit3,method="cv")
pred <- predict(fit3, dt[tst, ], n.trees=best.iter, type="response")

rmse3 <- sqrt(mean((y[tst] - pred)**2))

################################################
# Feature Hashing Trick for GLMNET
################################################
# get data ----------------------------------------------------------------
# UCI Diabetes 130-US hospitals for years 1999-2008 Data Set 
# https://archive.ics.uci.edu/ml/machine-learning-databases/00296/
require(RCurl)
binData <- getBinaryURL("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
                    ssl.verifypeer=FALSE)

conObj <- file("dataset_diabetes.zip", open = "wb")
writeBin(binData, conObj)
# don't forget to close it
close(conObj)

# open diabetes file
files <- unzip("dataset_diabetes.zip")
diabetes <- read.csv(files[1], stringsAsFactors = FALSE)

# quick look at the data
str(diabetes)

# drop useless variables
diabetes <- subset(diabetes,select=-c(encounter_id, patient_nbr))

# transform all "?" to 0s
diabetes[diabetes == "?"] <- NA

# remove zero variance - ty James http://stackoverflow.com/questions/8805298/quickly-remove-zero-variance-variables-from-a-data-frame
diabetes <- diabetes[sapply(diabetes, function(x) length(levels(factor(x,exclude=NULL)))>1)]

# prep outcome variable to those readmitted under 30 days
diabetes$readmitted <- ifelse(diabetes$readmitted == "<30",1,0)

# generalize outcome name
outcomeName <- 'readmitted'

# large factors to deal with
length(unique(diabetes$diag_1))
length(unique(diabetes$diag_2))
length(unique(diabetes$diag_3))

# dummy var version -------------------------------------------------------
diabetes_dummy <- diabetes
# alwasy a good excersize to see the length of data that will need to be transformed
# charcolumns <- names(diabetes_dummy[sapply(diabetes_dummy, is.character)])
# for (thecol in charcolumns) 
#         print(paste(thecol,length(unique(diabetes_dummy[,thecol]))))

# warning will need 2GB at least free memory
require(caret)
dmy <- dummyVars(" ~ .", data = diabetes_dummy)
diabetes_dummy <- data.frame(predict(dmy, newdata = diabetes_dummy))

# many features
dim(diabetes_dummy)

# change all NAs to 0
diabetes_dummy[is.na(diabetes_dummy)] <- 0

# split the data into training and testing data sets
set.seed(1234)
split <- sample(nrow(diabetes_dummy), floor(0.5*nrow(diabetes_dummy)))
objTrain <-diabetes_dummy[split,]
objTest <- diabetes_dummy[-split,]

predictorNames <- setdiff(names(diabetes_dummy),outcomeName)

# cv.glmnet expects a matrix 
library(glmnet)
# straight matrix model not recommended - works but very slow, go with a sparse matrix
# glmnetModel <- cv.glmnet(model.matrix(~., data=objTrain[,predictorNames]), objTrain[,outcomeName], 
#             family = "binomial", type.measure = "auc")

glmnetModel <- cv.glmnet(sparse.model.matrix(~., data=objTrain[,predictorNames]), objTrain[,outcomeName], 
                         family = "binomial", type.measure = "auc")
glmnetPredict <- predict(glmnetModel,sparse.model.matrix(~., data=objTest[,predictorNames]), s="lambda.min")

# dummy version score:
auc(objTest[,outcomeName], glmnetPredict)

# feature hashed version -------------------------------------------------
diabetes_hash <- diabetes
predictorNames <- setdiff(names(diabetes_hash),outcomeName)

# change all NAs to 0
diabetes_hash[is.na(diabetes_hash)] <- 0

set.seed(1234)
split <- sample(nrow(diabetes_hash), floor(0.5*nrow(diabetes_hash)))
objTrain <-diabetes_hash[split,]
objTest <- diabetes_hash[-split,]
 
library(FeatureHashing)
objTrain_hashed = hashed.model.matrix(~., data=objTrain[,predictorNames], hash.size=2^12, transpose=FALSE)
objTrain_hashed = as(objTrain_hashed, "dgCMatrix")
objTest_hashed = hashed.model.matrix(~., data=objTest[,predictorNames], hash.size=2^12, transpose=FALSE)
objTest_hashed = as(objTest_hashed, "dgCMatrix")
 
library(glmnet)
glmnetModel <- cv.glmnet(objTrain_hashed, objTrain[,outcomeName], 
                     family = "binomial", type.measure = "auc")
# hashed version score:
glmnetPredict <- predict(glmnetModel, objTest_hashed, s="lambda.min")
auc(objTest[,outcomeName], glmnetPredict)
