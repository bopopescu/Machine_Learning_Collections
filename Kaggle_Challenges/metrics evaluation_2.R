# You can write R code here and then click "Run" to run it on our platform

# The readr library is the best way to read and write CSV files in R
library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:

library(caTools)
library(caret)
library(Metrics)
library(doParallel)
library(xgboost)
library(Matrix)

train  = read.csv("../input/train.csv",stringsAsFactors = TRUE)
test = read.csv("../input/test.csv",stringsAsFactors = TRUE)


options(mc.cores = 10)

colnames(train)<- make.names(colnames(train))
colnames(test)<- make.names(colnames(test))


# 
for (i in 2:ncol(train) - 1){
  if(is.numeric(train[,i])){
    train = cbind(train,log(train[,i]+1))
    colnames(train)[ncol(train)] <- paste("log",names(train)[i],sep = "")}
}
for (i in 2:ncol(test)){
  if(is.numeric(test[,i])){
    test = cbind(test,log(test[,i]+1))
    colnames(test)[ncol(test)] <- paste("log",names(test)[i],sep = "")}
}



###### REQUIRES package qdapTools  #####
library(qdapTools)
factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))))
  rownames(temp) <- NULL
  
  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

### Returns mean, median, and sd for factor T1_V4 by factor level to be added to the training set.
#data.frame(head(train$T1_V4))
#head(factorToNumeric(train, train, "Hazard", "T1_V4", c("mean","median","sd","mad")))
for (i in 1:ncol(train)){
  if(is.factor(train[,i])){train = cbind(train,factorToNumeric(train, train, "Hazard", names(train)[i], c("mean","median","sd","mad")))}
}
for (i in 1:ncol(test)){
  if(is.factor(test[,i])){test = cbind(test,factorToNumeric(train, test, "Hazard", names(test)[i], c("mean","median","sd","mad")))}
}
for (i in 1:ncol(train)){
  if(is.factor(train[,i])){train[,i] = as.ordered(x = train[,i])}
}
for (i in 1:ncol(test)){
  if(is.factor(test[,i])){test[,i] = as.ordered(x = test[,i])}
}
for (i in 1:ncol(train)){
  if(is.ordered(train[,i])){
    train = cbind(train,as.numeric(train[,i]))
    colnames(train)[ncol(train)] <- paste(names(train)[i],"asnum",sep = "")}
}
for (i in 1:ncol(test)){
  if(is.ordered(test[,i])){
    test = cbind(test,as.numeric(test[,i]))
    colnames(test)[ncol(test)] <- paste(names(test)[i],"asnum",sep = "")}
}
train = as.data.frame(model.matrix(~.-1,data = train))
test = as.data.frame(model.matrix(~.-1,data = test))

colnames(train)<- make.names(colnames(train))
colnames(test)<- make.names(colnames(test))

# train = as.data.frame(model.matrix( ~ .-1, data=train, contrasts.arg = 
#               lapply(data.frame(train[,sapply(data.frame(train), is.factor)]),
#                      contrasts, contrasts = TRUE)))
# 
# test = as.data.frame(model.matrix( ~ .-1, data=test, contrasts.arg = 
#                   lapply(data.frame(test[,sapply(data.frame(test), is.factor)]),
#                          contrasts, contrasts = FALSE)))
"%w/o%" <- function(x, y) x[!x %in% y] #--  x without y
xx1 = colnames(train) %w/o% c("Id","Hazard")
print(paste0("Nvar predictor",length(xx1)))

nzv <- nearZeroVar(train, saveMetrics= TRUE)
train <- train[, -which(nzv$zeroVar)]
nzvtest <- nearZeroVar(test, saveMetrics = TRUE)
test <- test[,-which(nzvtest$zeroVar)]
xx1 = colnames(train) %w/o% c("Id","Hazard")
print(paste0("Nvar predictor",length(xx1)))

set.seed(244)
split = sample.split(train$Hazard, SplitRatio = 0.8)
predictor = (subset(train, split == TRUE))[,which(colnames(train) %in% xx1)]
response = (subset(train$Hazard, split == TRUE))
valPredictor = (subset(train, split == FALSE))[,which(colnames(train) %in% xx1)]
valResponse = (subset(train$Hazard, split == FALSE))
testpredictor = test[,which(colnames(test) %in% xx1)]

print(paste0("Nvar predictor",ncol(predictor)))

WeightedGini <- function(solution, weights, submission){
  df = data.frame(solution, weights, submission)
  n <- nrow(df)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = cumsum(df$weights/sum(df$weights))
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / df$cumPosFound[n]
  sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
}
NormalizedWeightedGini <- function(solution, weights = rep(1, length = length(solution)), submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}

### Creates a set of new columns for your data set that summarize factor levels by descriptive statistics such as mean,
### sd, skewness etc.

### Variables : train = training set
###             test = test set
###             response = response variable (Hazard in this competition)
###             variables = vector of column names you wise to summarize (T1_V4 for example). Must be strings.
###             metrics = vector of desctriptive statistics you wish to compute for each factor level. Must Be Strings.

### ex: factorToNumeric(train, test, "Hazard", "T1_V4", "mean") will return a column of hazard means by factor level
###     in column T1_V4

### You can specify the test parameter as train to get a column to add to your training set


#####################################################################
#xgbTree caret
#####################################################################

library(caret)
giniSummary <- function (data,
                         lev = NULL,
                         model = NULL) {
  out <- NormalizedWeightedGini(solution = data$obs,submission = data$pred)  
  names(out) <- "NWGini"
  out
}
trainControlxgbTree <- trainControl(method = "repeatedcv",number = 2,repeats = 1,summaryFunction = giniSummary,
                                    verboseIter = TRUE,
                                    preProcOptions = list(thresh = 0.99,ICAcomp = 111)
)

evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  gini <- NormalizedWeightedGini(solution = as.numeric(labels),submission = as.numeric(preds))
  return(list(metric = "NWGini", value = gini))
}
#xgbTree
xgtrain = xgb.DMatrix(data = sparse.model.matrix(~., data = predictor),label = response)
xgval = xgb.DMatrix(data = sparse.model.matrix(~., data = valPredictor),label = valResponse)
xgbTree_caret <- train(x = predictor,
                       y = response,
                       method = "xgbTree",
                       metric = "NWGini",
                       tuneGrid = expand.grid(nrounds = 10*(15:50),
                                              eta = c(0.012),
                                              max_depth = c(8)),                            
                        trControl = trainControlxgbTree,
                       #watchlist = list(val = xgval,train = xgtrain),
                       #feval = evalgini,
                       #eval_metric = "NWGini",
                       #printEveryN = 20,
                       maximize = TRUE,
                       #early.stop.round = 20,
                       min_child_weight = 7,
                       subsample = 0.8,
                        verbose = 1,
                       colsample_bytree =0.8,
                       base_score = 0.5,
                       nthread = 10
)

plot(xgbTree_caret)
predTrainxgbTree_caret = predict(xgbTree_caret,newdata = predictor, type = "raw")
NormalizedWeightedGini(solution = response,submission = predTrainxgbTree_caret)
predValxgbTree_caret = predict(xgbTree_caret,newdata = valPredictor, type = "raw")
NormalizedWeightedGini(solution = valResponse,submission = (predValxgbTree_caret))

impvarxgb = varImp(xgbTree_caret)
impvar = (row.names(impvarxgb$importance))[which(impvarxgb$importance$Overall > 2)]
predictor<- predictor[,which(colnames(predictor) %in% impvar)]
valPredictor<- valPredictor[,which(colnames(valPredictor) %in% impvar)]
testpredictor<- testpredictor[,which(colnames(testpredictor) %in% impvar)]

library(caretEnsemble)

folds = 3 # 5
repeats = 1 # 3
resamp_index = createMultiFolds(response, k = folds,times = repeats)
trControlEnsem = trainControl(method = "repeatedcv",
#                               number = folds,
#                               repeats = repeats,
                              index = resamp_index,
                              savePredictions = TRUE,
                              classProbs = FALSE,
                              returnData = FALSE,
                              verboseIter = TRUE,
                              summaryFunction = giniSummary,
                              preProcOptions = list(thresh = 0.99,ICAcomp = 111))
models <- caretList(x = predictor,
                    y = as.integer(response),
                    trControl = trControlEnsem,
                    tuneList = list(
                      xgbTree1 = caretModelSpec(method = "xgbTree",
                                                tuneGrid = expand.grid(nrounds = 10*(5:60),
                                                                       eta = c(0.01),
                                                                       max_depth = c(8)),                            
                                                min_child_weight = 7,
                                                subsample = 0.8,
                                                colsample_bytree = 0.8,
                                                returnData = FALSE,
                                                base_score = 50,
                                                verbose = 0,
                                                nthread = 10
                                                #preProcess = c("center","scale")
                                                )
                    )
)
#registerDoParallel(makeCluster(3))
plot(models$xgbTree1)
NormalizedWeightedGini(solution = valResponse,submission = predict(models$xgbTree1,valPredictor,type = "raw"))
models[['glmnet1']] <- train(x = predictor, y = as.integer(response),
                             trControl = trControlEnsem,
                             method = "glmnet",
                             tuneGrid = expand.grid(alpha = c(0.8,0.9),
                                                    lambda = 0.001*(1:100)),
                              family = "gaussian",
                             preProcess = c("center","scale")
)
NormalizedWeightedGini(solution = valResponse,submission = predict(models$glmnet1,valPredictor,type = "raw"))
models[['parRF']] <- train(x = predictor, y = as.integer(response),
                             trControl = trControlEnsem,
                             method = "parRF",
                             ntree = 300,
                             tuneGrid = expand.grid(mtry = 11),
                             do.trace = TRUE,
                             nodesize = 5,  
                            sampsize = 10000,
                            returnData = FALSE,
                            replace = TRUE,
                            corr.bias = TRUE,keep.forest = TRUE,
                             preProcess = c("center","scale")
)
NormalizedWeightedGini(solution = valResponse,submission = predict(models$parRF,valPredictor,type = "raw"))
models[['elm1']] <- train(x = predictor, y = as.integer(response),
                          trControl = trControlEnsem,
                          method = "elm",
                          tuneGrid = expand.grid(nhid = c(1500),
                                                 actfun = c("purelin")),
                          preProcess = c("center","scale")
                          
)
NormalizedWeightedGini(solution = valResponse,submission = predict(models$elm1,valPredictor,type = "raw"))

#stack_caret <- caretStack(models, 
 #                         method='xgbTree',
  #                        trControl=trainControl(method='cv',
   #                                              number=2,
    #                                             savePredictions=TRUE,
     #                                            classProbs=FALSE,
      #                                           verboseIter = TRUE,
       #                                          summaryFunction = giniSummary
        #                                         ),
         #                   do.trace = TRUE,
          #                  ntree = 300,
           #               tuneGrid = expand.grid(nrounds = 10*(1:20),
            #                                     eta = c(0.02),
             #                                    max_depth = c(2)),
              #            min_child_weight = 5,
               #           subsample = 0.8,
                #          verbose = 1,
                 #         #colsample_bytree =0.8,
                  #        base_score = 5
                   #       #do.trace = TRUE,
                    #      #tuneLength = 2
#)
#summary(stack_caret$error)
#plot(stack_caret$ens_model)
#predTrainstack_caret = predict(stack_caret,newdata = predictor)
#NormalizedWeightedGini(solution = response,submission = predTrainstack_caret)
#predValstack_caret = predict(stack_caret,newdata = valPredictor)
#NormalizedWeightedGini(solution = valResponse,submission = predValstack_caret)#


model_preds_test <- lapply(models, predict, newdata=testpredictor, type='raw')
model_preds_test <- data.frame(model_preds_test)
model_preds_val <- lapply(models, predict, newdata=valPredictor, type='raw')
model_preds_val <- data.frame(model_preds_val)

ensem_wts = c(1,rep(0,ncol(model_preds_val)-1))
best_ensem_wts = ensem_wts
pred_val_ensem = rowSums(as.data.frame(t(apply(model_preds_val,1,function(x) ensem_wts*x))))/sum(ensem_wts)
best_ensem_NWGini = NormalizedWeightedGini(solution = valResponse,submission = pred_val_ensem)
for( i in 1:1000){
  ensem_wts = runif(ncol(model_preds_val),min = 0, max = 1)
  pred_val_ensem = rowSums(as.data.frame(t(apply(model_preds_val,1,function(x) ensem_wts*x))))/sum(ensem_wts)
  ensem_NWGini = NormalizedWeightedGini(solution = valResponse,submission = pred_val_ensem)
  
  if (ensem_NWGini> best_ensem_NWGini){
    best_ensem_wts <- ensem_wts
    best_ensem_NWGini <- ensem_NWGini
  }
}

print(paste0("best ensem nwgini",best_ensem_NWGini))
print(paste0("best ensem wts",best_ensem_wts))

pred_test_ensem = rowSums(as.data.frame(t(apply(model_preds_test,1,function(x) ensem_wts*x))))/sum(ensem_wts)

PredTest_ensem_caret = data.frame(Id = test$Id, Hazard = pred_test_ensem)
write.csv(PredTest_ensem_caret, "Submission_ensem_caret.csv", row.names= FALSE)
system("gzip --force Submission_ensem_caret.csv")