setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(gbm)
library(h2o)
library(readr)

# Load in commonly used tools
source("./s0_toolbox.R")

# Common Setting
non_predictors <- c('parcelid', 'transactiondate', 'logerror', 'split', 'random',
                    'foldid', 'logerror_pred', 'rowidx')
predictors <- c('airconditioningtypeid', 'bathroomcnt', 'bathroomcnt_log', 'bedroomcnt', 'bedroomcnt_log',
                'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr', 'calculatedbathnbr_log',
                'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet_log', 'censustractandblock',
                'censustractandblock_12to12', 'censustractandblock_1to12', 'censustractandblock_1to4',
                'censustractandblock_1to8', 'censustractandblock_5to11', 'finishedfloor1squarefeet',
                'finishedfloor1squarefeet_log', 'finishedsquarefeet12', 'finishedsquarefeet12_log',
                'finishedsquarefeet15', 'finishedsquarefeet15_log', 'finishedsquarefeet50', 'finishedsquarefeet50_log',
                'finishedsquarefeet6', 'fullbathcnt', 'fullbathcnt_log', 'garagetotalsqft', 'garagetotalsqft_log',
                'hashottuborspa', 'heatingorsystemtypeid', 'landtaxvaluedollarcnt', 'landtaxvaluedollarcnt_log',
                'latitude', 'latitude_log', 'longitude', 'longitude_log', 'lotsizesquarefeet', 'lotsizesquarefeet_log',
                'numberofstories_log', 'poolcnt', 'poolcnt_log', 'pooltypeid7', 'propertycountylandusecode',
                'propertycountylandusecode_1to2', 'propertylandusetypeid', 'propertyzoningdesc', 'propertyzoningdesc_1to3',
                'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10', 'rawcensustractandblock', 'regionidcity',
                'regionidneighborhood', 'regionidzip', 'roomcnt', 'roomcnt_log', 'structuretaxvaluedollarcnt',
                'structuretaxvaluedollarcnt_log', 'taxamount', 'taxamount_log', 'taxdelinquencyflag',
                'taxdelinquencyyear', 'taxdelinquencyyear_log', 'taxvaluedollarcnt', 'taxvaluedollarcnt_log',
                'threequarterbathnbr', 'threequarterbathnbr_log', 'transactionMonth', 'transactionYear',
                'unitcnt', 'unitcnt_log', 'yardbuildingsqft17', 'yearbuilt', 'yearbuilt_log')
pure_char_vars <- c('airconditioningtypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'censustractandblock',
                'censustractandblock_12to12', 'censustractandblock_1to12', 'censustractandblock_1to4',
                'censustractandblock_1to8', 'censustractandblock_5to11', 'hashottuborspa', 'heatingorsystemtypeid',
                'pooltypeid7', 'propertycountylandusecode', 'propertycountylandusecode_1to2', 'propertylandusetypeid',
                'propertyzoningdesc', 'propertyzoningdesc_1to3', 'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10',
                'rawcensustractandblock', 'regionidcity', 'regionidneighborhood', 'regionidzip', 'taxdelinquencyflag',
                'transactionMonth', 'transactionYear')
num_vars <- c('bathroomcnt', 'bathroomcnt_log', 'bedroomcnt', 'bedroomcnt_log', 'calculatedbathnbr',
                'calculatedbathnbr_log', 'calculatedfinishedsquarefeet', 'calculatedfinishedsquarefeet_log',
                'finishedfloor1squarefeet', 'finishedfloor1squarefeet_log', 'finishedsquarefeet12',
                'finishedsquarefeet12_log', 'finishedsquarefeet15', 'finishedsquarefeet15_log', 'finishedsquarefeet50',
                'finishedsquarefeet50_log', 'finishedsquarefeet6', 'fullbathcnt', 'fullbathcnt_log', 'garagetotalsqft',
                'garagetotalsqft_log', 'landtaxvaluedollarcnt', 'landtaxvaluedollarcnt_log', 'latitude_log',
                'longitude_log', 'lotsizesquarefeet', 'lotsizesquarefeet_log', 'numberofstories_log', 'poolcnt',
                'poolcnt_log', 'roomcnt', 'roomcnt_log', 'structuretaxvaluedollarcnt', 'structuretaxvaluedollarcnt_log',
                'taxamount', 'taxamount_log', 'taxdelinquencyyear', 'taxdelinquencyyear_log', 'taxvaluedollarcnt',
                'taxvaluedollarcnt_log', 'threequarterbathnbr', 'threequarterbathnbr_log', 'unitcnt', 'unitcnt_log',
                'yardbuildingsqft17', 'yearbuilt', 'yearbuilt_log', 'numberofstories',
                "yardbuildingsqft17_log", "finishedsquarefeet6_log")
char_vars <- c('airconditioningtypeid', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid',
                'calculatedbathnbr', 'censustractandblock', 'censustractandblock_12to12', 'censustractandblock_1to12',
                'censustractandblock_1to4', 'censustractandblock_1to8', 'censustractandblock_5to11', 'fullbathcnt',
                'hashottuborspa', 'heatingorsystemtypeid', 'latitude', 'longitude', 'poolcnt', 'pooltypeid7',
                'propertycountylandusecode', 'propertycountylandusecode_1to2', 'propertylandusetypeid',
                'propertyzoningdesc', 'propertyzoningdesc_1to3', 'propertyzoningdesc_1to4', 'propertyzoningdesc_5to10',
                'rawcensustractandblock', 'regionidcity', 'regionidneighborhood', 'regionidzip', 'roomcnt',
                'taxdelinquencyflag', 'taxdelinquencyyear', 'threequarterbathnbr', 'transactionMonth', 'transactionYear',
                'unitcnt', 'yearbuilt')
month_vars <- c("transactionMonth", "transactionYear")

### Load in the files
full <- fread("./full.csv", na.strings = c("NA","na","NAN","","null", "NULL",
                                        "--", "-", "**", "*", "N/A","n/a", "Missing",
                                        "MISSING", "missing"))

########################################################
#        H2O Load Data and Add New Variables
########################################################
h2o.init(min_mem_size="70G", nthreads = -1)
full_h2o <- h2o.importFile("./full.csv")

idx_ <- h2o.which(full_h2o[, 'split']=="train")
# h2o.impute(full_h2o, method ="median")
response <- "logerror"
# convert the char vars to factor
for (col in pure_char_vars) {
    previous_fullh2o_id <- h2o.getId(full_h2o)
    # full_h2o <- h2o.getFrame(previous_fullh2o_id)
    full_h2o[col] <- as.factor(as.character(full_h2o[col]))
}
# convert the char vars to factor, this is the weird h2o program
# x_ <- which(pure_char_vars==col)
# full_h2o <- h2o.getFrame(previous_fullh2o_id)
# for (col in pure_char_vars[x_]) {
#     previous_fullh2o_id <- h2o.getId(full_h2o)
#     full_h2o[col] <- as.factor(full_h2o[col])
# }

# Add some new features
full_h2o[, "taxamount_byft"] = full_h2o[, "taxamount"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "taxvaluedollarcnt_byft"] = full_h2o[, "taxvaluedollarcnt"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "calculatedfinishedsquarefeet_byft"] = full_h2o[, "calculatedfinishedsquarefeet"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "structuretaxvaluedollarcnt_byft"] = full_h2o[, "structuretaxvaluedollarcnt"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "lotsizesquarefeet_byft"] = full_h2o[, "lotsizesquarefeet"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "landtaxvaluedollarcnt_byft"] = full_h2o[, "landtaxvaluedollarcnt"] / full_h2o[, "finishedsquarefeet12"]
full_h2o[, "taxvaluedollarcnt_bytax"] = full_h2o[, "taxvaluedollarcnt"] / full_h2o[, "taxamount"]
full_h2o[, "calculatedfinishedsquarefeet_bytax"] = full_h2o[, "calculatedfinishedsquarefeet"] / full_h2o[, "taxamount"]
full_h2o[, "structuretaxvaluedollarcnt_bytax"] = full_h2o[, "structuretaxvaluedollarcnt"] / full_h2o[, "taxamount"]
full_h2o[, "lotsizesquarefeet_bytax"] = full_h2o[, "lotsizesquarefeet"] / full_h2o[, "taxamount"]
full_h2o[, "landtaxvaluedollarcnt_bytax"] = full_h2o[, "landtaxvaluedollarcnt"] / full_h2o[, "taxamount"]

full_h2o[, "taxvaluedollarcnt_bytaxcnt"] = full_h2o[, "taxvaluedollarcnt"] / full_h2o[, "taxvaluedollarcnt"]
full_h2o[, "calculatedfinishedsquarefeet_bytaxcnt"] = full_h2o[, "calculatedfinishedsquarefeet"] / full_h2o[, "taxvaluedollarcnt"]
full_h2o[, "structuretaxvaluedollarcnt_bytaxcnt"] = full_h2o[, "structuretaxvaluedollarcnt"] / full_h2o[, "taxvaluedollarcnt"]
full_h2o[, "lotsizesquarefeet_bytaxcnt"] = full_h2o[, "lotsizesquarefeet"] / full_h2o[, "taxvaluedollarcnt"]
full_h2o[, "landtaxvaluedollarcnt_bytaxcnt"] = full_h2o[, "landtaxvaluedollarcnt"] / full_h2o[, "taxvaluedollarcnt"]
full_h2o[, "taxamount_bycaltedft"] = full_h2o[, "taxamount"] / full_h2o[, "calculatedfinishedsquarefeet"]
full_h2o[, "taxvaluedollarcnt_bycaltedft"] = full_h2o[, "taxvaluedollarcnt"] / full_h2o[, "calculatedfinishedsquarefeet"]
full_h2o[, "structuretaxvaluedollarcnt_bycaltedft"] = full_h2o[, "structuretaxvaluedollarcnt"] / full_h2o[, "calculatedfinishedsquarefeet"]
full_h2o[, "lotsizesquarefeet_bycaltedft"] = full_h2o[, "lotsizesquarefeet"] / full_h2o[, "calculatedfinishedsquarefeet"]
full_h2o[, "landtaxvaluedollarcnt_bycaltedft"] = full_h2o[, "landtaxvaluedollarcnt"] / full_h2o[, "calculatedfinishedsquarefeet"]

full_h2o[, "taxamount_byft2"] = 1/full_h2o[, "taxamount_byft"]
full_h2o[, "taxvaluedollarcnt_byft2"] = 1/full_h2o[, "taxvaluedollarcnt_byft"]
full_h2o[, "calculatedfinishedsquarefeet_byft2"] = 1/full_h2o[, "calculatedfinishedsquarefeet_byft"]
full_h2o[, "structuretaxvaluedollarcnt_byft2"] = 1/full_h2o[, "structuretaxvaluedollarcnt_byft"]
full_h2o[, "lotsizesquarefeet_byft2"] = 1/full_h2o[, "lotsizesquarefeet_byft"]
full_h2o[, "landtaxvaluedollarcnt_byft2"] = 1/full_h2o[, "landtaxvaluedollarcnt_byft"]
full_h2o[, "taxvaluedollarcnt_bytax2"] = 1/full_h2o[, "taxvaluedollarcnt_bytax"]
full_h2o[, "calculatedfinishedsquarefeet_bytax2"] = 1/full_h2o[, "calculatedfinishedsquarefeet_bytax"]
full_h2o[, "structuretaxvaluedollarcnt_bytax2"] = 1/full_h2o[, "structuretaxvaluedollarcnt_bytax"]
full_h2o[, "lotsizesquarefeet_bytax2"] = 1/full_h2o[, "lotsizesquarefeet_bytax"]
full_h2o[, "landtaxvaluedollarcnt_bytax2"] = 1/full_h2o[, "landtaxvaluedollarcnt_bytax"]
full_h2o[, "taxamount_bycaltedft2"] = 1/full_h2o[, "taxamount_bycaltedft"]
full_h2o[, "taxvaluedollarcnt_bycaltedft2"] = 1/full_h2o[, "taxvaluedollarcnt_bycaltedft"]
full_h2o[, "structuretaxvaluedollarcnt_bycaltedft2"] = 1/full_h2o[, "structuretaxvaluedollarcnt_bycaltedft"]
full_h2o[, "lotsizesquarefeet_bycaltedft2"] = 1/full_h2o[, "lotsizesquarefeet_bycaltedft"]
full_h2o[, "landtaxvaluedollarcnt_bycaltedft2"] = 1/full_h2o[, "landtaxvaluedollarcnt_bycaltedft"]

full_h2o[, "random_rnorm"] = as.h2o(rnorm(nrow(full_h2o)))

num_vars <- unique(c(num_vars, "taxamount_byft","taxvaluedollarcnt_byft",
                        "calculatedfinishedsquarefeet_byft",
                        "structuretaxvaluedollarcnt_byft",
                        "lotsizesquarefeet_byft", "landtaxvaluedollarcnt_byft",
                        "taxvaluedollarcnt_bytax", "calculatedfinishedsquarefeet_bytax",
                        "structuretaxvaluedollarcnt_bytax", "lotsizesquarefeet_bytax",
                        "landtaxvaluedollarcnt_bytax",
                        "taxamount_byft2","taxvaluedollarcnt_byft2",
                        "calculatedfinishedsquarefeet_byft2",
                        "structuretaxvaluedollarcnt_byft2",
                        "lotsizesquarefeet_byft2", "landtaxvaluedollarcnt_byft2",
                        "taxvaluedollarcnt_bytax2", "calculatedfinishedsquarefeet_bytax2",
                        "structuretaxvaluedollarcnt_bytax2", "lotsizesquarefeet_bytax2",
                        "landtaxvaluedollarcnt_bytax2",
                        "taxamount_bycaltedft2", "taxvaluedollarcnt_bycaltedft2", "structuretaxvaluedollarcnt_bycaltedft2",
                        "lotsizesquarefeet_bycaltedft2", "landtaxvaluedollarcnt_bycaltedft2",
                        "taxvaluedollarcnt_bytaxcnt", "calculatedfinishedsquarefeet_bytaxcnt",
                        "structuretaxvaluedollarcnt_bytaxcnt", "lotsizesquarefeet_bytaxcnt",
                        "landtaxvaluedollarcnt_bytaxcnt", "taxamount_bycaltedft",
                        "taxvaluedollarcnt_bycaltedft", "structuretaxvaluedollarcnt_bycaltedft",
                        "lotsizesquarefeet_bycaltedft", "landtaxvaluedollarcnt_bycaltedft",
                        "random_rnorm"))

predictors_prev <- predictors
predictors <- unique(c(predictors, num_vars))

full_train_h2o <- full_h2o[as.vector(idx_), ]

h2o.exportFile(full_h2o, path = "./full_h2o.csv")


########################################################
#        H2O for Quantile Regression
########################################################
# split into train and validation sets
# full_train_h2o.splits <- h2o.splitFrame(data = full_train_h2o, ratios=0.6, seed = 1234)
# train <- full_train_h2o.splits[[1]]
# valid <- full_train_h2o.splits[[2]]

# try using the `quantile_alpha` parameter:
# train your model, where you specify distribution = quantile
# and the quantile_alpha value

# h2o.rm(h2o.getId(train))
# h2o.rm(h2o.getId(valid))

# full_train_h2o_gbm <- h2o.gbm(x = predictors, y = response, training_frame = full_train_h2o,
#                         #validation_frame = valid,
#                         #keep_cross_validation_predictions= TRUE,
#                         nfolds = 20,
#                         #distribution = 'quantile',
#                         #quantile_alpha = .8,
#                         distribution = 'laplace',
#                         stopping_metric = "MAE",
#                         learn_rate = 0.1,
#                         stopping_rounds = 20,
#                         col_sample_rate = 0.6,
#                         sample_rate = 0.8,
#                         ntrees = 140, max_depth = 5,
#                         seed = 1234)

#h2o.cross_validation_predictions(full_train_h2o_gbm)
# h2o.predict(full_train_h2o_gbm, full_train_h2o)

n_folds <- 20
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

full_train_h2o[, "foldid"] = as.h2o(full[split=="train", foldid])
full[, predictions_0.15 := 0]
full[, predictions_0.85 := 0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    valididx <- which(full$split=="train" & full$foldid == i)
    trainframe <- full_train_h2o[trainidx, ]
    validframe <- full_train_h2o[valididx, ]

    full_train_h2o_gbm_0.15 <- h2o.gbm(x = predictors, y = response,
                            training_frame = trainframe,
                            validation_frame = validframe,
                            #keep_cross_validation_predictions= TRUE,
                            #nfolds = 5,
                            distribution = 'quantile',
                            quantile_alpha = .15,
                            stopping_rounds = 3,
                            learn_rate = 0.03,
                            col_sample_rate = 0.6,
                            sample_rate = 0.8,
                            ntrees = 3000,
                            max_depth = 5,
                            seed = 1234)

    full_train_h2o_gbm_0.85 <- h2o.gbm(x = predictors, y = response,
                            training_frame = trainframe,
                            validation_frame = validframe,
                            #keep_cross_validation_predictions= TRUE,
                            #nfolds = 5,
                            distribution = 'quantile',
                            quantile_alpha = .85,
                            stopping_rounds = 3,
                            learn_rate = 0.03,
                            col_sample_rate = 0.6,
                            sample_rate = 0.8,
                            ntrees = 3000,
                            max_depth = 5,
                            seed = 1234)

    # print the mse for validation set
    # print(h2o.mae(full_train_h2o_gbm, valid = TRUE))
    predictions_0.15_ <- as.vector(h2o.predict(full_train_h2o_gbm_0.15, validframe))
    predictions_0.85_ <- as.vector(h2o.predict(full_train_h2o_gbm_0.85, validframe))
    full[valididx, predictions_0.15 := predictions_0.15_]
    full[valididx, predictions_0.85 := predictions_0.85_]
    try(h2o.rm(h2o.getId(trainframe)))
    try(h2o.rm(h2o.getId(validframe)))
    try(h2o.rm(h2o.getId(trainframe)))
    try(h2o.rm(h2o.getId(validframe)))
}


x_ <- full[split=='train', sum(logerror > predictions_0.85 | logerror < predictions_0.15)]
print(paste0("There are ", x_, " records out of ", length(response_actual)," records are outliers"))
# "There are 28295 records out of 90275 records are outliers"
#x=c(4,1,3,2)
#x2= sort(x)
#x[match(x=x2, table=x)]
## Plot confidence intervals
response_actual <- full[split=='train', logerror]
predictions_0.15_ <- full[split=='train', predictions_0.15]
predictions_0.85_ <- full[split=='train', predictions_0.85]
order_ <- match(x=sort(response_actual), table=response_actual)
dat <- cbind(t(t(response_actual[order_])), t(t(predictions_0.15_[order_])), t(t(predictions_0.85_[order_])))
matplot(dat, type = c("l"), col = 1:3)
legend("topleft", legend = 1:3, col=1:3, pch=1) # optional legend

if (FALSE) {
    # grid over `quantile_alpha` parameter
    # select the values for `quantile_alpha` to grid over
    hyper_params <- list( quantile_alpha = c(.2, .5, .8) )
    # build grid search with previously made GBM and hyperparameters
    grid <- h2o.grid(x = predictors, y = response, training_frame = train,
                     validation_frame = valid, algorithm = "gbm",
                     grid_id = "full_train_h2o_grid",
                     distribution = "quantile",
                     hyper_params = hyper_params,
                     seed = 1234)

    # Sort the grid models by MSE
    sortedGrid <- h2o.getGrid("full_train_h2o_grid", sort_by = "mse", decreasing = FALSE)
    sortedGrid
}


######################################################################################################
#   Build Models after throwing out the outliers
######################################################################################################
rowidx_to_throwout <- full[split=='train' & (logerror < predictions_0.15 | logerror > predictions_0.85)]$rowidx

idx_ <- which(!(as.vector(full_h2o[, "rowidx"]) %in% rowidx_to_throwout))

# Quick Check if the following two are the same or not. Expect to be same
summary(response_actual[!((response_actual > predictions_0.85_) | (response_actual < predictions_0.15_))])
summary(full[idx_][split=="train", logerror])

################################################################################
#   LEGACY CODE
################################################################################
if (FALSE) {

}
