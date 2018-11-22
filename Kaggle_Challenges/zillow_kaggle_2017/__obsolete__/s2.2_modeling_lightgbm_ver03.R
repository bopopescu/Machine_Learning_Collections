setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(lightgbm)
library(readr)
library(lubridate)


load("s2_line415.RData")


##################  Add propensity by lightgbm  ##################
setkey(full, rowidx)

n_folds <- 10
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]


# Undo what previous xgboost has done, restart with new modeling steps
for (col in c("freq_bin1", "freq_bin2", "freq_bin3", "freq_bin4", "freq_bin5", "freq_bin6",
            "freq_bin7", "freq_bin8", "freq_bin9", "freq_bin10", "freq_bin11", "freq_bin12",
            "freq_bin13", "freq_bin14", "freq_bin15", "freq_bin16", "freq_v2_bin1", "freq_v2_bin2",
            "freq_v2_bin3", "freq_v2_bin4", "freq_v2_bin5", "freq_v2_bin6", "freq_v2_bin7")) {
    full[, paste0(col) := NULL]
}
new_predictors <- c() ; gc();

# Loop through the calculation for the propensity of prediciton bins
percentiles_to_pred <- c(0.005, 0.01, 0.03, 0.1, 0.25, 0.75, 0.9, 0.97, 0.99, 0.995)
percentiles_to_pred <- sort(unique(round(c(percentiles_to_pred, seq(0.05, 0.95, by=0.05)),3)))
gc()

for (leftcutoff in percentiles_to_pred) {
    leftcutoff_str <- gsub(pattern = "\\.",replacement = "", x = as.character(leftcutoff))
    quantile_ <- quantile(full[split=="train", logerror], probs=leftcutoff)
    new_target <- paste0("propens_less",leftcutoff_str)
    print(paste0("Predicting the propensity of ", "propens_less",leftcutoff_str))
    full[, paste0("propens_less",leftcutoff_str) := 1*(full$logerror < quantile_)]

    full[, paste0(new_target, "_pred") := 0.0]  # set initial starting point

    predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
    full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

    for (i in seq(1, n_folds)) {
        train_dm_reg <- lgb.Dataset(
                        data = full_mm[full$split=="train" & full$foldid != i, ],
                        label = full[full$split=="train" & full$foldid != i, get(new_target)]
                    )
        valid_dm_reg <- lgb.Dataset(
                        data = full_mm[full$split=="train" & full$foldid == i, ],
                        label = full[full$split=="train" & full$foldid == i, get(new_target)]
                    )
        test_dm_reg <- lgb.Dataset(data = full_mm[full$split=="test", ])
        param <- list(objective = "binary",
                        #metric="poisson", #metric="l2,l1"
                        metric="auc",
                        num_leaves = 63, # 2**8 -1, interaction depth=5
                        learning_rate = 0.1,
                        boost_from_average = TRUE,
                        boosting_type = 'dart')  #gbdt is another choice
        set.seed(10011)
        lgb1_reg <- lgb.train(params = param,
                            data = train_dm_reg,
                            valids = list(train=train_dm_reg, valid=valid_dm_reg),
                            nrounds = 5000,
                            #nfold = 8,
                            #device = "cpu", # or use gpu if it's available
                            num_threads = parallel::detectCores() - 1,
                            min_data = 2,
                            feature_fraction = 0.6,
                            bagging_fraction = 0.632,
                            bagging_freq = 1, # bagging_fraction every k iterations
                            early_stopping_rounds = 10,
                            lambda_l1 = 0.1,
                            lambda_l2 = 1,
                            verbose = 2)
        gc()

        best.iter <- lgb1_reg$best_iter
        # print(paste0("Best GBM iteration is ", best.iter))
        valid_pred_ <- predict(lgb1_reg, full_mm[full$split=="train" & full$foldid == i, ])
        test_pred_ <- predict(lgb1_reg, full_mm[full$split=="test", ])

        full[full$split=="train" & full$foldid == i, paste0(new_target, "_pred") := valid_pred_]
        full[full$split=="test", paste0(new_target, "_pred") :=  get(paste0(new_target, "_pred")) + test_pred_]

        rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
    }
    full[split=="test", paste0(new_target, "_pred") :=  get(paste0(new_target, "_pred")) / n_folds]
    new_predictors <- c(new_predictors, paste0(new_target, "_pred"))
}



################ First Round of Regression Adjusting other Quarters of Data #############
# remove some variables to start new round of model fitting
for (col in new_predictors) {
    if (col %in% colnames(full)) {full[, paste0(col) := NULL]}
    if (gsub("_pred", "", col) %in% colnames(full)) {full[, gsub("_pred", "", col) := NULL]}
}

# Add some new features
full[, taxamount_byft := taxamount / finishedsquarefeet12]
full[, taxvaluedollarcnt_byft := taxvaluedollarcnt / finishedsquarefeet12]
full[, calculatedfinishedsquarefeet_byft := calculatedfinishedsquarefeet / finishedsquarefeet12]
full[, structuretaxvaluedollarcnt_byft := structuretaxvaluedollarcnt / finishedsquarefeet12]
full[, lotsizesquarefeet_byft := lotsizesquarefeet / finishedsquarefeet12]
full[, landtaxvaluedollarcnt_byft := landtaxvaluedollarcnt / finishedsquarefeet12]
full[, poolcnt_byft := poolcnt / finishedsquarefeet12]
full[, taxvaluedollarcnt_bytax := taxvaluedollarcnt / taxamount]
full[, calculatedfinishedsquarefeet_bytax := calculatedfinishedsquarefeet / taxamount]
full[, structuretaxvaluedollarcnt_bytax := structuretaxvaluedollarcnt / taxamount]
full[, lotsizesquarefeet_bytax := lotsizesquarefeet / taxamount]
full[, landtaxvaluedollarcnt_bytax := landtaxvaluedollarcnt / taxamount]
full[, poolcnt_bytax := poolcnt / taxamount]

predictors <- unique(c(predictors, "taxamount_byft","taxvaluedollarcnt_byft",
                        "calculatedfinishedsquarefeet_byft",
                        "structuretaxvaluedollarcnt_byft",
                        "lotsizesquarefeet_byft", "landtaxvaluedollarcnt_byft",
                        "poolcnt_byft",
                        "taxvaluedollarcnt_bytax", "calculatedfinishedsquarefeet_bytax",
                        "structuretaxvaluedollarcnt_bytax", "lotsizesquarefeet_bytax",
                        "landtaxvaluedollarcnt_bytax", "poolcnt_bytax"))

setkey(full, rowidx)
full_train_ver2 <- full[split == "train"]

# the month variables are month_vars, they are related to the transaction time
#[1] "transactionMonth"      "transactionMonth2"     "transactionMonth3"
#[4] "transactionDayofYear"  "transactionDayofYear2" "transactionDayofYear3"
monthdf <- unique(full[, c(month_vars, "transactiondate"), with = F])
monthdf[, transactionYear := year(transactiondate)]
monthdf[, transactiondaysfrom2016 := lubridate::yday(transactiondate) + (transactionYear - 2016)*365]
monthdf[, transactiondaysfrom2016_log := log(transactiondaysfrom2016)]
# transactiondate is the key of monthdf

# Replace the columns in the monthdf in the full using the monthdf data
for (col in setdiff(copy(colnames(monthdf)), "transactiondate")) {
    if (col %in% colnames(full)) {full[, paste0(col) := NULL]}
    if (col %in% colnames(full_train_ver2)) {full_train_ver2[, paste0(col) := NULL]}
}

# full_train_ver2 is the copy of training data that will have the month info to be replaced
# > full_train_ver2[, sum(month(transactiondate) < 10)]
# [1] 81733  This is the number of the records that will need replacement
n_records_to_be_replaced <- full_train_ver2[, sum(month(transactiondate) < 10)]
transactiondates_to_use <- monthdf[year(transactiondate)==2016 & month(transactiondate)>=10 & lubridate::day(transactiondate)==15,
                                        unique(transactiondate)]
set.seed(10011)
months_to_fill <- sample(transactiondates_to_use, size = n_records_to_be_replaced, replace=T)
full_train_ver2[which(month(full_train_ver2$transactiondate) < 10), transactiondate := months_to_fill]

# Now replace the data's transaction date information using the monthdf info
full_train_ver2 <- merge(full_train_ver2, monthdf, by = "transactiondate")
setkey(full_train_ver2, rowidx)

full <- merge(full, monthdf, by = "transactiondate")
setkey(full, rowidx)

# check if the full, full_train_ver2 data all have the common set of variables
sum(ncol(full) != ncol(full_train_ver2))
# expect to be 0
length(setdiff(colnames(full_train_ver2), colnames(full)))

# align all column names
common_rowidx_offset <- 10000000
full_train_ver2[, rowidx := rowidx - common_rowidx_offset]
c_ <- copy(colnames(full))
full_train_ver2 <- full_train_ver2[, c_, with=F]

# Create the FULL data version2 by adding the full_train_ver2 to the full
full <- rbind(full, full_train_ver2)  # where the negative rowidx are the newly modified data added to train
rm(full_train_ver2)

################ First Round of Model Fitting using the modified full data #############
setkey(full, rowidx)

n_folds <- 20
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_ver1 := 0.0]

new_predictors <- setdiff(colnames(monthdf), "transactiondate")
predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float
full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])
gc()
for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & full$rowidx >= 0)
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx,], label = full[trainidx, logerror])
    valididx <- which(full$split=="train" & full$foldid == i & full$rowidx >= 0)
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx,], label = full[valididx, logerror])
    scoreidx <- which(full$split=="train" & full$rowidx < 0)
    score_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[scoreidx,], label = full[scoreidx, logerror])
    param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 127, # 2**8 -1, interaction depth=5
                    learning_rate = 0.01,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 493,
                    #nfold = 20,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 5,
                    feature_fraction = 0.1,
                    bagging_fraction = 0.8,
                    bagging_freq = 1, # bagging_fraction every k iterations
                    # early_stopping_rounds = 295,
                    lambda_l1 = 10,
                    lambda_l2 = 2.56,
                    verbose = 1)
    gc()
    # print(paste0("Best GBM iteration is ", lgb1_reg$best_iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bylgb_ver1 := valid_pred_]
    score_pred_ <- predict(lgb1_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bylgb_ver1 := logerror_pred_bylgb_ver1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg, score_dm_reg); gc();
}

full[scoreidx, logerror_pred_bylgb_ver1 := logerror_pred_bylgb_ver1 / n_folds]


################  Modify the train data with after changing the month  ##############
setkey(full, rowidx)

y_new_hat <- full[split=="train" & rowidx < 0, logerror_pred_bylgb_ver1]
y_hat <- full[split=="train" & rowidx >= 0, logerror_pred_bylgb_ver1]

idx_nochange <- which(full[split=="train" & rowidx >= 0]$transactionMonth >= 10)

y <- full[split=="train" & rowidx >= 0, logerror]
y_new <- y + (y_new_hat - y_hat)
y_new[idx_nochange] <- y[idx_nochange]  # this is the original 4th quarter logerror

full[split=="train" & rowidx < 0, logerror := y_new + 0.0]


n_folds <- 20
set.seed(10012)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_ver2 := 0.0]

# clear previous lightgbm objects
rm(lgb1_reg, full_dm)

#new_predictors <- unique(c(new_predictors, "logerror_pred_bylgb_ver1"))
predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float
full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])
gc()
for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & full$rowidx < 0)
    valididx <- which(full$split=="train" & full$foldid == i & full$rowidx < 0)
    scoreidx <- which(full$split=="test")
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx,], label = full[trainidx, logerror])
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx,], label = full[valididx, logerror])
    score_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[scoreidx,], label = full[scoreidx, logerror])
    param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 127, # 2**8 -1, interaction depth=5
                    learning_rate = 0.01,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 493,
                    #nfold = 20,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 5,
                    feature_fraction = 0.1,
                    bagging_fraction = 0.8,
                    bagging_freq = 1, # bagging_fraction every k iterations
                    # early_stopping_rounds = 295,
                    lambda_l1 = 10,
                    lambda_l2 = 2.56,
                    verbose = 1)
    gc()
    # print(paste0("Best GBM iteration is ", xgb1_reg$best_iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bylgb_ver2 := valid_pred_]
    score_pred_ <- predict(lgb1_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bylgb_ver2 := logerror_pred_bylgb_ver2 + score_pred_]
    rm(train_dm_reg, valid_dm_reg, score_dm_reg); gc();
}

full[scoreidx, logerror_pred_bylgb_ver2 := logerror_pred_bylgb_ver2 / n_folds]
# clear previous lightgbm objects
rm(lgb1_reg, full_dm)










########################################################################
# Create Submission
########################################################################
setkey(full, rowidx)

distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with ordre
    x = data.table(x)
    length_ = nrow(x)
    x[, ids := seq(1, length_)]
    x2 = x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}

test_df_raw <- fread("./source_data/sample_submission.csv", header = TRUE)
setnames(test_df_raw, "ParcelId", "parcelid")
test_df_raw[, rowid := seq(1, nrow(test_df_raw))]
submission <- matrix(data=full[split == "test", logerror_pred], ncol=6, byrow = TRUE)
parcelids_ <- distinct_field(full[split == "test", parcelid])
submission <- cbind(data.table(submission), data.table(parcelid = parcelids_))
submission <- merge(submission, test_df_raw, by = "parcelid")
setkey(submission, rowid)
submission <- submission[, c("parcelid", "V1", "V2", "V3", "V4", "V5", "V6"), with =F]
for (col in c("V1", "V2", "V3", "V4", "V5", "V6")) {
    submission[, paste0(col) := round(get(col), 8)]
}
setnames(submission, colnames(submission), c("ParcelId", "201610", "201611", "201612", "201710", "201711", "201712"))
submission[, 2:7] <- round(submission[, 2:7], 4)

write.csv(submission, file = "submission_lgb_s2.2_0718.csv", row.names = F)


############## LEGACY CODE ###################
if (FALSE) {
    # Get the weights for adjusting the ensemble
    mae <- function(a, b) {return(mean(abs(a-b)))}
    ensemble_adjust <- function(vector1, vector2, results,
                                measure_function = mae,
                                minimum = TRUE){
        # Return the best weight w for tuning the ensemble
        if (minimum) {
            measure_function2 <- measure_function
            sign_ <- 1.0
        } else {
            measure_function2 <- function(a, b) {return((-1.0) * measure_function(a,b))}
            sign_ <- -1.0
        }
        best_w <- 0.5
        best_res <- Inf
        for (w in seq(0.2, 0.8, by = 0.01)) {
            vec_ <- w * vector1 + (1-w) * vector2
            res_ <- measure_function2(vec_, results)
            if (best_res > res_) {
                best_w <- w
                best_res <- res_
            }
        }
        res1 <- measure_function2(vector1, results)
        res2 <- measure_function2(vector2, results)
        if (res1 < res2) {
            res_old <- res1
        } else {
            res_old <- res2
        }
        best_res <- sign_ * best_res
        res_old <- sign_ * res_old
        print(paste0("Error Metric improves from ", res_old, " to ", best_res))
        return(best_w)
    }

    if (FALSE) {
        # Test
        vector1 = runif(100000)
        vector2 = runif(100000)
        results = runif(100000)

        mae2 = function(a, b) {return(mae(a, b) * (-1))}
        ensemble_adjust(vector1, vector2, results, measure_function = mae2,
                        minimum = F)
        ensemble_adjust(vector1, vector2, results)
    }

    ######### ensemble ##########
    anothersubmission <- fread("./source_data/lgb_starter.csv", header = TRUE)
    weights <- c(0.67, 0.33)
    submission[, 2:7] <- ((weights[1]/sum(weights)) * anothersubmission[, 2:7] + (weights[2]/sum(weights)) * submission[, 2:7])
    submission[, 2:7] <- round(submission[, 2:7], 4)
}





##########  LEGACY CODE  ##########
if (FALSE) {
    train_dm_reg <- xgb.DMatrix(
                    data = full_mm[trainidx, ],
                    label = full[trainidx, logerror],
                    missing = NA
                )
    valid_dm_reg <- xgb.DMatrix(
                    data = full_mm[valididx, ],
                    label = full[valididx, logerror],
                    missing = NA
                )
    score_dm_reg <- xgb.DMatrix(
                data = full_mm[scoreidx, ],
                missing = NA
            )
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    xgb1_reg <- xgb.train(
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "mae",
                    #feval = mae,
                    print_every_n = 50,
                    early_stopping_rounds = 15, # early stopping may cause overfitting
                    maximize = FALSE,
                    colsample_bytree = 0.16,
                    colsample_bylevel = 1,
                    min_child_weight = 5,
                    objective = "reg:linear",
                    subsample = 0.632,
                    nthread = parallel::detectCores() - 1,
                    alpha = 100,
                    nrounds = 5000, # the best iteration is from xgb.cv calculation through same nfold
                    eta = 0.02,
                    max.depth = 3,
                    lambda = 100
                )
    gc()
    # LightGBM version
    full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx,], label = full[trainidx, logerror])
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx,], label = full[valididx, logerror])
    score_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[scoreidx,], label = full[scoreidx, logerror])
    param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 127, # 2**8 -1, interaction depth=5
                    learning_rate = 0.01,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 493,
                    #nfold = 20,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 5,
                    feature_fraction = 0.1,
                    bagging_fraction = 0.8,
                    bagging_freq = 1, # bagging_fraction every k iterations
                    # early_stopping_rounds = 295,
                    lambda_l1 = 10,
                    lambda_l2 = 2.56,
                    verbose = 1)
    gc()
}
