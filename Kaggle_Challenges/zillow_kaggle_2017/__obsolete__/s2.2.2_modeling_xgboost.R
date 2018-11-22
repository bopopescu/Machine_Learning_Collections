setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(lightgbm)
library(readr)
library(lubridate)
library(Matrix)
library(pdp)


load("s2.2_lgb_line230.RData")

#########################  Get the Month Coefficient  ###############
n_folds <- 20
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
predictors_ <- c(setdiff(predictors_, month_vars), "transactionMonth")
x_ <- full[, predictors_, with = F]
x_ <- x_[, transactionMonth := as.numeric(as.character(transactionMonth))]
full_mm <- data.matrix(x_) ; rm(x_); gc();
trainidx <- which(full[, split=="train"])
train_dm_reg <- xgb.DMatrix(
                data = full_mm[trainidx, ],
                label = full[trainidx, logerror],
                missing = NA
            )
set.seed(10011)
xgb0_reg <- xgb.train(
                data = train_dm_reg,
                watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                eval_metric = "mae",
                #feval = mae,
                print_every_n = 50,
                early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.03,
                colsample_bylevel = 1,
                min_child_weight = 5,
                objective = "reg:linear",
                subsample = 0.632,
                nthread = parallel::detectCores() - 1,
                alpha = 10,
                nrounds = 124, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.1,
                max.depth = 5,
                lambda = 20
            )
#Stopping. Best iteration:
#[124]   train-mae:0.067823+0.000101 test-mae:0.068038+0.001982
month_trend_df <- pdp::partial(xgb0_reg, pred.var = "transactionMonth", plot = FALSE, train = full_mm[trainidx, ])
x_ <- full[trainidx, "transactionMonth", with = F]
x_[, transactionMonth := as.numeric(as.character(transactionMonth))]
x_ <- table(x_$transactionMonth)
month_trend_df <- as.data.table(month_trend_df)
month_trend_df[, yhat := yhat - weighted.mean(month_trend_df$yhat, x_)]
setnames(month_trend_df, "yhat", "transactionMonth_bias")
month_trend_df[, transactionMonth := factor(transactionMonth, levels=levels(full$transactionMonth))]

full <- merge(full, month_trend_df, by = "transactionMonth")
setkey(full, rowidx)
rm(full_mm); gc();



##################  Add propensity by lightgbm  ##################
setkey(full, rowidx)

n_folds <- 10
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_new := logerror - transactionMonth_bias]

new_predictors <- c() ; gc();

# Loop through the calculation for the propensity of prediciton bins
percentiles_to_pred <- c(0.001, 0.002, 0.003, 0.004, 0.005, 0.006,0.007,0.008,0.009,
                        0.01, 0.03, 0.1,
                        0.25, 0.75, 0.9, 0.97, 0.99,
                        0.991, 0.992, 0.993, 0.994, 0.995,
                        0.996, 0.997, 0.998, 0.999)
percentiles_to_pred <- sort(unique(round(c(percentiles_to_pred, seq(0.05, 0.95, by=0.05)),3)))
gc()

# The loops also need to ensure there is no transaction date or month related info since
# that info is put in the offset as transactionMonth_bias
for (leftcutoff in percentiles_to_pred) {
    leftcutoff_str <- gsub(pattern = "\\.",replacement = "", x = as.character(leftcutoff))
    quantile_ <- quantile(full[split=="train", logerror_new], probs=leftcutoff)
    new_target <- paste0("propens_less",leftcutoff_str)
    print(paste0("Predicting the propensity of ", "propens_less",leftcutoff_str))
    full[, paste0("propens_less",leftcutoff_str) := 1*(full$logerror_new < quantile_)]
    full[, paste0(new_target, "_pred") := 0.0]  # set initial starting point

    #predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
    predictors_ <- setdiff(colnames(full_sparse),
                            colnames(full_sparse)[grepl(pattern = "trans*", x = colnames(full_sparse))])  # remove the month variables
    full_mm <- full_sparse[, predictors_]
    # Get best iteration
    scale_pos_weight <- full[split=="train", sum(get(new_target) ==0)/sum(get(new_target))]
    trainidx <- which(full$split=="train")
    train_dm_reg <- xgb.DMatrix(data = full_mm[trainidx, ],
                                label = full[trainidx, get(new_target)],
                                missing = NA
                                )
    set.seed(10011)
    x_ <- xgb.cv(data = train_dm_reg,
                        nfold = n_folds,
                        scale_pos_weight = scale_pos_weight,
                        num_threads = parallel::detectCores() - 1,
                        print_every_n = 50,
                        early_stopping_rounds = 50, # early stopping may cause overfitting
                        colsample_bytree = 0.5,
                        colsample_bylevel = 1,
                        min_child_weight = 5,
                        objective = "binary:logistic",
                        eval_metric = "auc",
                        subsample = 0.632,
                        nthread = parallel::detectCores() - 1,
                        alpha = 0.1,
                        nrounds = 5000, # the best iteration is from xgb.cv calculation through same nfold
                        eta = 0.1,
                        max.depth = 5,
                        lambda = 1)
    nrounds <- x_$best_iter
    for (i in seq(1, n_folds)) {
        trainidx <- which(full$split=="train" & full$foldid != i)
        train_dm_reg <- xgb.DMatrix(data = full_mm[trainidx, ],
                                label = full[trainidx, get(new_target)],
                                missing = NA
                                )
        valididx <- which(full$split=="train" & full$foldid == i)
        valid_dm_reg <- xgb.DMatrix(data = full_mm[valididx, ],
                                label = full[valididx, get(new_target)],
                                missing = NA
                                )
        set.seed(10011)
        xgb1_reg <- xgb.train(data = train_dm_reg,
                                watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                                scale_pos_weight = scale_pos_weight,
                                num_threads = parallel::detectCores() - 1,
                                print_every_n = 50,
                                colsample_bytree = 0.5,
                                colsample_bylevel = 1,
                                min_child_weight = 5,
                                objective = "binary:logistic",
                                eval_metric = "auc",
                                subsample = 0.632,
                                nthread = parallel::detectCores() - 1,
                                alpha = 0.1,
                                nrounds = nrounds, # the best iteration is from xgb.cv calculation through same nfold
                                eta = 0.1,
                                max.depth = 5,
                                lambda = 1)
        gc(); gc(); gc();

        valid_pred_ <- predict(xgb1_reg, full_mm[valididx, ])
        test_pred_ <- predict(xgb1_reg, full_mm[full$split=="test", ])

        full[valididx, paste0(new_target, "_pred") := valid_pred_]
        full[full$split=="test", paste0(new_target, "_pred") :=  get(paste0(new_target, "_pred")) + test_pred_]

        rm(train_dm_reg, valid_dm_reg); gc();
    }
    full[split=="test", paste0(new_target, "_pred") :=  get(paste0(new_target, "_pred")) / n_folds]
    new_predictors <- c(new_predictors, paste0(new_target, "_pred"))
}




######################  Only GLM on the propensity scores to predict logerror  #################
n_folds <- 20
set.seed(10101)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

full[, logerror_pred_bygblinear_v1 := 0]
full_mm <- scale(data.matrix(full[, new_predictors, with =F]))

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(
        data = full_mm[trainidx, ],
        label = full[trainidx, logerror-transactionMonth_bias],
        missing = 0
    )
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
        data = full_mm[valididx, ],
        label = full[valididx, logerror-transactionMonth_bias],
        missing = 0
    )
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    xgb0_reg <- xgb.train(
        data = train_dm_reg,
        booster ="gblinear",
        #nfold=20,
        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
        eval_metric = "mae",
        #feval = mae,
        print_every_n = 50,
        #early_stopping_rounds = 15, # early stopping may cause overfitting
        objective = "reg:linear",
        nthread = parallel::detectCores() - 1,
        alpha = 611.8,
        nrounds = 2, # the best iteration is from xgb.cv calculation through same nfold
        lambda = 6.4
    )
    gc(); gc(); gc();
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(xgb0_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bygblinear_v1 := valid_pred_]
    score_pred_ <- predict(xgb0_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bygblinear_v1 := logerror_pred_bygblinear_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bygblinear_v1 := logerror_pred_bygblinear_v1 / n_folds]
gc(); gc();


##############################################################################
#   Linear Regression
##############################################################################
#TODO

####### Step 1 of Linear Regression: Get variable importance
setkey(full, rowidx)

predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
predictors_ <- setdiff(predictors_, month_vars)
full_mm <- data.matrix(full[, predictors_, with =F])

# use median to do missing data imputation and scaling of matrix
medians_ <- sapply(colnames(full_mm), FUN=function(x){quantile(full_mm[, x], probs = 0.5, na.rm=T)})
for (x in seq_len(ncol(full_mm))){
    # missing data imputation
    full_mm[, x][is.na(full_mm[, x])] = medians_[x]
}
gc(); gc();
for (x in seq_len(ncol(full_mm))){
    # scaling design matrix
    full_mm[, x] <- (full_mm[, x]-mean(full_mm[, x]))/(var(full_mm[, x]) ** 0.5)
    gc(); gc();
}
# exclude the columns with 0 variance
varstotthrow_ <- c()
for (x in seq_len(ncol(full_mm))){
    if (length(unique(full_mm[, x])) == 1) {
        varstotthrow_ <- c(varstotthrow_, x)
    }
    gc(); gc();
}
full_mm <- full_mm[, -varstotthrow_]
trainidx <- which(full$split=="train")
train_dm_reg <- lgb.Dataset(data = full_mm[trainidx, ],
        label = full[trainidx, logerror - transactionMonth_bias])
param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 16, # 2**8 -1, interaction depth=5
                    learning_rate = 0.05,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
set.seed(10011)
lgb1_reg <- lgb.train(params = param,
                data = train_dm_reg,
                #valids = list(train=train_dm_reg, valid=valid_dm_reg),
                nrounds = 111,
                #nfold = 20,
                #device = "cpu", # or use gpu if it's available
                num_threads = parallel::detectCores() - 1,
                min_data = 1,
                feature_fraction = 0.05,
                bagging_fraction = 0.85,
                bagging_freq = 1, # bagging_fraction every k iterations
                #early_stopping_rounds = 50,
                lambda_l1 = 12.14,
                lambda_l2 = 100,
                verbose = 1)
var_importance <- lgb.importance(lgb1_reg)


######################################
####### Step 2 of Linear Regression: Linear Regressions by Various Variable Combinations
######################################
# Run Linear Model ROUND #3
trainidx <- which(full$split=="train")
train_dm_reg <- xgb.DMatrix(
                data = full_mm[trainidx, ],
                label = full[trainidx, logerror - transactionMonth_bias],
                missing = NA
            )
set.seed(10011)
xgb0_reg <- xgb.train(
                data = train_dm_reg,
                watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                eval_metric = "mae",
                #feval = mae,
                #nfold=40,
                print_every_n = 50,
                #early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.03,
                colsample_bylevel = 1,
                min_child_weight = 5,
                objective = "reg:linear",
                subsample = 0.632,
                nthread = parallel::detectCores() - 1,
                alpha = 50,
                nrounds = 77, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.1,
                max.depth = 5,
                lambda = 20
            )
var_importance2 <- xgb.importance(feature_names=colnames(full_mm), model=xgb0_reg)

#
full_mm_vars <- setdiff(var_importance2$Feature[1:50], new_predictors)
full[, logerror_pred_bygblinear_v2 := 0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(
        data = full_mm[trainidx, full_mm_vars], # use first top 3 vars from ranking
        label = full[trainidx, logerror - transactionMonth_bias],
        missing = NA
    )
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
        data = full_mm[valididx, full_mm_vars],
        label = full[valididx, logerror - transactionMonth_bias],
        missing = NA
    )
    gc(); gc();
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    xgb0_reg <- xgb.train(
        data = train_dm_reg,
        booster ="gblinear",
        #nfold=40,
        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
        eval_metric = "mae",
        #feval = mae,
        print_every_n = 50,
        #early_stopping_rounds = 15, # early stopping may cause overfitting
        objective = "reg:linear",
        nthread = parallel::detectCores() - 1,
        alpha = 200,
        nrounds = 1,
        lambda = 0
    )
    gc(); gc(); gc();
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] test-mae:0.067505+0.003075)
    valid_pred_ <- predict(xgb0_reg, full_mm[valididx, full_mm_vars])
    full[valididx, logerror_pred_bygblinear_v2 := valid_pred_]
    score_pred_ <- predict(xgb0_reg, full_mm[scoreidx, full_mm_vars])
    full[scoreidx, logerror_pred_bygblinear_v2 := logerror_pred_bygblinear_v2 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bygblinear_v2 := logerror_pred_bygblinear_v2 / n_folds]
gc(); gc();


if (FALSE) {
    full_pred_temp <- full[, c("rowidx",
                            "logerror_pred_bygblinear_v1",
                            "logerror_pred_bygblinear_v2"), with =F]
    save(full_pred_temp, file="s2.2_xgb_zzz_full_preds.RData")
    rm(full_pred_temp); gc();
}

















####################  Final Minor Global Bias Tuning  ##################
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

# add adjustment and month trend bias
full[, logerror_pred_bylgb_v2 := logerror_pred_byxgb_v1 + transactionMonth_bias]
full[, logerror_pred_bygblinear_v2 := logerror_pred_bygblinear_v1 + transactionMonth_bias]

w1 <- ensemble_adjust(full[split=="train", logerror_pred_bylgb_v2 - transactionMonth_bias],
                full[split=="train", logerror_pred_bygblinear_v2 - transactionMonth_bias],
                full[split=="train", logerror - transactionMonth_bias], measure_function = mae, minimum = TRUE)
full[, logerror_pred := 0 * logerror_pred_bylgb_v2 + 1 * logerror_pred_bygblinear_v2]


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
submission[, 2:7] <- round(submission[, 2:7], 6)

write.csv(submission, file = "submission_lgb_mix_gblinear_s2.2_0727.csv", row.names = F)




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
    submission[, 2:7] <- round(submission[, 2:7], 6)
}


##########  LEGACY CODE  ##########
if (FALSE) {
    # Partial Dependency Plot
    library(pdp)
    partial(xgb_model_object, pred.var = "x_var_to_check", plot = TRUE, train = x_matrix)
    df <- partial(xgb_model_object, pred.var = "x_var_to_check", plot = FALSE, train = x_matrix)

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
                    colsample_bytree = 0.1,
                    colsample_bylevel = 1,
                    min_child_weight = 5,
                    objective = "reg:linear",
                    subsample = 0.632,
                    nthread = parallel::detectCores() - 1,
                    alpha = 10,
                    nrounds = 5000, # the best iteration is from xgb.cv calculation through same nfold
                    eta = 0.1,
                    max.depth = 5,
                    lambda = 2
                )
    gc()
    # LightGBM version
    full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx,], label = full[trainidx, logerror])
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx,], label = full[valididx, logerror])
    score_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[scoreidx,], label = full[scoreidx, logerror])
    param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 63, # 2**8 -1, interaction depth=5
                    learning_rate = 0.05,
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
                    bagging_fraction = 0.632,
                    bagging_freq = 1, # bagging_fraction every k iterations
                    # early_stopping_rounds = 295,
                    lambda_l1 = 10,
                    lambda_l2 = 2.56,
                    verbose = 1)
    gc()

    float_merge <- function(target_df, df2, merge_key, new_column, minimum_decimal_precision=4) {
        # this is to merge the target_df, df2 tables using merge_key, where merge_key column
        # is float
        # target_df is the data.table that has all records, so-called left merge
        # new_column is the column that will be brought into target_df
        # minimum_decimal_precision = 4 means the minimum requirement for considering a merge
            # is the 4th decimal place
        # sample input
        #target_df <- data.table(a = c(1.231, -0.9278347234, 0.348372))
        #df2 <- data.table(a = c(1.23, 0.348372327944, 0.34837), b=c(1,2,3))
        #merge_key <- "a"
        #new_column <- "b"
        #minimum_decimal_precision <- 4
        target_df[, rowidx := seq(1, nrow(target_df))]
        target_df[, paste0(new_column) := 0+NA]
        counter <- 1
        for (decimals in seq(16, minimum_decimal_precision, by = -1)) {
            target_df[, copy_column := round(get(merge_key), decimals)]
            df2[, copy_column := round(get(merge_key), decimals)]
            df2[, paste0("new", counter) := get(new_column)]
            df3 <- df2[, lapply(.SD, mean, na.rm=T), .SDcols=paste0("new", counter), by="copy_column"]
            target_df <- merge(target_df, df3[, c("copy_column",paste0("new", counter)), with=F], by="copy_column",
                                all.x=T, all.y=F)
            setkey(target_df, rowidx)
            target_df[is.na(get(new_column)), paste0(new_column) := get(paste0("new", counter))]
            target_df[, copy_column := NULL]
            target_df[, paste0("new", counter) := NULL]
            counter <- counter + 1
            if (target_df[, sum(is.na(get(new_column)))] == 0) {break}
        }
        target_df[, rowidx := NULL]
        return(target_df)

        # > setkey(DT1,x,time)
        # > DT1
        #    x time v
        # 1: a   10 1
        # 2: a   30 2
        # 3: a   60 3
        # 4: b   10 4
        # 5: b   30 5
        # 6: b   60 6
        # 7: c   10 7
        # 8: c   30 8
        # 9: c   60 9
        # > DT2
        #    x time
        # 1: a   17
        # 2: b   54
        # 3: c    3
        # > DT1[DT2,roll="nearest"]
        #    x time v
        # 1: a   17 1
        # 2: b   54 6
        # 3: c    3 7
    }

    resample_to_match_distribution <- function(vec, ref_vec) {
        # vec is the vector that will have the weight/density assigned by the
        #      reference vector that will serve to provide the density function
        #      for the vec vector.
        # ref_vec is the reference vector that will provide the density function
        #      as the reference vector
        require(data.table)
        # sample input
        #vec <- full[split=="train", logerror_pred_bygblinear_v1]
        #ref_vec <- full[split=="test", logerror_pred_bygblinear_v1]

        ref_vec_density <- density(ref_vec, adjust=0.01)
        ref_vec_density$x <- round(ref_vec_density$x, 8)
        ref_vec_density_df <- data.table(vec=ref_vec_density$x, weights=ref_vec_density$y)
        setkey(ref_vec_density_df, vec)

        vec_density <- density(vec, adjust=0.01)
        vec_density$x <- round(vec_density$x, 8)
        vec_density_df <- data.table(vec=vec_density$x, weights=vec_density$y)
        setkey(vec_density_df, vec)

        #resample.obs <- sample(ref_vec_density$x, length(vec), replace=TRUE, prob=ref_vec_density$y)

        vec_df <- data.table(vec = round(vec, 8), weights = 0, rowidx = seq(1, length(vec)))
        vec_df_ <- vec_density_df[vec_df[, "vec", with=F], roll="nearest"]
        setnames(vec_df_, "weights", "orig_weights")

        vec_df <- ref_vec_density_df[vec_df[, "vec", with=F], roll="nearest"]
        setnames(vec_df, "weights", "ref_vec_weights")
        vec_df[, orig_weights := vec_df_[, orig_weights]]

        vec_df[, rec_weight := (ref_vec_weights/sum(ref_vec_weights))/(orig_weights/sum(orig_weights))]
        return(vec_df)
    }


}
