setwd("E:/zillow_kaggle_2017/")

require(dplyr)
require(data.table)
require(xgboost)
require(lightgbm)
require(glmnet)
require(Matrix)
require(readr)


#####################
load("s3.0.1_line302.RData")  # load in the workspace from the file ss3.0.1
#####################

cap_num_var <- function(df, col, percentiles = c(0.5, 99.5)) {
    # cap the numeric varaible col at two ends
    if (class(df[, get(col)]) %in% c("numeric", "integer")) {
        twoends <- quantile(df[, get(col)], probs=0.01*percentiles, na.rm=T)
        df[, paste0(col) := pmin(twoends[2], pmax(twoends[1], df[, get(col)]))]
        return(df)
    } else {
        return(df)
    }
}


############################################################
# Add some new features
############################################################
full[, taxamount_byft := taxamount / finishedsquarefeet12]
full[, taxvaluedollarcnt_byft := taxvaluedollarcnt / finishedsquarefeet12]
full[, calculatedfinishedsquarefeet_byft := calculatedfinishedsquarefeet / finishedsquarefeet12]
full[, structuretaxvaluedollarcnt_byft := structuretaxvaluedollarcnt / finishedsquarefeet12]
full[, lotsizesquarefeet_byft := lotsizesquarefeet / finishedsquarefeet12]
full[, landtaxvaluedollarcnt_byft := landtaxvaluedollarcnt / finishedsquarefeet12]
full[, taxvaluedollarcnt_bytax := taxvaluedollarcnt / taxamount]
full[, calculatedfinishedsquarefeet_bytax := calculatedfinishedsquarefeet / taxamount]
full[, structuretaxvaluedollarcnt_bytax := structuretaxvaluedollarcnt / taxamount]
full[, lotsizesquarefeet_bytax := lotsizesquarefeet / taxamount]
full[, landtaxvaluedollarcnt_bytax := landtaxvaluedollarcnt / taxamount]

full[, taxamount_byft3 := 1/taxamount_byft]
full[, taxvaluedollarcnt_byft3 := 1/taxvaluedollarcnt_byft]
full[, calculatedfinishedsquarefeet_byft3 := 1/calculatedfinishedsquarefeet_byft]
full[, structuretaxvaluedollarcnt_byft3 := 1/structuretaxvaluedollarcnt_byft]
full[, lotsizesquarefeet_byft3 := 1/lotsizesquarefeet_byft]
full[, landtaxvaluedollarcnt_byft3 := 1/landtaxvaluedollarcnt_byft]
full[, taxvaluedollarcnt_bytax3 := 1/taxvaluedollarcnt_bytax]
full[, calculatedfinishedsquarefeet_bytax3 := 1/calculatedfinishedsquarefeet_bytax]
full[, structuretaxvaluedollarcnt_bytax3 := 1/structuretaxvaluedollarcnt_bytax]
full[, lotsizesquarefeet_bytax3 := 1/lotsizesquarefeet_bytax]
full[, landtaxvaluedollarcnt_bytax3 := 1/landtaxvaluedollarcnt_bytax]

full[, random_rnorm := rnorm(nrow(full))]

num_vars <- unique(c(num_vars, "taxamount_byft","taxvaluedollarcnt_byft",
                        "calculatedfinishedsquarefeet_byft",
                        "structuretaxvaluedollarcnt_byft",
                        "lotsizesquarefeet_byft", "landtaxvaluedollarcnt_byft",
                        "taxvaluedollarcnt_bytax", "calculatedfinishedsquarefeet_bytax",
                        "structuretaxvaluedollarcnt_bytax", "lotsizesquarefeet_bytax",
                        "landtaxvaluedollarcnt_bytax",
                        "taxamount_byft3","taxvaluedollarcnt_byft3",
                        "calculatedfinishedsquarefeet_byft3",
                        "structuretaxvaluedollarcnt_byft3",
                        "lotsizesquarefeet_byft3", "landtaxvaluedollarcnt_byft3",
                        "taxvaluedollarcnt_bytax3", "calculatedfinishedsquarefeet_bytax3",
                        "structuretaxvaluedollarcnt_bytax3", "lotsizesquarefeet_bytax3",
                        "landtaxvaluedollarcnt_bytax3",
                        "random_rnorm"))

# capping the two ends of new variable values
for (col in c("taxamount_byft","taxvaluedollarcnt_byft",
                        "calculatedfinishedsquarefeet_byft",
                        "structuretaxvaluedollarcnt_byft",
                        "lotsizesquarefeet_byft", "landtaxvaluedollarcnt_byft",
                        "taxvaluedollarcnt_bytax", "calculatedfinishedsquarefeet_bytax",
                        "structuretaxvaluedollarcnt_bytax", "lotsizesquarefeet_bytax",
                        "landtaxvaluedollarcnt_bytax",
                        "taxamount_byft3","taxvaluedollarcnt_byft3",
                        "calculatedfinishedsquarefeet_byft3",
                        "structuretaxvaluedollarcnt_byft3",
                        "lotsizesquarefeet_byft3", "landtaxvaluedollarcnt_byft3",
                        "taxvaluedollarcnt_bytax3", "calculatedfinishedsquarefeet_bytax3",
                        "structuretaxvaluedollarcnt_bytax3", "lotsizesquarefeet_bytax3",
                        "landtaxvaluedollarcnt_bytax3",
                        "random_rnorm")) {
    full <- cap_num_var(full, col, percentiles = c(0.5, 99.5))
}


predictors <- unique(c(predictors, num_vars))
num_vars <- copy(intersect(colnames(full), num_vars))
predictors <- copy(intersect(colnames(full), predictors))


############################################################
# XGB Propensity Model
############################################################
for (col in setdiff(pure_char_vars, month_vars)) {
    if (class(full[, get(col)]) %in% c("numeric", "integer")) {
        next
    } else {
        vec_ <- full[, sort(unique(get(col)))]
        full[, paste0(col) := factor(get(col), levels = vec_)]
    }
}

full_mm <- data.matrix(full[, predictors, with = F])
gc()

varstothrow <- c()
for (col in setdiff(pure_char_vars, "rawcensustractandblock")) {
    vec_ <- full[, sort(unique(get(col)))]
    full[, paste0(col) := factor(get(col), levels = vec_)]
    if (length(vec_) == 1) {varstothrow <- c(varstothrow, col)}
}

x_ <- full[, setdiff(predictors, c(varstothrow, "rawcensustractandblock")), with=F]
for (col in colnames(x_)) {
    if (class(x_[, get(col)]) %in% c("integer", "numeric")) {
        median_ <- quantile(as.vector(x_[, get(col)]), probs=0.5, na.rm=T)
        x_[is.na(get(col)), paste0(col) := median_]
    } else {
        x_[is.na(get(col)), paste0(col) := "NAN"]
    }
}
full_sparse <- Matrix::sparse.model.matrix(~.-1, x_)
rm(x_); gc()


# Loop through the calculation for the propensity of prediciton bins
percentiles_to_pred <- c(0.005, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 0.995)
buckets <- c(c(-Inf, Inf), quantile(full[, logerror], probs = setdiff(percentiles_to_pred, c(0,1)), na.rm=T))
buckets <- sort(unique(buckets))
full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]
bucket_freqs <- table(full[split=='train', logerror_bucket])
bucket_values <- sort(unique(full[split=="train", logerror_bucket]))


n_folds <- 20
setkey(full, rowidx)
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

for (bucketvalue in bucket_values) {
    #######################################################
    # Get Propensity
    #######################################################
    # TODO!!
    propensity_pred_name <- paste0("propensity_bucket", bucketvalue)
    target_pred_name <- paste0("target_value_bucket", bucketvalue)
    full[, paste0(propensity_pred_name) := 0.0]
    full[, paste0(target_pred_name) := 0.0]

    for (ifold in seq(1, n_folds)) {
        if (FALSE) {
            train_dm_reg_ <- xgb.DMatrix(
                            data = full_mm[which(full$split=='train'), ],
                            label = full[which(full$split=='train'), 1*(logerror_bucket == bucketvalue)],
                            missing = NA
                        )
            bestauc <- 0
            for (a in 0.05*(2 ** seq(0,12))){
                set.seed(10011)
                x_ <- xgb.cv(data = train_dm_reg_,
                            eval_metric = "auc",
                            print_every_n = 20,
                            nfold = n_folds,
                            early_stopping_rounds = 20,
                            colsample_bytree = 0.8,
                            min_child_weight = 2,
                            objective = "binary:logistic",
                            subsample = 0.85,
                            nthread = parallel::detectCores() - 1,
                            alpha = a,
                            nrounds = 5000,
                            eta = 0.1,
                            max.depth = 8,
                            lambda=1
                        )
                best_auc <- x_$evaluation_log[x_$best_iteration]$test_auc_mean
                best_iter <- x_$best_iteration
                if (best_auc > bestauc) {
                    bestauc <- best_auc
                    besta <- a
                    bestiter <- best_iter
                }
                gc()
            }
            print(paste0("Best AUC is ", bestauc, " with best a ",besta," at iteration ", bestiter, " for bucketvalue ", bucketvalue))
            }
        # after the tuning above, now let's score
        trainidx <- which(full$split=='train' & full$foldid != ifold)
        valididx <- which(full$split=='train' & full$foldid == ifold)
        scoreidx <- which(full$split=='test')
        train_dm_reg <- xgb.DMatrix(
                            data = full_mm[trainidx, ],
                            label = full[trainidx, 1*(logerror_bucket == bucketvalue)],
                            missing = NA
                        )
        valid_dm_reg <- xgb.DMatrix(
                            data = full_mm[valididx, ],
                            label = full[valididx, 1*(logerror_bucket == bucketvalue)],
                            missing = NA
                        )
        # After tuning parameters, the parameters are:
        if (bucketvalue == 0) {
                colsample_bytree=0.8; subsample=0.85; alpha=1; max.depth=8; lambda=1;
                nrounds=75;
            } else if (bucketvalue == 1) {
                colsample_bytree=0.8; subsample=0.85; alpha=0.20113; max.depth=8; lambda=1;
                nrounds=68;
            } else if (bucketvalue == 2) {
                colsample_bytree=0.8; subsample=0.85; alpha=0.4; max.depth=8; lambda=1;
                nrounds=64;
            } else if (bucketvalue == 3) {
                colsample_bytree=0.8; subsample=0.85; alpha=17.6; max.depth=8; lambda=1;
                nrounds=93;
            } else if (bucketvalue == 4) {
                colsample_bytree=0.8; subsample=0.85; alpha=51.2; max.depth=8; lambda=1;
                nrounds=177;
            } else if (bucketvalue == 5) {
                colsample_bytree=0.8; subsample=0.85; alpha=51.2; max.depth=8; lambda=1;
                nrounds=200;
            } else if (bucketvalue == 6) {
                colsample_bytree=0.8; subsample=0.85; alpha=25.6; max.depth=8; lambda=1;
                nrounds=145;
            } else if (bucketvalue == 7) {
                colsample_bytree=0.8; subsample=0.85; alpha=0.05; max.depth=8; lambda=1;
                nrounds=59;
            } else if (bucketvalue == 8) {
                colsample_bytree=0.8; subsample=0.85; alpha=0.1; max.depth=8; lambda=1;
                nrounds=46;
            } else {
                TRUE
            }
        set.seed(10011)
        xgb0_reg <- xgb.train(
                        data = train_dm_reg,
                        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "auc",
                        print_every_n = 50,
                        early_stopping_rounds = 20,
                        colsample_bytree = colsample_bytree,
                        min_child_weight = 2,
                        objective = "binary:logistic",
                        subsample = subsample,
                        nthread = parallel::detectCores() - 1,
                        alpha = alpha,
                        nrounds = nrounds,
                        eta = 0.1,
                        max.depth = max.depth,
                        lambda=lambda
                    )
        valid_pred_ <- predict(xgb0_reg, full_mm[valididx, ])
        full[valididx, paste0(propensity_pred_name) := valid_pred_]
        score_pred_ <- predict(xgb0_reg, full_mm[scoreidx, ])
        full[scoreidx, paste0(propensity_pred_name) := get(propensity_pred_name) + score_pred_]
        rm(train_dm_reg, valid_dm_reg); gc();
    }

    full[scoreidx, paste0(propensity_pred_name) := get(propensity_pred_name) / n_folds]
    # Need rebalancing the propensities due to the imbancing issue
    #adj_factor <- mean(full[split=='train', 1*(logerror_bucket == bucketvalue)]) / mean(full[split=='train', get(propensity_pred_name)])
    #full[split=='train', paste0(propensity_pred_name) := get(propensity_pred_name) * adj_factor]
    print(paste0("Done with bucketvalue ", bucketvalue))






















    #TODO!
    #######################################################
    # Get Severity
    #######################################################
    for (ifold in seq(1, n_folds)) {
        if (FALSE) {
            train_dm_reg_ <- xgb.DMatrix(
                            data = full_sparse[which(full$split=='train' & full$logerror_bucket == bucketvalue), ],
                            label = full[which(full$split=='train' & full$logerror_bucket == bucketvalue), logerror],
                            missing = NA
                        )
            best_mae <- 1000000
            for (a in 0.1*(1.15 ** seq(0,50))){
                set.seed(10011)
                x_ <- xgb.cv(data = train_dm_reg_,
                            eval_metric = "mae",
                            print_every_n = 100,
                            nfold = n_folds,
                            early_stopping_rounds = 20,
                            objective = "reg:linear",
                            nthread = parallel::detectCores() - 1,
                            alpha = a,
                            nrounds = 5000,
                            lambda=1
                        )
                bestmae <- x_$evaluation_log[x_$best_iteration]$test_mae_mean
                if (bestmae < best_mae) {
                    best_mae <- bestmae
                    best_iter <- x_$best_iteration
                    best_a <- a
                }
                gc()
            }
            print(paste0("Best MAE is ", best_mae, " with best a ",best_a," at iteration ", best_iter, " for bucketvalue ", bucketvalue))
        }
        # after the tuning above, now let's score
        trainidx <- which(full$split=='train' & full$logerror_bucket == bucketvalue & full$foldid != ifold)
        valididx <- which(full$split=='train' & full$foldid == ifold)
        scoreidx <- which(full$split=='test')
        train_dm_reg <- xgb.DMatrix(
                            data = full_sparse[trainidx, ],
                            label = full[trainidx, logerror],
                            missing = NA
                        )
        valid_dm_reg <- xgb.DMatrix(
                            data = full_sparse[valididx, ],
                            label = full[valididx, logerror],
                            missing = NA
                        )
        # After tuning parameters, the parameters are:
        if (bucketvalue == 0) {
                alpha=0.266; lambda=0.2; nrounds=7;
            } else if (bucketvalue == 1) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 2) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 3) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 4) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 5) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 6) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 7) {
                alpha=; lambda=; nrounds=;
            } else if (bucketvalue == 8) {
                alpha=; lambda=; nrounds=;
            } else {
                TRUE
            }
        set.seed(10011)
        xgb0_reg <- xgb.train(
                        data = train_dm_reg,
                        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "auc",
                        print_every_n = 50,
                        objective = "binary:logistic",
                        nthread = parallel::detectCores() - 1,
                        alpha = alpha,
                        nrounds = nrounds,
                        lambda=lambda
                    )
        valid_pred_ <- predict(xgb0_reg, full_sparse[valididx, ])
        full[valididx, propensity_pred_name := valid_pred_]
        score_pred_ <- predict(xgb0_reg, full_sparse[scoreidx, ])
        full[scoreidx, propensity_pred_name := propensity_pred_name + score_pred_]
        rm(train_dm_reg, valid_dm_reg); gc();
    }

    full[scoreidx, propensity_pred_name := propensity_pred_name / n_folds]
    print(paste0("Done with bucketvalue ", bucketvalue))

}






















freq_bin_cols <- paste0("freq_bin", seq(1, full[split=="train", length(unique(logerror_bucket))]))
for (col in freq_bin_cols) {
    full[, paste0(col) := 0.0]
}

n_folds <- 20
setkey(full, rowidx)
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

# Add Frequency
for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & !(full$rowidx %in% rowidx_to_throwout))
    valididx <- which(full$split=="train" & full$foldid == i)
    scoreidx <- which(full$split=="test")
    train_dm_freq <- xgb.DMatrix(
                    data = full_mm[trainidx, ],
                    label = full[trainidx, logerror_bucket],
                    missing = NA
                )
    valid_dm_freq <- xgb.DMatrix(
                    data = full_mm[valididx, ],
                    label = full[valididx, logerror_bucket],
                    missing = NA
                )
    test_dm_freq <- xgb.DMatrix(
                data = full_mm[scoreidx, ],
                missing = NA
            )
    set.seed(10011)
    xgb1_freq <- xgb.train(
                        data = train_dm_freq,
                        watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
                        eval_metric = "mlogloss",
						objective = "multi:softprob",
                        print_every_n = 20,
                        # early_stopping_rounds = 20,
						num_class = length(buckets)-1,
						colsample_bytree = 0.6,
						subsample = 0.632,
						nthread = parallel::detectCores() -1,
						alpha = 15,
						nrounds = 152,
						eta = 0.1,
						max.depth = 6,
						lambda = 1
                    )
    gc()
    pred_m_valid <- matrix(predict(xgb1_freq, valid_dm_freq), nrow = nrow(valid_dm_freq), byrow = TRUE)
    pred_m_valid <- data.table(pred_m_valid)
    setnames(pred_m_valid, colnames(pred_m_valid), freq_bin_cols)

    pred_m_test <- matrix(predict(xgb1_freq, test_dm_freq), nrow = nrow(test_dm_freq), byrow = TRUE)
    pred_m_test <- data.table(pred_m_test)
    setnames(pred_m_test, colnames(pred_m_test), freq_bin_cols)

    for (col in freq_bin_cols) {
        full[valididx, paste0(col) := pred_m_valid[, get(col)]]
        full[scoreidx, paste0(col) := get(col) + pred_m_test[, get(col)]]
    }
    rm(train_dm_freq, valid_dm_freq, test_dm_freq); gc();
}

for (col in freq_bin_cols) {
    full[full$split=="test", paste0(col) := get(col) / n_folds]
}







































############################################################
# GBLinear Model
############################################################
n_folds <- 40
setkey(full, rowidx)
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]


for (col in setdiff(pure_char_vars, month_vars)) {
    if (class(full[, get(col)]) %in% c("numeric", "integer")) {
        next
    } else {
        vec_ <- full[, sort(unique(get(col)))]
        full[, paste0(col) := factor(get(col), levels = vec_)]
    }
}

full_mm <- data.matrix(full[, predictors, with = F])
idx_ <- which(!(full$rowidx %in% rowidx_to_throwout))
gc()

# Missing Data Imputation using Median
for (col in colnames(full_mm)) {
    vec_ <- full_mm[, col]
    if (sum(is.na(vec_)) >0) {
        median_ <- quantile(vec_, probs=0.5, na.rm=T)
        vec_[is.na(vec_)] <- median_
        full_mm[, col] <- vec_
    }
}

# Check
if (sum(!complete.cases(full_mm)) > 0) {stop("full_mm still has some missing values")}
full_mm <- scale(data.matrix(full_mm))
gc()

trainidx <- which(full$split=="train" & !(full$rowidx %in% rowidx_to_throwout))
train_dm_reg <- xgb.DMatrix(
    data = full_mm[trainidx, ],
    label = full[trainidx, logerror],
    missing = NA
)

best_mae = 1000
best_alpha = 0
for (alpha_ in 10 *(1.5**seq(0,10))) {
    set.seed(10011)
    xgb0_reg <- xgb.cv(data = train_dm_reg,
                        booster ="gblinear",
                        nfold=20,
                        #watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "mae",
                        #feval = mae,
                        print_every_n = 50,
                        early_stopping_rounds = 15,
                        objective = "reg:linear",
                        nthread = parallel::detectCores() - 1,
                        alpha = 0.01,
                        nrounds = 1000,
                        lambda = 0.1)
    mae = xgb0_reg$evaluation_log$test_mae_mean[xgb0_reg$best_iteration]
    if (mae < best_mae) {best_alpha = alpha_ ; best_mae = mae}
}


n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bygblinear_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & !(full$rowidx %in% rowidx_to_throwout))
    train_dm_reg <- xgb.DMatrix(data = full_mm[trainidx, ],
                        label = full[trainidx, logerror],
                        missing = NA)
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(data = full_mm[valididx, ],
                        label = full[valididx, logerror],
                        missing = NA)
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    xgb2_reg <- xgb.train(data = train_dm_reg,
                        booster ="gblinear",
                        #nfold=20,
                        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "mae",
                        #feval = mae,
                        print_every_n = 50,
                        #early_stopping_rounds = 15,
                        objective = "reg:linear",
                        nthread = parallel::detectCores() - 1,
                        alpha = 0.01,
                        nrounds = 58,
                        lambda = 0.1)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-xgb2_reg$best_iter
    #best_valid_mae <- xgb2_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(xgb2_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bygblinear_v1 := valid_pred_]
    score_pred_ <- predict(xgb2_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bygblinear_v1 := logerror_pred_bygblinear_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bygblinear_v1 := logerror_pred_bygblinear_v1 / n_folds]
# The Public LB score is 0.0649143 for using logerror_pred_bygblinear_v1



############################################################
# XGBoost
############################################################
trainidx <- which(full$split=="train" & !(full$rowidx %in% rowidx_to_throwout))

full_mm_ <- cbind(full_mm, t(t(full[,logerror_pred_bygblinear_v1])))

train_dm_reg <- xgb.DMatrix(
    data = full_mm_[trainidx, ],
    label = full[trainidx, logerror],
    missing = NA
)

best_mae = 1000
best_alpha = 0
for (alpha_ in 0.01 *(2**seq(0,10))) {
    set.seed(10011)
    xgb0_reg <- xgb.cv(
                data = train_dm_reg,
                #watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                eval_metric = "mae",
                base_score = mean(full[trainidx, logerror]),
                #feval = mae,
                nfold = 40,
                print_every_n = 50,
                early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.3,
                colsample_bylevel = 1,
                min_child_weight = 2,
                objective = "reg:linear",
                subsample = 0.85,
                nthread = parallel::detectCores() - 1,
                alpha = 4,
                nrounds = 2000, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.02,
                max.depth = 8,
                lambda = 0.32
            )
    mae = xgb0_reg$evaluation_log$test_mae_mean[xgb0_reg$best_iteration]
    if (mae < best_mae) {best_alpha = alpha_ ; best_mae = mae}
    gc();gc();gc()
    # Best Iteration: 466; Best MAE: 0.040284+0.000522
}

n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_byxgboost_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & !(full$rowidx %in% rowidx_to_throwout))
    train_dm_reg <- xgb.DMatrix(data = full_mm_[trainidx, ],
                        label = full[trainidx, logerror],
                        missing = NA)
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(data = full_mm_[valididx, ],
                        label = full[valididx, logerror],
                        missing = NA)
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    xgb3_reg <- xgb.train(data = train_dm_reg,
                        #watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                        eval_metric = "mae",
                        base_score = mean(full[trainidx, logerror]),
                        #feval = mae,
                        print_every_n = 50,
                        # early_stopping_rounds = 50, # early stopping may cause overfitting
                        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        maximize = FALSE,
                        colsample_bytree = 0.3,
                        colsample_bylevel = 1,
                        min_child_weight = 2,
                        objective = "reg:linear",
                        subsample = 0.85,
                        nthread = parallel::detectCores() - 1,
                        alpha = 4,
                        nrounds = 466, # the best iteration is from xgb.cv calculation through same nfold
                        eta = 0.02,
                        max.depth = 8,
                        lambda = 0.32)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-xgb3_reg$best_iter
    #best_valid_mae <- xgb3_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(xgb3_reg, full_mm_[valididx, ])
    full[valididx, logerror_pred_byxgboost_v1 := valid_pred_]
    score_pred_ <- predict(xgb3_reg, full_mm_[scoreidx, ])
    full[scoreidx, logerror_pred_byxgboost_v1 := logerror_pred_byxgboost_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_byxgboost_v1 := logerror_pred_byxgboost_v1 / n_folds]
# The Public LB score is 0.0645819 for using logerror_pred_byxgboost_v1



############################################################
# LightGBM
############################################################
full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])

trainidx <- which(full$split=="train" & !(full$rowidx %in% rowidx_to_throwout))
train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, logerror])
# Get best iteration
param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 128, # 2**8 -1, interaction depth=5
                    learning_rate = 0.02,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
set.seed(10011)
lgb0_reg <- lgb.cv(params = param,
                    data = train_dm_reg,
                    #valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 1000,
                    nfold = 40,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 2,
                    feature_fraction = 0.85,
                    bagging_fraction = 0.65,
                    bagging_freq = 2, # bagging_fraction every k iterations
                    early_stopping_rounds = 50,
                    lambda_l1 = 0.5,
                    lambda_l2 = 100,
                    verbose = 1)
cat(paste0("Best Iteration: ",lgb0_reg$best_iter, "; Best MAE: ", lgb0_reg$record_evals$valid$l1$eval[[lgb0_reg$best_iter]]))
# Best Iteration: 713; Best MAE: 0.0402101851439317

n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i & !(full$rowidx %in% rowidx_to_throwout))
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, logerror])
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx, ],
                                label = full[valididx, logerror])
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 713,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 2,
                    feature_fraction = 0.85,
                    bagging_fraction = 0.65,
                    bagging_freq = 2, # bagging_fraction every k iterations
                    #early_stopping_rounds = 50,
                    lambda_l1 = 0.5,
                    lambda_l2 = 100,
                    verbose = 1)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bylgb_v1 := valid_pred_]
    score_pred_ <- predict(lgb1_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bylgb_v1 := logerror_pred_bylgb_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bylgb_v1 := logerror_pred_bylgb_v1 / n_folds]
rm(full_dm)  # The Public LB score is 0.0645255 for using logerror_pred_bylgb_v1





################################################################################
#   LEGACY CODE
################################################################################
if (FALSE) {
}
