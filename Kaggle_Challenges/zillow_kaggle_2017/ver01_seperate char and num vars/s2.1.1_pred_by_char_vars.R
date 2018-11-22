setwd("E:/zillow_kaggle_2017/")

require(dplyr)
require(data.table)
require(xgboost)
require(lightgbm)
require(glmnet)
require(Matrix)
require(readr)


#####################
load("s2.1.0_lastrow.RData")
#####################


############################################################
# XGBoost on Full Sparse with only Character Variables
############################################################
# tune parameters
trainidx <- which(full[, split=="train"])
train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, logerror],
                missing = NA
            )
set.seed(10011)
xgb0_reg <- xgb.cv(
                data = train_dm_reg,
                #watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                eval_metric = "mae",
                base_score = -0.1,
                #feval = mae,
                nfold = 40,
                print_every_n = 50,
                early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.07,
                colsample_bylevel = 1,
                min_child_weight = 2,
                objective = "reg:linear",
                subsample = 0.65,
                nthread = parallel::detectCores() - 1,
                alpha = 8,
                nrounds = 2000, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.02,
                max.depth = 5,
                lambda = 2
            )
#Stopping. Best iteration:
#[149]   train-mae:0.068007+0.000082 test-mae:0.068039+0.003136


n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_byxgb_charvars_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, logerror],
                missing = NA
    )
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
                data = full_sparse[valididx, ],
                label = full[valididx, logerror],
                missing = NA
    )
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    # pmin(0.4,pmax(-0.4, logerror))
    xgb1_reg <- xgb.train(
                data = train_dm_reg,
                watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                eval_metric = "mae",
                base_score = -0.1,
                #feval = mae,
                print_every_n = 50,
                early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.07,
                colsample_bylevel = 1,
                min_child_weight = 2,
                objective = "reg:linear",
                subsample = 0.65,
                nthread = parallel::detectCores() - 1,
                alpha = 8,
                nrounds = 149, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.02,
                max.depth = 5,
                lambda = 2
    )
    gc(); gc()
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    valid_pred_ <- predict(xgb1_reg, full_sparse[valididx, ])
    full[valididx, logerror_pred_byxgb_charvars_v1 := valid_pred_]
    score_pred_ <- predict(xgb1_reg, full_sparse[scoreidx, ])
    full[scoreidx, logerror_pred_byxgb_charvars_v1 := logerror_pred_byxgb_charvars_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_byxgb_charvars_v1 := logerror_pred_byxgb_charvars_v1 / n_folds]
xgb_import_vars <- xgb.importance(feature_names = colnames(full_sparse), xgb1_reg);  gc(); gc();
# The Public LB score is 0.0653197 for using logerror_pred_bylgb_charvars_v1



############################################################
# LightGBM on Full Sparse with only Character Variables
############################################################
# tune parameters
setkey(full, rowidx)
full_mm <- data.matrix(full[, char_predictors, with = F]) * 1.0  # convert integer to float
full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror])

trainidx <- which(full[, split=="train"])
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
                    lambda_l1 = 12,
                    lambda_l2 = 100,
                    verbose = 1)
cat(paste0("Best Iteration: ",lgb0_reg$best_iter, "; Best MAE: ", lgb0_reg$record_evals$valid$l1$eval[[lgb0_reg$best_iter]]))
# Best Iteration: 152; Best MAE: 0.0678640320908229

n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_charvars_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
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
                    nrounds = 152,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 2,
                    feature_fraction = 0.85,
                    bagging_fraction = 0.65,
                    bagging_freq = 2, # bagging_fraction every k iterations
                    #early_stopping_rounds = 50,
                    lambda_l1 = 12,
                    lambda_l2 = 100,
                    verbose = 1)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bylgb_charvars_v1 := valid_pred_]
    score_pred_ <- predict(lgb1_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bylgb_charvars_v1 := logerror_pred_bylgb_charvars_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bylgb_charvars_v1 := logerror_pred_bylgb_charvars_v1 / n_folds]
rm(full_dm)  # The Public LB score is 0.0649320 for using logerror_pred_bylgb_charvars_v1



############################################################
# GBLinear on Full Sparse with only Character Variables
############################################################
# Missing data imputaiton for full_sparse
# scale the sparse matrix
colSdColMeans <- function(x) {
  n <- nrow(x)
  colVar <- colMeans(x*x) - (colMeans(x))^2
  return(sqrt(colVar * n/(n-1)))
}

scale_sparse_matrix <- function(x) {
    # This is to scale the sparse matrix without centering
    stds <- colSdColMeans(x)
    stds[stds == 0] <- 1  # avoid breaking the program
    return(t(t(x) / stds))
}

nacolumns <- intersect(colnames(full_sparse), char_predictors)
for (col in nacolumns) {
    x_ <- full_sparse[, col]
    if (sum(is.na(x_)) == 0) {
        next
    } else {
        median_ <- quantile(x_, probs = 0.5, na.rm = T)
        full_sparse <- full_sparse[, setdiff(colnames(full_sparse), col)]
        x_[is.na(x_)] <- median_
        x2_ <- as(t(t(x_)), "sparseMatrix")
        colnames(x2_) <- col
        full_sparse <- cbind(full_sparse, x2_)
        rm(x2_, x_); gc();
    }
}

full_sparse <- scale_sparse_matrix(full_sparse)

trainidx <- which(full[, split=="train"])
train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, pmin(0.35, pmax(-0.35, logerror))],
                missing = NA
            )
best_alpha = 0 ; best_mae = 1000;
for (alpha_ in 2*(2 ** seq(0,10))) {
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
                    alpha = 128, #alpha_
                    nrounds = 1000,
                    lambda = 102)
    mae_ <- xgb0_reg$evaluation_log$test_mae_mean[xgb0_reg$best_iteration]
    if (mae_ < best_mae) {best_alpha <- alpha_ ; best_mae <- mae_}
}

n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bygblinear_charvars_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, pmin(0.35, pmax(-0.35, logerror))],
                missing = NA
            )
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
                data = full_sparse[valididx, ],
                label = full[valididx, pmin(0.35, pmax(-0.35, logerror))],
                missing = NA
            )
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    gblinear0_reg <- xgb.train(data = train_dm_reg,
                            booster ="gblinear",
                            #nfold=20,
                            watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                            eval_metric = "mae",
                            #feval = mae,
                            print_every_n = 50,
                            #early_stopping_rounds = 15,
                            objective = "reg:linear",
                            nthread = parallel::detectCores() - 1,
                            alpha = 128,
                            nrounds = 27,
                            lambda = 102)
    gc(); gc(); gc();
    #best_iter <-gblinear0_reg$best_iter
    #best_valid_mae <- gblinear0_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(gblinear0_reg, full_sparse[valididx, ])
    full[valididx, logerror_pred_bygblinear_charvars_v1 := valid_pred_]
    score_pred_ <- predict(gblinear0_reg, full_sparse[scoreidx, ])
    full[scoreidx, logerror_pred_bygblinear_charvars_v1 := logerror_pred_bygblinear_charvars_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bygblinear_charvars_v1 := logerror_pred_bygblinear_charvars_v1 / n_folds]
# The Public LB score is 0.0649857 for using logerror_pred_bygblinear_charvars_v1



############################################################
# XGBoost on Full Sparse with only Character Variables
############################################################
# tune parameters
trainidx <- which(full[, split=="train"])
train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, pmin(0.35, pmax(-0.35,logerror))],
                missing = NA
            )
best_alpha = 0 ; best_mae = 1000;
for (alpha_ in 0.5*(1.15 ** seq(0,20))) {
    set.seed(10011)
    xgb0_reg <- xgb.cv(
                data = train_dm_reg,
                #watchlist = list(train = train_dm_reg, valid = train_dm_reg),
                eval_metric = "mae",
                base_score = 0.01,
                #feval = mae,
                nfold = 40,
                print_every_n = 50,
                early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.07,
                colsample_bylevel = 1,
                min_child_weight = 2,
                objective = "reg:linear",
                subsample = 0.65,
                nthread = parallel::detectCores() - 1,
                alpha = 1.156,
                nrounds = 2000, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.02,
                max.depth = 5,
                lambda = 0.08
            )
    mae_ <- xgb0_reg$evaluation_log$test_mae_mean[xgb0_reg$best_iteration]
    if (mae_ < best_mae) {best_alpha <- alpha_ ; best_mae <- mae_}
    gc()
}

#Stopping. Best iteration:
#[859]   train-mae:0.058083+0.000040 test-mae:0.058471+0.001460


n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_byxgb_charvars_v2 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(
                data = full_sparse[trainidx, ],
                label = full[trainidx, pmin(0.35, pmax(-0.35,logerror))],
                missing = NA
    )
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
                data = full_sparse[valididx, ],
                label = full[valididx, pmin(0.35, pmax(-0.35,logerror))],
                missing = NA
    )
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    # pmin(0.4,pmax(-0.4, logerror))
    xgb1_reg <- xgb.train(
                data = train_dm_reg,
                watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                eval_metric = "mae",
                base_score = 0.01,
                #feval = mae,
                print_every_n = 50,
                #early_stopping_rounds = 50, # early stopping may cause overfitting
                maximize = FALSE,
                colsample_bytree = 0.07,
                colsample_bylevel = 1,
                min_child_weight = 2,
                objective = "reg:linear",
                subsample = 0.65,
                nthread = parallel::detectCores() - 1,
                alpha = 1.156,
                nrounds = 859, # the best iteration is from xgb.cv calculation through same nfold
                eta = 0.02,
                max.depth = 5,
                lambda = 0.08
    )
    gc(); gc()
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    valid_pred_ <- predict(xgb1_reg, full_sparse[valididx, ])
    full[valididx, logerror_pred_byxgb_charvars_v2 := valid_pred_]
    score_pred_ <- predict(xgb1_reg, full_sparse[scoreidx, ])
    full[scoreidx, logerror_pred_byxgb_charvars_v2 := logerror_pred_byxgb_charvars_v2 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_byxgb_charvars_v2 := logerror_pred_byxgb_charvars_v2 / n_folds]
xgb_import_vars <- xgb.importance(feature_names = colnames(full_sparse), xgb1_reg);  gc(); gc();
# The Public LB score is 0.0647248 for using logerror_pred_byxgb_charvars_v2



############################################################
# LightGBM on Full Sparse with only Character Variables
############################################################
# tune parameters
setkey(full, rowidx)
full_mm <- data.matrix(full[, char_predictors, with = F]) * 1.0  # convert integer to float
full_dm <- lgb.Dataset(data = full_mm, label = full[, pmin(0.35, pmax(-0.35,logerror))])

trainidx <- which(full[, split=="train"])
train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, pmin(0.35, pmax(-0.35,logerror))])
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
                    lambda_l1 = 5,
                    lambda_l2 = 5,
                    verbose = 1)
cat(paste0("Best Iteration: ",lgb0_reg$best_iter, "; Best MAE: ", lgb0_reg$record_evals$valid$l1$eval[[lgb0_reg$best_iter]]))
# Best Iteration: 229; Best MAE: 0.0583934129590724

n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_charvars_v2 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, pmin(0.35, pmax(-0.35,logerror))])
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx, ],
                                label = full[valididx, pmin(0.35, pmax(-0.35,logerror))])
    scoreidx <- which(full$split=="test")
    set.seed(10011)
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 229,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 2,
                    feature_fraction = 0.85,
                    bagging_fraction = 0.65,
                    bagging_freq = 2, # bagging_fraction every k iterations
                    #early_stopping_rounds = 50,
                    lambda_l1 = 5,
                    lambda_l2 = 5,
                    verbose = 1)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-lgb1_reg$best_iter
    #best_valid_mae <- lgb1_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
    full[valididx, logerror_pred_bylgb_charvars_v2 := valid_pred_]
    score_pred_ <- predict(lgb1_reg, full_mm[scoreidx, ])
    full[scoreidx, logerror_pred_bylgb_charvars_v2 := logerror_pred_bylgb_charvars_v2 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bylgb_charvars_v2 := logerror_pred_bylgb_charvars_v2 / n_folds]
rm(full_dm)  # The Public LB score is 0.0648572 for using logerror_pred_bylgb_charvars_v2



############################################################
# Mix Results for only Character Variables
############################################################
full[, logerror_pred_mix_charvars_v1 := 0.86*(0.5*logerror_pred_bylgb_charvars_v2 + 0.5*logerror_pred_byxgb_charvars_v2) +
                    0.14*logerror_pred_bygblinear_charvars_v1]
# The Public LB score is 0.0646942 for using logerror_pred_bylgb_charvars_v2

################################################################################
#   LEGACY CODE
################################################################################
if (FALSE) {
}
