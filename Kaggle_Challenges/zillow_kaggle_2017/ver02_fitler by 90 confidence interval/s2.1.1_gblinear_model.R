setwd("E:/zillow_kaggle_2017/")

require(dplyr)
require(data.table)
require(xgboost)
require(lightgbm)
require(glmnet)
require(Matrix)
require(readr)


#####################
load("s2.0.1_line270.RData")  # load in the workspace from the file s2.0.1_data confidence interval.R
#####################


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
predictors <- unique(c(predictors, num_vars))
num_vars <- copy(intersect(colnames(full), num_vars))
predictors <- copy(intersect(colnames(full), predictors))


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
                        base_score = 0,
                        #watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "mae",
                        #feval = mae,
                        print_every_n = 50,
                        early_stopping_rounds = 15,
                        objective = "reg:linear",
                        nthread = parallel::detectCores() - 1,
                        alpha = 20,
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
                        base_score = 0,
                        watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                        eval_metric = "mae",
                        #feval = mae,
                        print_every_n = 50,
                        #early_stopping_rounds = 15,
                        objective = "reg:linear",
                        nthread = parallel::detectCores() - 1,
                        alpha = 20,
                        nrounds = 9,
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
