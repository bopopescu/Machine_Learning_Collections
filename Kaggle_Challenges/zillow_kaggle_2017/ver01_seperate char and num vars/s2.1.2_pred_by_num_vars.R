setwd("E:/zillow_kaggle_2017/")

require(dplyr)
require(data.table)
require(xgboost)
require(lightgbm)
require(glmnet)
require(Matrix)
require(readr)


#####################
load("s2.2.1_line514.RData")  # load in the workspace from the file s2.0.1_data confidence interval.R
#####################


############################################################
# Logerror Propensities Prediction
############################################################
#TODO

# Add some new features
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

num_predictors2 <- unique(c(num_predictors, "taxamount_byft","taxvaluedollarcnt_byft",
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

n_folds <- 40
setkey(full, rowidx)
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

full_temp <- copy(full[, num_predictors2, with = F])
vars_to_throwout <- NULL
for (col in num_predictors2) {
    if (full[!is.na(get(col)), length(unique(get(col)))] == 1){vars_to_throwout <- c(vars_to_throwout, col)}
    median_ <- quantile(full[, get(col)], probs = 0.5, na.rm=T)
    full_temp[is.na(get(col)), paste0(col) := median_ + 0.0]
    gc();gc();
}

full_temp <- full_temp[, setdiff(colnames(full_temp), vars_to_throwout), with = F]
gc()

# Check
if (sum(!complete.cases(full_temp)) > 0) {stop("full_temp still has some missing values")}
full_temp <- scale(data.matrix(full_temp))

trainidx <- which(full$split=="train")
train_dm_reg <- xgb.DMatrix(
    data = full_temp[trainidx, ],
    label = full[trainidx, pmin(0.1, pmax(-0.1,logerror)) - logerror_pred_mix_charvars_v1],
    missing = NA
)

# pmin(0.4,pmax(-0.4, logerror))
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
                        alpha = 179,
                        nrounds = 1000,
                        lambda = 1)
    mae = xgb0_reg$evaluation_log$test_mae_mean[xgb0_reg$best_iteration]
    if (mae < best_mae) {best_alpha = alpha_ ; best_mae = mae}
}


n_folds <- 40
set.seed(10011)
setkey(full, rowidx)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bygblinear_numvars_v1 := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- xgb.DMatrix(data = full_temp[trainidx, ],
                        label = full[trainidx, pmin(2, pmax(-2.0,logerror))-logerror_pred_mix_charvars_v1],
                        missing = NA)
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- xgb.DMatrix(data = full_temp[valididx, ],
                        label = full[valididx, pmin(2.0, pmax(-2.0,logerror))-logerror_pred_mix_charvars_v1],
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
                        alpha = 179,
                        nrounds = 1,
                        lambda = 1)
    gc(); gc(); gc();
    print(paste0("Processing fold ", i, " of ", n_folds, " folds."))
    #best_iter <-xgb2_reg$best_iter
    #best_valid_mae <- xgb2_reg$record_evals$valid$l1$eval[[best_iter]]
    #print(paste0("best_iter is ", best_iter, " and best valid mae is ",best_valid_mae))
    #print([1] "best_iter is 107 and best valid mae is 0.0676140437969164")
    valid_pred_ <- predict(xgb2_reg, full_temp[valididx, ])
    full[valididx, logerror_pred_bygblinear_numvars_v1 := valid_pred_]
    score_pred_ <- predict(xgb2_reg, full_temp[scoreidx, ])
    full[scoreidx, logerror_pred_bygblinear_numvars_v1 := logerror_pred_bygblinear_numvars_v1 + score_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}

full[scoreidx, logerror_pred_bygblinear_numvars_v1 := logerror_pred_bygblinear_numvars_v1 / n_folds]
rm(full_dm)  # The Public LB score is 0.0648572 for using logerror_pred_bygblinear_numvars_v1


################################################################################
#   LEGACY CODE
################################################################################
if (FALSE) {
}
