setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(lightgbm)
library(readr)
library(lubridate)
library(Matrix)
library(pdp)


load("s2_line415.RData")



objectSizeList <- function(...) {
    # return object sizes in the workspace
    # sample usage:
        # objectSizeList()
    objs_ <- ls(envir = globalenv())
    for (obj_ in objs_) {
        sz <- object.size(get(obj_))
        sz2 <- utils:::format.object_size(sz, "auto")
        if (!exists("object_size_chart")) {
            object_size_chart <- data.frame(object=obj_, size=sz2, size_byte=c(as.numeric(sz)))
        } else {
            object_size_chart <- rbind(object_size_chart,
                                       data.frame(object=obj_, size=sz2, size_byte=c(as.numeric(sz))))
        }
    }

    object_size_chart$size_byte <- as.numeric(as.vector(object_size_chart$size_byte))
    object_size_chart2 <- object_size_chart[with(object_size_chart, order(-size_byte)) ,]
    return(object_size_chart2)
}

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

#  DISABLE IT
if (FALSE) {
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
}


################ First Round of Regression Adjusting other Quarters of Data #############
# remove some variables to start new round of model fitting
conver2factor <- function(df, char_vars) {
    # This function convers the specified char_vars to factors
    for (col in char_vars) {
        df[, paste0(col) := as.character(get(col))]
        values <- sort(df[, unique(get(col))])
        df[, paste0(col) := factor(get(col), levels = values)]
    }
    return(df)
}


log_char_vars <- function(df, char_vars_tolog) {
    # This function convers the specified char_vars to factors
    options(warn=-1)
    new_vars_added <- c()
    res <- list()
    for (col_ in char_vars_tolog) {
        col <- paste0(col_, "_copy")
        df[, paste0(col) := as.numeric(get(col_))]
        if (sum(!is.na(as.numeric(df[, get(col)]))) > 0) {
            if (!paste0(col_, "_log") %in% colnames(df)) {
                df[, paste0(col_, "_log") := log(get(col))]
                new_vars_added <- unique(c(new_vars_added, paste0(col_, "_log")))
            }
        }
        df[, paste0(col) := NULL]
    }
    options(warn=0)
    res$df <- df
    res$added_vars <- new_vars_added
    return(res)
}


factorNA <- function(df) {
    # this function makes factor levels NA to "NaN"
    for (col in colnames(df)) {
        if (class(df[, get(col)]) == "factor") {
            df[is.na(get(col)), paste0(col) := "NaN"]
        }
    }
    return(df)
}


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

char_vars <- c("airconditioningtypeid", "bathroomcnt", "bedroomcnt", "buildingclasstypeid",
                    "buildingqualitytypeid", "calculatedbathnbr", "censustractandblock",
                    "censustractandblock_1to12", "censustractandblock_1to4", "censustractandblock_1to8",
                    "fullbathcnt", "hashottuborspa", "heatingorsystemtypeid", "poolcnt",
                    "pooltypeid7", "propertycountylandusecode", "propertycountylandusecode_1to2",
                    "propertylandusetypeid", "propertyzoningdesc", "propertyzoningdesc_1to3",
                    "propertyzoningdesc_1to4", "propertyzoningdesc_5to10", "regionidcity",
                    "regionidneighborhood", "regionidzip", "roomcnt", "taxdelinquencyflag",
                    "taxdelinquencyyear", "transactionMonth", "unitcnt", "yearbuilt")
char_vars_tolog <- c("airconditioningtypeid", "buildingclasstypeid", "buildingqualitytypeid", "calculatedbathnbr",
                        "censustractandblock", "censustractandblock_1to12", "censustractandblock_1to4",
                        "censustractandblock_1to8", "heatingorsystemtypeid", "propertylandusetypeid",
                        "regionidcity", "regionidneighborhood")
full <- conver2factor(full, char_vars)
res_ <- log_char_vars(full, char_vars_tolog)
full <- res_$df
predictors <- unique(c(predictors, res_$added_vars))
full <- factorNA(full)
rm(res_); gc();

previous_na_action <- options('na.action')
options(na.action='na.pass')
predictors_ <- intersect(setdiff(predictors, c(non_predictors)), colnames(full))
full_sparse <- sparse.model.matrix(~.-1, full[, predictors_, with=F])
options(na.action=previous_na_action$na.action)
gc()


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
rm(full_mm, full_sparse); gc();


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

    predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
    predictors_ <- setdiff(predictors_, month_vars)  # remove the month variables
    full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float
    full_dm <- lgb.Dataset(data = full_mm, label = full[, get(new_target)])
    # Get best iteration
    scale_pos_weight <- full[split=="train", sum(get(new_target) ==0)/sum(get(new_target))]
    trainidx <- which(full$split=="train")
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, get(new_target)])
    param <- list(objective = "binary", metric="auc", num_leaves = 63, learning_rate = 0.1,
                    boost_from_average = TRUE, boosting_type = 'dart')
    set.seed(10011)
    x_ <- lgb.cv(params = param,
                        data = train_dm_reg,
                        nrounds = 5000,
                        nfold = n_folds,
                        #device = "cpu", # or use gpu if it's available
                        scale_pos_weight = scale_pos_weight,
                        num_threads = parallel::detectCores() - 1,
                        min_data = 2,
                        feature_fraction = 0.6,
                        bagging_fraction = 0.85,
                        bagging_freq = 1, # bagging_fraction every k iterations
                        early_stopping_rounds = 100,
                        lambda_l1 = 0.1,
                        lambda_l2 = 1,
                        verbose = 1)
    nrounds <- x_$best_iter
    for (i in seq(1, n_folds)) {
        trainidx <- which(full$split=="train" & full$foldid != i)
        train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, get(new_target)])
        valididx <- which(full$split=="train" & full$foldid == i)
        valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx, ],
                                label = full[valididx, get(new_target)])
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
                            nrounds = nrounds,
                            #nfold = 8,
                            #device = "cpu", # or use gpu if it's available
                            scale_pos_weight = scale_pos_weight,
                            num_threads = parallel::detectCores() - 1,
                            min_data = 2,
                            feature_fraction = 0.6,
                            bagging_fraction = 0.85,
                            bagging_freq = 1, # bagging_fraction every k iterations
                            #early_stopping_rounds = 50,
                            lambda_l1 = 0.1,
                            lambda_l2 = 1,
                            verbose = 1)
        gc(); gc(); gc();

        valid_pred_ <- predict(lgb1_reg, full_mm[valididx, ])
        test_pred_ <- predict(lgb1_reg, full_mm[full$split=="test", ])

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
        alpha = 623,
        nrounds = 6, # the best iteration is from xgb.cv calculation through same nfold
        lambda = 0.8  #(if use mae, alpha lambda are 600, 1000, if use mse, alpha lambda are 25.6, 12.8)
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



#########################  Model Fitting after using Month Trend as offset  ###############
n_folds <- 20
set.seed(10101)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full[, logerror_pred_bylgb_v1 := 0.0]

if (FALSE) {
    # The sparser version's prediction is way too slow for lightgbm
    month_related_vars_in_fullsparse <- colnames(full_sparse)[grepl("trans*", colnames(full_sparse))]
    predictors_ <- setdiff(colnames(full_sparse), month_related_vars_in_fullsparse)
    full_dm <- lgb.Dataset(data = full_sparse[,predictors_], label = full[, logerror - transactionMonth_bias])
}
gc();

predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
predictors_ <- setdiff(predictors_, month_vars)
full_mm <- data.matrix(full[, predictors_, with =F])
full_dm <- lgb.Dataset(data = full_mm, label = full[, logerror - transactionMonth_bias])

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[trainidx, ],
                                label = full[trainidx, logerror - transactionMonth_bias])
    valididx <- which(full$split=="train" & full$foldid == i)
    valid_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[valididx, ],
                                label = full[valididx, logerror - transactionMonth_bias])
    scoreidx <- which(full$split=="test")
    #score_dm_reg <- lgb.Dataset.create.valid(full_dm, full_mm[scoreidx, ], label = full[scoreidx, logerror])
    param <- list(objective = "regression",
                    metric="l1",
                    num_leaves = 16, # 2**8 -1, interaction depth=5
                    learning_rate = 0.05,
                    boost_from_average = TRUE,
                    boosting_type = 'dart')  #gbdt is another choice
    set.seed(10011)
    # pmin(0.382,pmax(-0.3425, logerror))
    lgb1_reg <- lgb.train(params = param,
                    data = train_dm_reg,
                    valids = list(train=train_dm_reg, valid=valid_dm_reg),
                    nrounds = 61,
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
    gc(); gc(); gc();
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
rm(full_dm)
# full[, logerror_pred_bylgb_v1 := logerror_pred_bylgb_v1 + transactionMonth_bias]
gc(); gc()



####################  Final Minor Global Bias Tuning  ##################
global_bias_to_add2mae <- function(target_vec, prediction_vec) {
    # This function is to find the global bias
    #target_vec <- full[split=="train", logerror - transactionMonth_bias]
    #prediction_vec <- full[split=="train", logerror_pred_bylgb_v1]
    best_mae <- mean(abs(target_vec - prediction_vec))
    best_mae0 <- best_mae+0
    best_adj <- 0
    for (adj in seq(-0.01, 0.01, by=0.00001)) {
        mae_ <- mean(abs(target_vec - (prediction_vec + adj)))
        if (mae_ < best_mae) {
            best_mae <- mae_
            best_adj <- adj
        }
    }
    print(paste0("Best MAE is changed from ", best_mae0, "to ", best_mae, " with adjustment ", best_adj))
    return(best_adj)
}

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
full[, logerror_pred_bylgb_v2 := logerror_pred_bylgb_v1 + transactionMonth_bias]
full[, logerror_pred_bygblinear_v2 := logerror_pred_bygblinear_v1 + transactionMonth_bias]

w1 <- ensemble_adjust(full[split=="train", logerror_pred_bylgb_v2 - transactionMonth_bias],
                full[split=="train", logerror_pred_bygblinear_v2 - transactionMonth_bias],
                full[split=="train", logerror - transactionMonth_bias], measure_function = mae, minimum = TRUE)
full[, logerror_pred := 0.55 * logerror_pred_bylgb_v2 + (1- 0.55) * logerror_pred_bygblinear_v2]


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
    submission[, 2:7] <- round(submission[, 2:7], 4)
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
}
