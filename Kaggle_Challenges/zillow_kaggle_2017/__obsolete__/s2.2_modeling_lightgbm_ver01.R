setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(lightgbm)
library(readr)


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


##################  Test Train split Propensity model  ##################
predictors_ <- intersect(setdiff(predictors, c(month_vars, non_predictors)),
                            colnames(full))
full_small <- unique(full[, unique(c(predictors_, "parcelid", "split")), with=F])
# throw out the parcelids in the train since test is more comprehensive
idx_ <- which(full_small$split=="test" & full_small$parcelid %in% full_small[split=="train", parcelid])
full_small <- full_small[-idx_]
full_mm <- data.matrix(full_small[, predictors_, with = F]) * 1.0  # convert integer to float
n_folds <- 5
set.seed(10011)
full_small[, random := runif(nrow(full_small))]
full_small[, foldid := ceiling(random / (1/n_folds))]
full_small[, random := NULL]
full_small[, test_split_propensity_by_lgb := 0.99999]
scale_pos_weight <- nrow(full_small[split=="train"])/nrow(full_small[split=="test"])

for (i in seq(1, n_folds)) {
    train_dm_reg <- xgb.DMatrix(
                    data = full_mm[full_small$foldid != i, ],
                    label = full_small[foldid != i, 1*(split == "test")],
                    missing = NA
                )
    valid_dm_reg <- xgb.DMatrix(
                    data = full_mm[full_small$foldid == i, ],
                    label = full_small[foldid == i, 1*(split=="test")],
                    missing = NA
                )
    set.seed(10011)
    xgb0_reg <- xgb.train(
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "auc",
                    print_every_n = 50,
                    early_stopping_rounds = 10,
                    colsample_bytree = 0.8,
                    min_child_weight =5,
                    scale_pos_weight = scale_pos_weight,
                    objective = "binary:logistic",
                    subsample = 0.85,
                    nthread = parallel::detectCores() - 1,
                    alpha = 1,
                    nrounds = 5000, # the best iteration is at 92
                    eta = 0.1,
                    max.depth = 5,
                    lambda=1
                )
    gc()

    # print(paste0("Best GBM iteration is ", best.iter))
    valid_pred_ <- predict(xgb0_reg, full_mm[full_small$foldid == i, ])
    full_small[foldid == i, test_split_propensity_by_lgb := valid_pred_]
    rm(train_dm_reg, valid_dm_reg); gc();
}


# apply weight formula
full_small[, rec_weight_bylgb := test_split_propensity_by_lgb/(1-test_split_propensity_by_lgb)]
full <- merge(full,
                full_small[, c("parcelid", "test_split_propensity_by_lgb", "rec_weight_bylgb"), with = F],
                by=c("parcelid"))
rm(full_small); gc()


#####################################################  Regression Model  ##################
setkey(full, rowidx)

n_folds <- 20
set.seed(10011)
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]

#new_predictors <- unique(c(new_predictors, "logerror_pred_byxgboost"))  # add xgboost prediction in the predictors

if ("logerror_pred" %in% colnames(full)) {setnames(full, "logerror_pred", "logerror_pred_byxgboost")}
full[, logerror_pred_bylgb := 0.0]

predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- lgb.Dataset(
                    data = full_mm[trainidx, ],
                    label = full[trainidx, pmin(0.382,pmax(-0.3425, logerror))],
                    weight = rep(1, length(trainidx)) # full[trainidx, rec_weight_bylgb]
                )
    valididx <- which(full$split=="train" & full$foldid == i & full$transactionMonth >= 10)
    valid_dm_reg <- lgb.Dataset(
                    data = full_mm[valididx, ],
                    label = full[valididx, pmin(0.382,pmax(-0.3425, logerror))],
                    weight = rep(1, length(valididx))
                )
    test_dm_reg <- lgb.Dataset(
                    data = full_mm[full$split=="test", ]
                )
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
                    nrounds = 663,
                    #nfold = 8,
                    #device = "cpu", # or use gpu if it's available
                    num_threads = parallel::detectCores() - 1,
                    min_data = 5,
                    feature_fraction = 0.1,
                    bagging_fraction = 0.8,
                    bagging_freq = 1, # bagging_fraction every k iterations
                    # early_stopping_rounds = 295,
                    lambda_l1 = 20.48,
                    lambda_l2 = 2.56,
                    verbose = 1)
    gc()

    best.iter <- lgb1_reg$best_iter
    # print(paste0("Best GBM iteration is ", best.iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[full$split=="train" & full$foldid == i, ])
    full[split=="train" & foldid == i, logerror_pred_bylgb := valid_pred_]
    test_pred_ <- predict(lgb1_reg, full_mm[full$split=="test", ])
    full[split=="test", logerror_pred_bylgb := logerror_pred_bylgb + test_pred_]

    rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
}

full[split=="test", logerror_pred_bylgb := logerror_pred_bylgb / n_folds]

# check mae of logerror with adjusted weight
full[split=="train" & transactionMonth >=10,
     weighted.mean(abs(logerror - logerror_pred_bylgb), rec_weight_bylgb)]


######## Fit One Model instead of CV version  ############
trainidx <- which(full$split=="train")
train_dm_reg <- lgb.Dataset(
                data = full_mm[trainidx, ],
                label = full[trainidx, pmin(0.382,pmax(-0.3425, logerror))],
                weight = rep(1, length(trainidx)) # full[trainidx, rec_weight_bylgb]
            )
set.seed(10011)
# pmin(0.382,pmax(-0.3425, logerror))
lgb1_reg_noCV <- lgb.train(params = param,
                            data = train_dm_reg,
                            nrounds = 663,
                            num_threads = parallel::detectCores() - 1,
                            min_data = 5,
                            feature_fraction = 0.1,
                            bagging_fraction = 0.8,
                            bagging_freq = 1, # bagging_fraction every k iterations
                            # early_stopping_rounds = 295,
                            lambda_l1 = 20.48,
                            lambda_l2 = 2.56,
                            verbose = 1)

x_ <- unique(full[, "transactiondate", with=F])
x_[, trnascationYear := year(transactiondate)]
full <- merge(full, x_, by = "transactiondate")
setkey(full, rowidx)



#####################################################  Re-sample Train to fit Test  #####
############## Folloiwng Code is only to Replace the Months in Test using months in Train
# with same distribution
# TODO!!!!
full_small <- full[(trnascationYear == 2016) &
                    ((split == "train") | (split == "test" & transactionMonth == 10))]
setkey(full_small, rowidx)
full_mm_small <- full_mm[full_small$rowidx + 1, ]
# resample the months from train to put in test
dens.obs <- density(full_small[split=='train', transactionMonth], adjust=0.001)
set.seed(90061)
resample.obs <- round(sample(dens.obs$x, nrow(full_small[split=='test']), replace=TRUE, prob=dens.obs$y))
full_small[split=="test", transactionMonth := resample.obs]
transaciton_cols <- colnames(full)[grepl("transa*", colnames(full))]
trandf <- unique(full[split=="train", transaciton_cols, with = F])
trandf <- trandf[, lapply(.SD, mean, na.rm=T), by="transactionMonth",
                    .SDcols = c("transactionDayofYear", "transactionDayofYear2",
                                "transactionDayofYear3", "transactionMonth2",
                                "transactionMonth3")]
full_small_ <- full_small[split=="test"]
full_small_ <- full_small_[, setdiff(copy(colnames(full_small)), c("transactionDayofYear", "transactionDayofYear2",
                                "transactionDayofYear3", "transactionMonth2",
                                "transactionMonth3")), with = F]
full_small_ <- merge(full_small_, trandf, by = "transactionMonth")
full_small_ <- full_small_[, copy(colnames(full_small)), with = F]
setkey(full_small_, rowidx)
full_small <- rbind(full_small[split=="train"], full_small_)
rm(full_small_); setkey(full_small, rowidx); gc();
cols_to_replace_ <- c("transactionMonth","transactionMonth2","transactionDayofYear",
                                "transactionDayofYear2",
                                "transactionDayofYear3", "transactionMonth2",
                                "transactionMonth3")
full_mm_small[, cols_to_replace_] <- data.matrix(full_small[, cols_to_replace_, with =F])
test_pred_ <- predict(lgb1_reg_noCV, full_mm_small)
full_small[split=="test", logerror_pred_bylgb := test_pred_[full_small[,which(split=="test")]]]
rm(full_mm_small); gc();

############  Train Test Split  ##########
n_folds <- 5
set.seed(10011)
full_small[, random := runif(nrow(full_small))]
full_small[, foldid := ceiling(random / (1/n_folds))]
full_small[, random := NULL]
full_small[, test_split_propensity_by_lgb := 0.99999]
full_mm <- data.matrix(full_small[, c("propens_less05_pred",
                                "propens_less04_pred",
                                "propens_less055_pred",
                                "propens_less03_pred",
                                "propens_less065_pred",
                                "propens_less06_pred",
                                "propens_less02_pred",
                                "propens_less045_pred",
                                "propens_less07_pred",
                                "propens_less003_pred",
                                "propens_less08_pred",
                                "propens_less025_pred",
                                "propens_less015_pred",
                                "propens_less035_pred",
                                "propens_less01_pred",
                                "propens_less075_pred",
                                "propens_less005_pred",
                                "propens_less001_pred",
                                "propens_less085_pred",
                                "propens_less09_pred",
                                "propens_less099_pred",
                                "logerror_pred_bylgb"), with = F])
scale_pos_weight <- nrow(full_small[split=="train"])/nrow(full_small[split=="test"])

for (i in seq(1, n_folds)) {
    trainidx <- which(full_small$foldid != i)
    train_dm_reg <- xgb.DMatrix(
                    data = full_mm[trainidx, ],
                    label = full_small[trainidx, 1*(split=="test")],
                    missing = NA
                )
    valididx <- which(full_small$foldid == i)
    valid_dm_reg <- xgb.DMatrix(
                    data = full_mm[valididx, ],
                    label = full_small[valididx, 1*(split=="test")],
                    missing = NA
                )
    set.seed(10011)
    xgb0_reg <- xgb.train(
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "auc",
                    print_every_n = 50,
                    early_stopping_rounds = 20,
                    colsample_bytree = 0.8,
                    scale_pos_weight = scale_pos_weight,
                    min_child_weight =5,
                    objective = "binary:logistic",
                    subsample = 0.85,
                    nthread = parallel::detectCores() - 1,
                    alpha = 0.1,
                    nrounds = 5000, # the best iteration is at 92
                    eta = 0.1,
                    max.depth = 5,
                    lambda=1
                )
    gc()

    best.iter <- lgb1_reg$best_iter
    # print(paste0("Best GBM iteration is ", best.iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[full_small$split=="train" & full_small$foldid == i, ])
    full_small[split=="train" & foldid == i, logerror_pred_bylgb := valid_pred_]
    test_pred_ <- predict(lgb1_reg, full_mm[full_small$split=="test", ])
    full_small[split=="test", logerror_pred_bylgb := logerror_pred_bylgb + test_pred_]

    rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
}

























full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

for (i in seq(1, n_folds)) {
    trainidx <- which(full$split=="train" & full$foldid != i)
    train_dm_reg <- lgb.Dataset(
                    data = full_mm[trainidx, ],
                    label = full[trainidx, pmin(0.382,pmax(-0.3425, logerror))],
                    weight = rep(1, length(trainidx)) # full[trainidx, rec_weight_bylgb]
                )
    valididx <- which(full$split=="train" & full$foldid == i & full$transactionMonth >= 10)
    valid_dm_reg <- lgb.Dataset(
                    data = full_mm[valididx, ],
                    label = full[valididx, pmin(0.382,pmax(-0.3425, logerror))],
                    weight = rep(1, length(valididx))
                )
    test_dm_reg <- lgb.Dataset(
                    data = full_mm[full$split=="test", ]
                )
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
                nrounds = 663,
                #nfold = 8,
                #device = "cpu", # or use gpu if it's available
                num_threads = parallel::detectCores() - 1,
                min_data = 5,
                feature_fraction = 0.1,
                bagging_fraction = 0.8,
                bagging_freq = 1, # bagging_fraction every k iterations
                # early_stopping_rounds = 295,
                lambda_l1 = 20.48,
                lambda_l2 = 2.56,
                verbose = 1)
    gc()

    best.iter <- lgb1_reg$best_iter
    # print(paste0("Best GBM iteration is ", best.iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[full$split=="train" & full$foldid == i, ])
    full[split=="train" & foldid == i, logerror_pred_bylgb := valid_pred_]
    test_pred_ <- predict(lgb1_reg, full_mm[full$split=="test", ])
    full[split=="test", logerror_pred_bylgb := logerror_pred_bylgb + test_pred_]

    rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
}

full[split=="test", logerror_pred_bylgb := logerror_pred_bylgb / n_folds]

# check mae of logerror with adjusted weight
full[split=="train" & transactionMonth >=10,
     weighted.mean(abs(logerror - logerror_pred_bylgb), rec_weight_bylgb)]





dens.obs <- density(full[split=='test', logerror_pred_bylgb], adjust=0.8)
# Now, resample from the density estimate to get the modeled values. We set prob=dens.obs$y so that the probability of a value in dens.obs$x being chosen is proportional to its modeled density.
set.seed(90061)
resample.obs <- sample(dens.obs$x, nrow(full[split=='train']), replace=TRUE, prob=dens.obs$y)




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
