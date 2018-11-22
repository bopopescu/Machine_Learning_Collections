setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(readr)


###########################  Functions  ###########################
char2num_dt <- function(data_table, column) {
    column_num <- paste0(column, "_num")
    if (!class(data_table[, get(column)]) %in% c("character", "factor", "integer64")) {
        print(paste0("No new numerical variable is added for column ", column))
        return(data_table)
    } else {
        data_table[, zzz_rowid := seq(1, nrow(data_table))]
        if (class(data_table[, get(column)]) == "integer64") {
            data_table[, zzz_column := as.character(get(column))]
        } else {
            data_table[, zzz_column := get(column)]
        }
        dt_ <- unique(data_table[, "zzz_column", with = F])
        dt_ <- dt_[complete.cases(dt_)]
        dt_[, paste0(column_num) := as.numeric(as.character(zzz_column))]
        data_table <- merge(data_table, dt_, by="zzz_column", all.x=T, all.y=F)
        setkey(data_table, zzz_rowid)
        data_table[, zzz_rowid := NULL]
        data_table[, zzz_column := NULL]
        rm(dt_); gc();
        return(data_table)
    }
}

substring_dt <- function(data_table, column, range=c(1,100)) {
    # range is the start and end points for the substring
    if (class(data_table[, get(column)]) != "character") {
        data_table[, zzz_column := as.character(get(column))]
    } else {
        data_table[, zzz_column := get(column)]
    }
    data_table[, zzz_rowid := seq(1, nrow(data_table))]
    dt_ <- unique(data_table[, "zzz_column", with = F])
    dt_ <- dt_[complete.cases(dt_)]
    dt_[, paste0(column, "_", range[1], "to", range[2]) := substring(zzz_column, range[1],pmin(nchar(zzz_column),range[2]))]
    data_table <- merge(data_table, dt_, by="zzz_column", all.x=T, all.y=F)
    print(paste0("New column ", paste0(column, "_", range[1], "to", range[2]), " is created..."))
    setkey(data_table, zzz_rowid)
    data_table[, zzz_rowid := NULL]
    data_table[, zzz_column := NULL]
    rm(dt_); gc();
    return(data_table)
}


mae <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    weight <- rep(1, length(labels))
    try({weight <- getinfo(dtrain, "weight")}, silent=T)
    err <- sum(weight * abs(labels-preds))/sum(weight)
    return(list(metric = "mae",  value=err))
}
# in case if the xgboost version is old, use this
# as the eval_metric, just use feval = mae in xgb


distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with ordre
    x <- data.table(x)
    length_ <- nrow(x)
    x[, ids := seq(1, length_)]
    x2 <- x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}


label_encode_factor <- function(dt, col){
    # dt is the data table
    # col is the column name of the varaible that needs to apply label encoder
    class_ <- class(dt[1, get(col)])
    if (class_ %in% c("character", "factor")) {
        levels_ <- sort(unique(as.character(dt[, get(col)])))
        dt[, paste0(col) := factor(get(col), levels = levels_)]
    }
    return(dt)
}


label_encode_factor2 <- function(dt) {
    # dt is the data table
    for (col in colnames(dt)) {
        dt <- label_encode_factor(dt, col)
    }
    return(dt)
}


# Common Setting
non_predictors <- c("parcelid", "transactiondate", "logerror", "split", "random", "foldid",
                     "logerror_bucket", "logerror_pred", "rowidx")
### Load in the files
full <- fread("./full.csv")

predictors <- fread("./Var_Importance.csv")$Feature
month_vars <- c("transactionMonth", "transactionMonth2", "transactionMonth3",
                "transactionDayofYear", "transactionDayofYear2",
                "transactionDayofYear3")

full <- label_encode_factor2(full)


















































################################################################################
#   LEGACY CODE
################################################################################
if (FALSE) {

	###############  Train/Test Split Propensity Model  ###############
	predictors_ <- intersect(setdiff(predictors, c(month_vars, non_predictors)),
								colnames(full))

	full_small <- unique(full[, unique(c(predictors_, "parcelid", "split")), with=F])
	full_mm <- data.matrix(full_small[, predictors_, with = F]) * 1.0  # convert integer to float
	n_folds <- 20
	set.seed(10011)
	full_small[, random := runif(nrow(full_small))]
	full_small[, foldid := ceiling(random / (1/n_folds))]
	full_small[, random := NULL]

	if (FALSE) {
		train_dm_reg <- xgb.DMatrix(
						data = full_mm[full_small$foldid != 9999, ],
						label = full_small[full_small$foldid != 9999, 1*(split == "test")],
						missing = NA
					)
		for (alpha_ in 0.1*(2 ** seq(0,10))) {
			for (lambda_ in 0.1*(2 ** seq(0,10))) {
				x_ <- xgb.cv(data = train_dm_reg,
						eval_metric = "auc",
						print_every_n = 100,
						early_stopping_rounds = 20,
						nfold = 8,
						colsample_bytree = 0.8,
						min_child_weight = 5,
						objective = "binary:logistic",
						subsample = 0.85,
						nthread = parallel::detectCores() - 1,
						alpha = alpha_,
						nrounds = 500, # the best iteration is at 92
						eta = 0.1,
						max.depth = 5,
						lambda=lambda_
						)
				gc();
				print(paste0(rep("#", 40), collapse=""))
				mintestmean_ <- x_[get("test.auc.mean") == max(get("test.auc.mean"))][1]$test.auc.mean
				print(paste0("AUC for alpha ", alpha_, " and lambda ", lambda_, " is ", mintestmean_))
				print(paste0(rep("#", 40), collapse=""))
				# picked alpha = 1, lambda = 1
			}
		}
	}


	# Add Propensity to Train/Test split
	full_small[, test_split_propensity := 0.99999]
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
		xgb0_reg <- xgb.train(
						data = train_dm_reg,
						watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
						eval_metric = "auc",
						print_every_n = 50,
						early_stopping_rounds = 20,
						colsample_bytree = 0.8,
						min_child_weight =5,
						objective = "binary:logistic",
						subsample = 0.85,
						nthread = parallel::detectCores() - 1,
						alpha = 1,
						nrounds = 500, # the best iteration is at 92
						eta = 0.1,
						max.depth = 5,
						lambda=1
					)
		gc()
		valid_pred_ <- predict(xgb0_reg, valid_dm_reg)
		full_small[foldid == i, test_split_propensity := valid_pred_]
		rm(train_dm_reg, valid_dm_reg, valid_pred_); gc();
	}

	full_small <- full_small[, c("parcelid", "split", "test_split_propensity"), with = F]
	full <- merge(full, full_small, by = c("parcelid", "split"), all.x=T, all.y=F)
	setkey(full, rowidx)

	new_predictors <- "test_split_propensity"


	###############  logerror Bucket Model Fitting  ###############
	# Train XGBoost model using multi-class classification mode
	bucket_quantiles <- c(0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99)
	bucket_quantiles <- c(bucket_quantiles, seq(0, 1, by = 0.1))
	bucket_quantiles <- sort(unique(bucket_quantiles))
	buckets <- c(c(-Inf, Inf), quantile(full$logerror, probs = setdiff(bucket_quantiles, c(0,1)), na.rm=T))
	buckets <- sort(unique(buckets))
	full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]

	setkey(full, rowidx)

	n_folds <- 20
	set.seed(10011)
	full[, random := runif(nrow(full))]
	full[, foldid := ceiling(random / (1/n_folds))]
	full[, random := NULL]

	freq_bin_cols <- paste0("freq_bin", seq(1, full[split=="train", length(unique(logerror_bucket))]))
	for (col in freq_bin_cols) {
		full[, paste0(col) := 0.0]
	}
	# Add Bin Prediction
	predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
	full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

	for (i in seq(1, n_folds)) {
		train_dm_freq <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid != i, ],
						label = full[split=="train" & foldid != i, logerror_bucket],
						missing = NA
					)
		valid_dm_freq <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid == i, ],
						label = full[split=="train" & foldid == i, logerror_bucket],
						missing = NA
					)
		test_dm_freq <- xgb.DMatrix(
					data = full_mm[full$split=="test" & full$foldid == i, ],
					missing = NA
				)
		xgb1_freq <- xgb.train(
							data = train_dm_freq,
							watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
							eval_metric = "mlogloss",
							print_every_n = 20,
							early_stopping_rounds = 20,
							objective = "multi:softprob",
							num_class = length(buckets)-1,
							subsample = 0.632,
							nthread = parallel::detectCores() -1,
							alpha = 6.4,
							nrounds = 500,
							eta = 0.1,
							max.depth = 3,
							lambda = 0
						)
		gc()
		pred_m_valid <- matrix(predict(xgb1_freq, valid_dm_freq), nrow = nrow(valid_dm_freq), byrow = TRUE)
		pred_m_valid <- data.table(pred_m_valid)
		setnames(pred_m_valid, colnames(pred_m_valid), freq_bin_cols)

		pred_m_test <- matrix(predict(xgb1_freq, test_dm_freq), nrow = nrow(test_dm_freq), byrow = TRUE)
		pred_m_test <- data.table(pred_m_test)
		setnames(pred_m_test, colnames(pred_m_test), freq_bin_cols)

		for (col in freq_bin_cols) {
			full[split=="train" & foldid == i, paste0(col) := pred_m_valid[, get(col)]]
			full[split=="test" & foldid == i, paste0(col) := pred_m_test[, get(col)]]
		}

		rm(train_dm_freq, valid_dm_freq, test_dm_freq); gc();
	}

	new_predictors <- sort(unique(c(new_predictors, freq_bin_cols)))


	###############  logerror Bucket Model Fitting ver2 ###############
	# Train XGBoost model using multi-class classification mode
	bucket_quantiles <- c(0.01, 0.03, 0.1, 0.9, 0.97, 0.99)
	bucket_quantiles <- sort(unique(bucket_quantiles))
	buckets <- c(c(-Inf, Inf), quantile(full$logerror, probs = setdiff(bucket_quantiles, c(0,1)), na.rm=T))
	buckets <- sort(unique(buckets))
	full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]

	setkey(full, rowidx)

	n_folds <- 20
	set.seed(10011)
	full[, random := runif(nrow(full))]
	full[, foldid := ceiling(random / (1/n_folds))]
	full[, random := NULL]

	freq_bin_cols <- paste0("freq_v2_bin", seq(1, full[split=="train", length(unique(logerror_bucket))]))
	for (col in freq_bin_cols) {
		full[, paste0(col) := 0.0]
	}
	# Add Bin Prediction
	predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
	full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

	for (i in seq(1, n_folds)) {
		train_dm_freq <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid != i, ],
						label = full[split=="train" & foldid != i, logerror_bucket],
						missing = NA
					)
		valid_dm_freq <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid == i, ],
						label = full[split=="train" & foldid == i, logerror_bucket],
						missing = NA
					)
		test_dm_freq <- xgb.DMatrix(
					data = full_mm[full$split=="test" & full$foldid == i, ],
					missing = NA
				)
		xgb1_freq <- xgb.train(
							data = train_dm_freq,
							watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
							eval_metric = "mlogloss",
							print_every_n = 20,
							early_stopping_rounds = 10,
							objective = "multi:softprob",
							num_class = length(buckets)-1,
							subsample = 0.632,
							nthread = parallel::detectCores() -1,
							alpha = 3.2,
							nrounds = 500,
							eta = 0.1,
							max.depth = 3,
							lambda = 0
						)
		gc()
		pred_m_valid <- matrix(predict(xgb1_freq, valid_dm_freq), nrow = nrow(valid_dm_freq), byrow = TRUE)
		pred_m_valid <- data.table(pred_m_valid)
		setnames(pred_m_valid, colnames(pred_m_valid), freq_bin_cols)

		pred_m_test <- matrix(predict(xgb1_freq, test_dm_freq), nrow = nrow(test_dm_freq), byrow = TRUE)
		pred_m_test <- data.table(pred_m_test)
		setnames(pred_m_test, colnames(pred_m_test), freq_bin_cols)

		for (col in freq_bin_cols) {
			full[split=="train" & foldid == i, paste0(col) := pred_m_valid[, get(col)]]
			full[split=="test" & foldid == i, paste0(col) := pred_m_test[, get(col)]]
		}

		rm(train_dm_freq, valid_dm_freq, test_dm_freq); gc();
	}

	new_predictors <- sort(unique(c(new_predictors, freq_bin_cols)))


	###############  Regression Model Fitting  ###############
	setkey(full, rowidx)

	n_folds <- 20
	set.seed(10011)
	full[, random := runif(nrow(full))]
	full[, foldid := ceiling(random / (1/n_folds))]
	full[, random := NULL]

	full[, logerror_pred := 0.0]
	# Add Bin Prediction
	predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
	full_mm <- data.matrix(full[, predictors_, with = F]) * 1.0  # convert integer to float

	for (i in seq(1, n_folds)) {
		train_dm_reg <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid != i, ],
						label = full[split=="train" & foldid != i, logerror],
						missing = NA
					)
		valid_dm_reg <- xgb.DMatrix(
						data = full_mm[full$split=="train" & full$foldid == i, ],
						label = full[split=="train" & foldid == i, logerror],
						missing = NA
					)
		test_dm_reg <- xgb.DMatrix(
					data = full_mm[full$split=="test", ],
					missing = NA
				)

		set.seed(10011)
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
		# 0.067910 on test mae, alpha, lambda = 100, 100, nrounds 965
		valid_pred_ <- predict(xgb1_reg, valid_dm_reg)
		test_pred_ <- predict(xgb1_reg, test_dm_reg)

		full[split=="train" & foldid == i, logerror_pred := valid_pred_]
		full[split=="test", logerror_pred := logerror_pred + test_pred_]

		rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
	}

	full[split=="test", logerror_pred := logerror_pred/n_folds]
	setnames(full, "logerror_pred", "logerror_pred_byxgboost")

}
