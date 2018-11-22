library(dplyr)
library(data.table)
library(xgboost)
library(readr)

setwd("E:/zillow_kaggle_2017/")

###########################  Load in Data ###########################
property_df <- fread("./source_data/properties_2016.csv",
                        na.strings=c("NA","", "NAN", "N/A", "<NA>"))
train_df <- fread("./source_data/train_2016_v2.csv")
test_df <- fread("./source_data/sample_submission.csv", header = TRUE)

setkey(property_df, parcelid)
setkey(train_df, parcelid, transactiondate)

# Load Test Data
print("Prepare for the prediction data file ...")
setnames(test_df, "ParcelId", "parcelid")
test_df_raw <- copy(test_df)
transac_date_columns <- c("201610", "201611", "201612", "201710", "201711", "201712")
setnames(test_df, transac_date_columns, paste0(transac_date_columns, "ZZ"))
counter <- 1
for (col in paste0(transac_date_columns, "ZZ")) {
    if (counter == 1) {
        test_df <- test_df_raw[, "parcelid", with = F]
        test_df[, transactiondate := paste0(substring(col, 1, 4), "-",substring(col, 5, 6),"-15")]
    } else {
        test_df_ <- test_df_raw[, "parcelid", with = F]
        test_df_[, transactiondate := paste0(substring(col, 1, 4), "-",substring(col, 5, 6),"-15")]
        test_df <- rbind(test_df, test_df_)
        rm(test_df_); gc()
    }
    counter <- counter + 1
}
setkey(test_df, parcelid, transactiondate)
test_df$logerror <- NA
test_df$split <- "test"
train_df$split <- "train"
test_df <- test_df[, colnames(train_df), with =F]

# Merge Train and Test Data
full <- merge(rbind(train_df, test_df), property_df, by="parcelid", all.x=T, all.y=F)
# since submission does not have date, we use dummy day 15 for all days in each month
full[, transactiondate := paste0(substring(transactiondate, 1, 7), "-15")]
nrow_ <- copy(nrow(full))
full[, rowidx := seq(1, nrow_)]

# remove some useless data, keep test_df_raw to use it for scoring submission
rm(train_df, test_df, property_df)


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


distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with ordre
    x = data.table(x)
    length_ = nrow(x)
    x[, ids := seq(1, length_)]
    x2 = x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}


###########################  Encode the Categorical Variables  ###########################
non_predictors <- c("parcelid", "transactiondate", "logerror", "split", "random", "foldid",
                     "logerror_bucket", "logerror_pred", "rowidx")

vars_not_be_logged <-  c(non_predictors,
            "airconditioningtypeid", "architecturalstyletypeid",
            "buildingclasstypeid",
            "decktypeid", "regionidzip",
            "storytypeid",
            "typeconstructiontypeid",
            "fips", "heatingorsystemtypeid",
            "pooltypeid10",
            "pooltypeid2", "pooltypeid7", "propertylandusetypeid",
            "rawcensustractandblock", "regionidcity", "regionidneighborhood",
            "transactionYear", "transactionMonth", "transactionDayofYear",
            "transactionDayofYear2",
            "rawcensustractandblock_num", "censustractandblock_num")

# Process some character variables
full <- char2num_dt(full, "rawcensustractandblock")
full[, censustractandblock := as.character(censustractandblock)]
full <- char2num_dt(full, "censustractandblock")
full <- substring_dt(full, "censustractandblock", range=c(1,4))
full <- substring_dt(full, "censustractandblock", range=c(1,8))
full <- substring_dt(full, "censustractandblock", range=c(1,12))
full <- substring_dt(full, "propertyzoningdesc", range=c(1,3))
full <- substring_dt(full, "propertyzoningdesc", range=c(1,4))
full <- substring_dt(full, "propertycountylandusecode", range=c(1,2))

# Add Date info from transaction date
full[, transactionYear := year(as.Date(as.character(transactiondate), "%Y-%m-%d"))]
full[, transactionMonth := month(as.Date(as.character(transactiondate), "%Y-%m-%d"))]
full[, transactionDayofYear := yday(as.Date(as.character(transactiondate), "%Y-%m-%d"))]
full[, transactionDayofYear2 := transactionYear + transactionDayofYear/366]

# Label encode the character variabless
char_vars_to_loop_ <- copy(setdiff(colnames(full), non_predictors))
for (col in char_vars_to_loop_) {
    class_ <- full[, class(get(col))]
    if (class_ %in% c("character", "factor", "integer64")) {
        values_ <- full[, sort(unique(as.character(get(col))))]
        full[, paste0(col) := factor(get(col), levels = values_)]
    }
}

# Log the numerical variables
num_vars_to_loop_ <- copy(setdiff(colnames(full), vars_not_be_logged))
for (col in num_vars_to_loop_) {
    class_ <- full[, class(get(col))]
    if (class_ %in% c("numeric", "integer")) {
        print(paste0("Taking log of variable ", col))
        min_ <- min(full[, get(col)], na.rm=T)
        full[, paste0(col,"_log") := log(get(col) - min_ + 1)]
    }
}

###########################  Create Buckets and Add Propensity to it  ###########################
predictors <- setdiff(colnames(full), non_predictors)
predictors <- copy(predictors)
gc()

# Train XGBoost model using multi-class classification mode
bucket_quantiles <- c(0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99)
buckets <- c(c(-Inf, Inf), quantile(full$logerror, probs = setdiff(bucket_quantiles, c(0,1)), na.rm=T))
buckets <- sort(unique(buckets))
full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]

sampleidx <- sample(seq(1, nrow(full)), size = as.integer(0.01 * nrow(full)))  # this is only for testing

# Check if a variable only has a unique value
for (col in colnames(full)) {
    if (length(unique(full[, get(col)])) == 1) {
        print(paste0("Column ", col, " only has ONE value"))
    }
}


###########################  Train Model -- Frequency ###########################
predictors_backup01 <- predictors

predictors <- c("finishedsquarefeet12",
                    "taxamount",
                    "longitude",
                    "calculatedfinishedsquarefeet",
                    "structuretaxvaluedollarcnt",
                    "yearbuilt",
                    "taxvaluedollarcnt",
                    "regionidzip",
                    "landtaxvaluedollarcnt",
                    "transactionMonth",
                    "lotsizesquarefeet",
                    "latitude",
                    "taxdelinquencyyear",
                    "censustractandblock",
                    "poolcnt",
                    "regionidneighborhood",
                    "finishedsquarefeet15",
                    "propertyzoningdesc",
                    "propertycountylandusecode",
                    "regionidcity",
                    "basementsqft",
                    "propertylandusetypeid",
                    "bedroomcnt",
                    "garagetotalsqft",
                    "finishedfloor1squarefeet",
                    "calculatedbathnbr",
                    "rawcensustractandblock",
                    "bathroomcnt",
                    "threequarterbathnbr")

setkey(full, rowidx)
full_mm <- data.matrix(full[, predictors, with = F]) * 1.0  # convert integer to float

n_folds <- 20
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]

freq_bin_cols <- paste0("freq_bin", seq(1, full[split=="train", length(unique(logerror_bucket))]))
for (col in freq_bin_cols) {
    full[, paste0(col) := 0.0]
}
# Add Frequency
for (i in seq(1, n_folds)) {
    train_dm_freq <- xgb.DMatrix(
                    data = full_mm[full$split=="train" & full$foldid != i, ],
                    label = full[full$split=="train" & full$foldid != i, logerror_bucket],
                    missing = NA
                )
    valid_dm_freq <- xgb.DMatrix(
                    data = full_mm[full$split=="train" & full$foldid == i, ],
                    label = full[full$split=="train" & full$foldid == i, logerror_bucket],
                    missing = NA
                )
    test_dm_freq <- xgb.DMatrix(
                data = full_mm[full$split=="test", ],
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
                        alpha = 1,
                        nrounds = 500,
                        eta = 0.1,
                        max.depth = 3,
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
        full[full$split=="train" & full$foldid == i, paste0(col) := pred_m_valid[, get(col)]]
        full[full$split=="test", paste0(col) := get(col) + pred_m_test[, get(col)]]
    }

    rm(train_dm_freq, valid_dm_freq, test_dm_freq); gc();
}

for (col in freq_bin_cols) {
    full[full$split=="test", paste0(col) := get(col) / n_folds]
}

predictors2 <- unique(c(predictors, freq_bin_cols))



###########################  Train Model -- Full Model ###########################

n_folds <- 40

full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]

setkey(full, rowidx)

full_mm <- data.matrix(full[, predictors2, with = F]) * 1.0  # convert integer to float
full[, logerror_pred := 0.0]

# Add Frequency
for (i in seq(1, n_folds)) {
    train_dm_reg <- xgb.DMatrix(
                    data = full_mm[full$split=="train" & full$foldid != i, ],
                    label = full[full$split=="train" & full$foldid != i, logerror],
                    missing = NA
                )
    valid_dm_reg <- xgb.DMatrix(
                    data = full_mm[full$split=="train" & full$foldid == i, ],
                    label = full[full$split=="train" & full$foldid == i, logerror],
                    missing = NA
                )
    test_dm_reg <- xgb.DMatrix(
                data = full_mm[full$split=="test", ],
                missing = NA
            )

    xgb1_reg <- xgb.train(
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "mae",
                    print_every_n = 20,
                    early_stopping_rounds = 20,
                    colsample_bytree = 0.16,
                    min_child_weight =5,
                    objective = "reg:linear",
                    subsample = 0.85,
                    nthread = parallel::detectCores() - 1,
                    alpha = 100,
                    nrounds = 500, # the best iteration is at 92
                    eta = 0.1,
                    max.depth = 3,
                    lambda=100
                )
    gc()

    valid_pred_ <- predict(xgb1_reg, valid_dm_reg)
    test_pred_ <- predict(xgb1_reg, test_dm_reg)

    full[full$split=="train" & full$foldid == i, logerror_pred := valid_pred_]
    full[full$split=="test", logerror_pred := logerror_pred + test_pred_]

    rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
}

full[full$split=="test", logerror_pred := logerror_pred / n_folds]

save.image("zillow_image_v01.RData", compress = F)


########################################################################
# Create Submission
########################################################################
setkey(full, rowidx)

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

write.csv(submission, file = "submission_0623_v1.csv", row.names = F)

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





##################  LEGACY CODE  ##################
if (FALSE) {
    train_dm_freq <- xgb.DMatrix(
                    data = full_mm[full[, split=="train"], ],
                    label = full[split=="train", logerror_bucket],
                    missing = NA
                )

    gc()

    xgb1_freq <- xgb.cv(
        data = train_dm_freq,
        #watchlist = list(valid = valid_dm_freq, train = train_dm_freq),
        #eval_metric = "mae",
        metrics = "mlogloss",
        print_every_n = 20,
        early_stopping_rounds = 20,
        objective = "multi:softprob",
        num_class = length(buckets)-1,
        subsample = 0.8,
        nfold = 8,
        nthread = parallel::detectCores() -1,
        alpha = 1,
        nrounds = 10000,
        eta = 0.1,
        max.depth = 3,
        lambda = 1
    )

    xgb1_reg <- xgb.cv(
                    data = train_dm_reg,
                    #watchlist = list(valid = valid_dm_freq, train = train_dm_freq),
                    #eval_metric = "mae",
                    metrics = "mae",
                    print_every_n = 20,
                    early_stopping_rounds = 20,
                    objective = "reg:linear",
                    subsample = 0.8,
                    nfold = 10,
                    nthread = parallel::detectCores() - 1,
                    alpha = 30,
                    nrounds = 150, # the best iteration is at 92
                    eta = 0.1,
                    max.depth = 3,
                    lambda=30
                )

    ### Parameter Tuning
    summary <- data.table(alpha=0, lambda=0, bestmae=0)
    for (alpha_ in c(600, 300, 200, 100, 60, 30, 10, 6, 3, 1)) {
        for (lambda_ in c(100, 60, 30, 10, 6, 3, 1)) {
            xgbmodel = xgb.cv(
                data = train_dm_reg,
                #watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                eval_metric = "mae",
                print_every_n = 20,
                early_stopping_rounds = 20,
                objective = "reg:linear",
                subsample = 0.632,
                nthread = parallel::detectCores() - 1,
                alpha = alpha_,
                nfold=8,
                nrounds = 500, # the best iteration is at 92
                eta = 0.1,
                max.depth = 3,
                lambda=lambda_)

            bestmae <- xgbmodel$evaluation_log[xgbmodel$best_iteration]$test_mae_mean
            summary <- rbind(summary,
                             data.table(alpha=alpha_, lambda=lambda_, bestmae=bestmae))
        }
    }
    gc()
    # so far, the best results are for alpha lambda 100, 30, 0.06785075
}

