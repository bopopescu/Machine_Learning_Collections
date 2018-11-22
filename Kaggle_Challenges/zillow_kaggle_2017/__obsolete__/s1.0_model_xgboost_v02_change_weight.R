require(dplyr)
require(data.table)
require(xgboost)
require(readr)


###########################  Load in Data ###########################
load("zillow_image_v01.RData")


mae <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    weight <- rep(1, length(labels))
    try({weight <- getinfo(dtrain, "weight")})
    err <- sum(weight * abs(labels-preds))/sum(weight)
    return(list(metric = "mae",  value=err))
}

# in case if the xgboost version is old, use this
# as the eval_metric, just use feval = mae in xgb

distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with ordre
    x = data.table(x)
    length_ = nrow(x)
    x[, ids := seq(1, length_)]
    x2 = x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}


######################  Train Model -- Full Model ###################
full_small3 <- full_small2[, lapply(.SD, mean, na.rm=T), .SDcols = "record_weight", by="parcelid"]

n_folds <- 200
full[, rowidx := seq(1, nrow(full))]
full[, random := runif(nrow(full))]
full <- merge(full, full_small3, by = "parcelid", all.x=T, all.y=F)
setkey(full, rowidx)
full[, foldid := ceiling(random / (1/n_folds))]
full[, random := NULL]
full_mm <- data.matrix(full[, c(predictors2, "record_weight"), with = F]) * 1.0  # convert integer to float
full[, logerror_pred := 0.0]

for (i in seq(1, n_folds)) {
    trainidx <- full$split=="train" & full$foldid != i
    train_dm_reg <- xgb.DMatrix(
                    data = full_mm[which(trainidx), ],
                    label = full[which(trainidx), logerror],
                    weight = full[which(trainidx), record_weight],
                    missing = NA
                )
    valididx <- full$split=="train" & full$foldid == i
    valid_dm_reg <- xgb.DMatrix(
                    data = full_mm[which(valididx), ],
                    label = full[which(valididx), logerror],
                    weight = full[which(valididx), record_weight],
                    missing = NA
                )
    testidx <- full$split=="test"
    test_dm_reg <- xgb.DMatrix(
                data = full_mm[which(testidx), ],
                missing = NA
            )

    set.seed(10011)
    xgb1_reg <- xgb.train(
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "mae",
                    #feval = mae,
                    print_every_n = 50,
                    early_stopping_rounds = 35, # early stopping may cause overfitting
                    maximize = FALSE,
                    colsample_bytree = 0.16,
                    min_child_weight =5,
                    objective = "reg:linear",
                    subsample = 0.8,
                    nthread = parallel::detectCores() - 1,
                    alpha = 100,
                    nrounds = 1000, # the best iteration is from xgb.cv calculation through same nfold
                    eta = 0.05,
                    max.depth = 3,
                    lambda = 100
                )
    gc()
    # 0.067910 on test mae, alpha, lambda = 100, 100, nrounds 965
    valid_pred_ <- predict(xgb1_reg, valid_dm_reg)
    test_pred_ <- predict(xgb1_reg, test_dm_reg)

    full[which(valididx), logerror_pred := valid_pred_]
    full[which(testidx), logerror_pred := logerror_pred + test_pred_]
}

rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
full[which(testidx), logerror_pred := logerror_pred / n_folds]

xgb_pred_s1.0_v02 <- full[, c("logerror", "logerror_pred", "split", "transactionMonth", "rowidx"), with = F]
save(xgb_pred_s1.0_v02, file= "xgb_pred_s1.0_v02.RData")

# v01 version, first version, has mae 0.06512598
# full[full$split=="train" & full$transactionMonth >= 10, mean(abs(logerror - logerror_pred))]

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

write.csv(submission, file = "submission_0701_s1.0_xgb_v02.csv", row.names = F)

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
