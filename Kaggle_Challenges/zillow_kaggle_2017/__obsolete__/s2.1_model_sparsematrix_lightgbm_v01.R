
library(lightgbm)
library(dplyr)
library(data.table)
library(xgboost)
library(readr)

setwd("E:/zillow_kaggle_2017/")

distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with ordre
    x = data.table(x)
    length_ = nrow(x)
    x[, ids := seq(1, length_)]
    x2 = x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}


###########################  Load in Data ###########################
# load in the data from the

# load in the data from the
load("s2.1_image_v01.RData")

full[, rowidx := seq(1, nrow(full))]
setkey(full, rowidx)
n_folds <- 20

full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]

full_mm <- data.matrix(full[, predictors2, with = F]) * 1.0  # convert integer to float
full[, logerror_pred := 0.0]

# Add Frequency
for (i in seq(1, n_folds)) {
    train_dm_reg <- lgb.Dataset(
                    data = full_mm[full$split=="train" & full$foldid != i, ],
                    label = full[full$split=="train" & full$foldid != i, logerror],
                    missing = NA
                )
    valid_dm_reg <- lgb.Dataset(
                    data = full_mm[full$split=="train" & full$foldid == i, ],
                    label = full[full$split=="train" & full$foldid == i, logerror],
                    missing = NA
                )
    test_dm_reg <- lgb.Dataset(
                data = full_mm[full$split=="test", ],
                missing = NA
            )
	param <- list(objective = "regression",
				  #metric="poisson", #metric="l2,l1"
				  metric="l1",
				  num_leaves = 7, # 2**8 -1, interaction depth=5
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
						  min_data = 5,
						  feature_fraction = 1,
						  bagging_fraction = 0.85,
						  bagging_freq = 1, # bagging_fraction every k iterations
						  early_stopping_rounds = 50,
						  lambda_l1 = 70,
						  lambda_l2 = 70,
						  verbose = 2)
    gc()

	best.iter <- lgb1_reg$best_iter
    # print(paste0("Best GBM iteration is ", best.iter))
    valid_pred_ <- predict(lgb1_reg, full_mm[full$split=="train" & full$foldid == i, ])
    test_pred_ <- predict(lgb1_reg, full_mm[full$split=="test", ])

    full[full$split=="train" & full$foldid == i, logerror_pred := valid_pred_]
    full[full$split=="test", logerror_pred := logerror_pred + test_pred_]

    rm(train_dm_reg, valid_dm_reg, test_dm_reg); gc();
}

full[full$split=="test", logerror_pred := logerror_pred / n_folds]


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

write.csv(submission, file = "submission_0624_v2_lightgbm.csv", row.names = F)

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

