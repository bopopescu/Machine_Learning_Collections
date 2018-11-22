require(dplyr)
require(data.table)
require(xgboost)
require(readr)
require(glmnet)
require(Matrix)
require(irlba)
require(rsvd)


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
load("zillow_image_v01.RData")  # this is the file expoorted from the program s1.0_model_xgboost_v01.R
rm(full_mm, pred_m_test, pred_m_valid, test_pred_, sampleidx)

full[, rowidx := seq(1, nrow(full))]
setkey(full, rowidx)

load("modelmatrix_svd100.RData")  # load in the matrix from the svd-100

features_to_select <- fread("./s2.0_features_for_sparse_matrix_multiclass.csv")
bucket_quantiles <- c(c(0.005, 0.01, 0.03, 0.97, 0.99, 0.995), seq(0, 1, by = 0.05))
buckets <- c(c(-Inf, Inf), quantile(full$logerror, probs = setdiff(bucket_quantiles, c(0,1)), na.rm=T))
buckets <- sort(unique(buckets))
full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]

n_folds <- 40
full[, random := runif(nrow(full))]
full[, foldid := ceiling(random / (1/n_folds))]

freq_bin_cols <- paste0("freq_bin", seq(1, full[split=="train", length(unique(logerror_bucket))]))
for (col in freq_bin_cols) {
    full[, paste0(col) := 0.0]
}


mlogloss2 <- function(preds, train_dm_freq) {
	n_classes = 26
	mlogloss <- function(pred_matrix, actual_m) {
		pred_matrix <- pmin(pred_matrix, 1 - 10 ^ -15)
		pred_matrix <- pmax(pred_matrix, 10 ^ -15)
		pred_matrix <- pred_matrix / rowSums(pred_matrix)
		pred_matrix <- log(pred_matrix)
		ll <- sum(pred_matrix * actual_m)
		return(-1 * ll / nrow(actual_m))
	}
    labels <- getinfo(train_dm_freq, "label")
    preds2 <- matrix(preds, nrow = length(preds)/n_classes, byrow = TRUE)
    err <- mlogloss(preds2, labels)
    return(list(metric = "mlogloss", value = err))
}

# Add Frequency
for (i in seq(1, n_folds)) {
    train_dm_freq <- xgb.DMatrix(
                    data = modelmatrix2[full$split=="train" & full$foldid != i, ],
                    label = full[full$split=="train" & full$foldid != i, logerror_bucket],
                    missing = NA
                )
    valid_dm_freq <- xgb.DMatrix(
                    data = modelmatrix2[full$split=="train" & full$foldid == i, ],
                    label = full[full$split=="train" & full$foldid == i, logerror_bucket],
                    missing = NA
                )
    test_dm_freq <- xgb.DMatrix(
                data = modelmatrix2[full$split=="test", ],
                missing = NA
            )
	gc()
    xgb1_freq <- xgb.train(
                        data = train_dm_freq,
                        watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
                        # eval_metric = "mlogloss",
						eval_metric = mlogloss2, # in case if this error is not supported
						print_every_n = 20,
						early_stopping_rounds = 20,
						objective = "multi:softprob",
						num_class = length(buckets)-1,
						nthread = parallel::detectCores() -1,
						alpha = 10,
						nrounds = 500,
						lambda = 1,
						subsample = 0.8,
						eta = 0.15,
						max.depth = 3)
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

save.image("s2.1_justdone_freq.RData", compress = F)


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

save.image("s2.1_image_v01.RData", compress = F)
