library(dplyr)
library(data.table)
library(xgboost)
library(readr)
library(glmnet)
library(Matrix)
library(irlba)
library(rsvd)


setwd("E:/zillow_kaggle_2017/")


b_spline_creation2 <- function(vec, col_name,
                              knots_percentiles = seq(0,1, by=0.2),
                              polynomial_degrees = 3) {
    # vec is the vector to be converted to b-spline
    # col_name is the name of the column/variable
    require(splines)
    knots <- quantile(vec, knots_percentiles)  # knots are selected on data records level, not distinct value level
    vec <- unique(vec)
    knots_ <- sort(unique(pmin(pmax(min(vec)+ (1e-10), knots), max(vec) - (1e-10))))
    bspline <- splines::ns(vec, df = polynomial_degrees, knots=knots_)
    res=list()
    res$splinefunction <- bspline
    bspline <- cbind(as.data.frame(vec), as.data.frame(bspline))
    colnames(bspline) <- c(col_name, paste(col_name,"_s", seq_len(ncol(bspline)-1),sep=""))
    if (ncol(bspline) > 1 & FALSE) { #disable the drop of last column feature
        # drop last column since the original value is included here
        bspline <- bspline[,seq(1,ncol(bspline)-1)]
    }
    bspline <- data.table(bspline)
    eval(parse(text=paste("setkey(bspline, ",col_name,")", sep="")))
    print(paste("The data returned is", class(bspline[1])[1]))
    res$bspline <- bspline
    return(res)
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

create_bins <- function(x, n_bins=31) {
    # x is the vector to be binned
    # n_bins is max number of bins to create
    require(stringr)
    quantiles_ <- quantile(x, sort(unique(c(0, 1, seq(0, 1, by=1/n_bins)))))
    quantiles_ <- c(-Inf, sort(unique(quantiles_)), Inf)
    newlabels <- paste0(str_pad(seq(1,length(quantiles_)-1),
                                               nchar(as.character(length(quantiles_)-1)),
                                               pad = "0"))
    x_ <- cut(x=x, breaks = quantiles_, labels = newlabels)
    return(x_)
}


###########################  Load in Data ###########################
# load in the data from the
load("zillow_image_v01.RData")
rm(full_mm, pred_m_test, pred_m_valid, test_pred_, sampleidx)

full[, rowidx := seq(1, nrow(full))]
setkey(full, rowidx)

load("properties_refill_df.RData")
properties_filled_vars <- copy(setdiff(colnames(properties_refill_df), "parcelid"))

for (col in unique(c(properties_filled_vars, paste0(properties_filled_vars, "_log")))) {
    if (col %in% colnames(full)) {
        full[, paste0(col) := NULL]
    }
}

for (col in copy(colnames(full))) {
    if (paste0(col, "_log") %in% colnames(full)) {
        full[, paste0(col, "_log") := NULL]
    }
}
full <- merge(full, properties_refill_df, by="parcelid")

char_vars_df <- fread("./char_vars_in_originaldata.csv")
char_vars <- char_vars_df$Features
for (col in char_vars) {
    if (!col %in% colnames(full)) {next}
    class_ <- full[1, class(get(col))]
    if (class_ %in% c("numeric", "integer")) {
        full[, paste0(col) := as.factor(as.character(get(col)))]
    }
}
predictors3 <- c()
counter <- 1
for (col in predictors_backup01) {
    if (!col %in% colnames(full)) {next}
    print(paste0("Processing variable ", col, ", ", counter, " out of ", length(predictors_backup01)," variable"))
    class_ <- full[1, class(get(col))]
    if (class_ %in% c("character", "factor")) {
        full[is.na(get(col)), paste0(col) := "NAN"]
        full[, paste0(col) := as.factor(as.character(get(col)))]
        predictors3 <- c(predictors3, col)
    }
    if (class_ %in% c("numeric", "integer")) {
        vec_ <- full[, get(col)]
        median_ <- quantile(vec_, probs = 0.5, na.rm = T)
        vec_[is.na(vec_)] <- median_
        full[, paste0(col, "_bin01") := create_bins(vec_, n_bins = 31)]
        full[, paste0(col, "_bin02") := create_bins(vec_, n_bins = 10)]
        full[, paste0(col, "_bin03") := create_bins(vec_, n_bins = 5)]
        predictors3 <- c(predictors3, c(paste0(col, "_bin01"), paste0(col, "_bin03"),paste0(col, "_bin03")))
    }
    counter <- counter + 1
}

if (FALSE) {"non_predictors is the list of varaibls not in modeling"}

# sparse matrix creation
counter <- 1
for (col in predictors3) {
    print(paste0("Processing ", counter, " out of ", length(predictors3)," variables"))
    if (counter == 1) {
        m_ <- sparse.model.matrix(~.-1, full[, col, with=F])
        m_ <- m_[, -which(colSums(m_) == 0)]
        modelmatrix <- m_
    } else {
        try({m_ <- sparse.model.matrix(~.-1, full[, col, with=F]);
            x_ <- colnames(modelmatrix);
            m_ <- m_[, -which(colSums(m_) == 0)];
            modelmatrix <- cbind(modelmatrix, m_);
            colnames(modelmatrix) <- c(x_, colnames(m_));
            rm(m_)})
        gc()
    }
    counter <- counter + 1
}

save.image("s2.0_glmnet_v02_aftersparse.RData", compress = F)







## Sparse Matrix SVD
modelmatrix2 <- irlba(modelmatrix, 100, tol = 1e-5)$u
save(modelmatrix2, file="modelmatrix_svd100.RData", compress= F)
# another version is S$v, the right singlar vector, on column dimension, DO NOT USE!
colnames(modelmatrix2) <- paste0("V", seq(1, ncol(modelmatrix2)))
gc()
modelmatrix2 <- scale(modelmatrix2)
gc()

#########################  Freq Model Fitting  ###################
features_to_select <- fread("./s2.0_features_for_sparse_matrix_multiclass.csv")
bucket_quantiles <- c(c(0.005, 0.01, 0.03, 0.97, 0.99, 0.995), seq(0, 1, by = 0.05))
buckets <- c(c(-Inf, Inf), quantile(full$logerror, probs = setdiff(bucket_quantiles, c(0,1)), na.rm=T))
buckets <- sort(unique(buckets))
full[, logerror_bucket := as.integer(cut(logerror, buckets)) - 1]

train_dm_freq <- xgb.DMatrix(
                    data = modelmatrix2[(full$split=="train") & (full$random < 0.85), ],
                    label = full[(full$split=="train") & (full$random < 0.85), logerror_bucket],
                    missing = NA
                )
gc()
valid_dm_freq <- xgb.DMatrix(
                    data = modelmatrix2[(full$split=="train") & (full$random >= 0.85), ],
                    label = full[(full$split=="train") & (full$random >= 0.85), logerror_bucket],
                    missing = NA
                )
gc()
test_dm_freq <- xgb.DMatrix(
                    data = modelmatrix2[(full$split=="test") & (full$random < 0.85), ],
                    missing = NA
        )
gc()

xgb1_freq <- xgb.cv(
                    data = train_dm_freq,
                    #gblinear = "gblinear",
                    #watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
                    eval_metric = "mlogloss",
                    print_every_n = 20,
                    early_stopping_rounds = 20,
                    objective = "multi:softprob",
                    num_class = length(buckets)-1,
                    nfold = 4,
                    nthread = parallel::detectCores() -1,
                    alpha = 10,
                    nrounds = 500,
                    lambda = 1,
                    subsample = 0.8,
                    eta = 0.15,
                    max.depth = 3
                    )

#########################  Model Fitting  ###################
vars_to_select <- paste0('freq_bin', seq(1, 10))
freqs_ <- model.matrix(~.-1, full[, vars_to_select, with=F])

subset <- intersect(subset, colnames(modelmatrix))

train_dm_reg <- xgb.DMatrix(
                    data = cbind(as.matrix(modelmatrix[(full$split=="train") & (full$random < 0.85), subset]),
                                freqs_[(full$split=='train') & (full$random < 0.85), ]),
                    label = full[(full$split=="train") & (full$random < 0.85), logerror],
                    missing = NA
                )
gc()
valid_dm_reg <- xgb.DMatrix(
                    data = cbind(as.matrix(modelmatrix[(full$split=='train') & (full$random >= 0.85), subset]),
                                freqs_[(full$split=='train') & (full$random >= 0.85), ]),
                    label = full[(full$split=="train") & (full$random >= 0.85), logerror],
                    missing = NA
                )
gc()
test_dm_reg <- xgb.DMatrix(
                    data = cbind(as.matrix(modelmatrix[full$split=="test", subset]), freqs_[full[, which(split=='test')], ]),
                    missing = NA
        )
gc()
xgb2_glmnet <- xgb.train(
                    gblinear = "gblinear",
                    data = train_dm_reg,
                    watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
                    eval_metric = "mae",
                    #nfold = 8,
                    print_every_n = 20,
                    early_stopping_rounds = 20,
                    objective = "reg:linear",
                    nthread = parallel::detectCores() -1,
                    alpha = 130,
                    lambda = 100,
                    nrounds = 500    # use alpha, lamba 30, 30 for version without freq_bin predictors
                )

test_pred_ <- predict(xgb2_glmnet, test_dm_reg)
full[full$split=="test", logerror_pred := test_pred_]

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

write.csv(submission, file = "submission_0625_v1_glmnet.csv", row.names = F)

if (FALSE) {
    # ensembel
    anothersubmission <- fread("./source_data/lgb_starter.csv", header = TRUE)
    weights <- c(0.67, 0.33)
    submission[, 2:7] <- ((weights[1]/sum(weights)) * anothersubmission[, 2:7] + (weights[2]/sum(weights)) * submission[, 2:7])
	submission[, 2:7] <- round(submission[, 2:7], 4)
}
