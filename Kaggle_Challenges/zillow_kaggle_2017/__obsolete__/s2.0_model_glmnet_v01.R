library(dplyr)
library(data.table)
library(xgboost)
library(readr)
library(glmnet)
library(Matrix)

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

###########################  Load in Data ###########################
# load in the data from the
load("zillow_image_v01.RData")
rm(full_mm, pred_m_test, pred_m_valid, test_pred_, sampleidx)

full[, rowidx := seq(1, nrow(full))]
setkey(full, rowidx)


# list of numerical predictors in the properties that need some missing data imputation
num_predictors_to_impute <- c("finishedsquarefeet12", "taxamount", "structuretaxvaluedollarcnt",
                 "calculatedfinishedsquarefeet", "landtaxvaluedollarcnt",
                 "taxvaluedollarcnt", "finishedfloor1squarefeet", "finishedsquarefeet15",
                 "lotsizesquarefeet", "taxdelinquencyyear", "garagetotalsqft",
                 "yearbuilt", "basementsqft", "calculatedbathnbr")

# filter out the variables with too many missing values
too_many_missing <- c()
# Check missing percentage
for (col in num_predictors_to_impute) {
    missing_ <- full[split == "train", sum(is.na(get(col)))] / full[, sum(split == "train")]
	if (missing_ > 0.5) {
	    print(paste0("Variable --  ", col, "  -- has ", round(100*missing_,2),"% missing values..."))
		too_many_missing <- unique(c(too_many_missing, col))
	} else {
	    print(paste0(col, ": ", round(100*missing_,2),"% missing ..."))
	}
}

num_predictors_to_impute <- setdiff(num_predictors_to_impute, too_many_missing)


###########################  Numerical Variable Missing Value Imputation  ###########################
## Back-fill the missing values
backfill_pred_vars <- setdiff(predictors_backup01, predictors_backup01[grepl(pattern="*_log", predictors_backup01)])

for (col in num_predictors_to_impute) {
    missing_ <- full[, sum(is.na(get(col)))]
    if (missing_ == 0) {
	    next
	} else {
        vars_sel <- setdiff(backfill_pred_vars,
                                c("transactionYear",
                                    "transactionMonth",
                                    "transactionDayofYear",
                                    "transactionDayofYear2"))
        if (!"properties_refill_df" %in% ls()) {
            properties_refill_df <- unique(full[, unique(c(vars_sel, "parcelid")), with = F])
        }
		properties_refill_df[, random := runif(nrow(properties_refill_df))]
        properties_refill_df <- properties_refill_df[parcelid %in% full[, unique(parcelid)]]
		full_m_ <- data.matrix(properties_refill_df[, setdiff(vars_sel, col), with = F]) * 1.0
		gc()
		train_cut <- 0.85
		train_dm_reg <- xgb.DMatrix(
						data = full_m_[!is.na(properties_refill_df[, get(col)]) & properties_refill_df$random < train_cut, ],
						label = properties_refill_df[!is.na(get(col)) & random < train_cut, get(col)],
						missing = NA
					)
		valid_dm_reg <- xgb.DMatrix(
						data = full_m_[!is.na(properties_refill_df[, get(col)]) & properties_refill_df$random >= train_cut, ],
						label = properties_refill_df[!is.na(get(col)) & random >= train_cut, get(col)],
						missing = NA
					)
		test_dm_reg <- xgb.DMatrix(
						data = full_m_[is.na(properties_refill_df[, get(col)]), ],
						missing = NA
					)
		xgb1_reg <- xgb.train(
						data = train_dm_reg,
						watchlist = list(train = train_dm_reg, valid = valid_dm_reg),
						eval_metric = "rmse",
						print_every_n = 20,
						#nfold = 4,
						early_stopping_rounds = 20,
						colsample_bytree = 0.8,
						min_child_weight = 5,
						objective = "reg:linear",
						subsample = 0.5,
						nthread = parallel::detectCores() - 1,
						alpha = 1,
						nrounds = 5000, # the best iteration is at 92
						eta = 0.15,
						max.depth = 4,
						lambda=1
					)
		gc()
        test_pred_ <- predict(xgb1_reg, test_dm_reg)
        boundaries <- properties_refill_df[!is.na(get(col)), quantile(get(col), probs = c(0.01, 0.99))]
        test_pred_ <- pmax(boundaries[1], pmin(boundaries[2], test_pred_))
        if (class(properties_refill_df[, get(col)]) == "integer") {
            test_pred_ <- as.integer(round(test_pred_, 0))
        }
        properties_refill_df[is.na(get(col)), paste0(col) := test_pred_]
        rm(full_m_) ; gc();
	}
}

all_vars_ <- c()
for (col in colnames(properties_refill_df)) {
    if (col == "random") {next}
    if (properties_refill_df[, sum(is.na(get(col)))] == 0) {
        all_vars_ <- unique(c(all_vars_, col))
    }
}

properties_refill_df <- properties_refill_df[, unique(c("parcelid", all_vars_)), with=F]
save(properties_refill_df, file="properties_refill_df.RData")


###########################  Categorical Variable Missing Value Imputation  ###########################
char_predictors_to_impute <- c("regionidzip", "censustractandblock", "regionidneighborhood",
                                "propertyzoningdesc", "propertycountylandusecode",
                                "regionidcity", "propertylandusetypeid", "rawcensustractandblock")
# filter out the variables with too many missing values
too_many_missing <- c()
# Check missing percentage
for (col in char_predictors_to_impute) {
    missing_ <- full[split == "train", sum(is.na(get(col)))] / full[, sum(split == "train")]
    if (missing_ > 0.1) {
        print(paste0("Variable --  ", col, "  -- has ", round(100*missing_,2),"% missing values..."))
        too_many_missing <- unique(c(too_many_missing, col))
    } else {
        print(paste0(col, ": ", round(100*missing_,2),"% missing ..."))
    }
}

char_predictors_to_impute <- copy(setdiff(char_predictors_to_impute, too_many_missing))
properties_refill_df_ <- unique(full[, c("parcelid",char_predictors_to_impute), with = F])

for (col in char_predictors_to_impute) {
    most_ <- sort(properties_refill_df_[, table(get(col))], decreasing=T)[1]
    most_ <- names(most_)
    class_ <- class(properties_refill_df_[, get(col)])
    if (class_ %in% c("numeric", "integer")) {
        most_ <- as.numeric(most_)
        if (class_ == "integer") {
            most_ <- as.integer(most_)
        }
    }
    print(most_)
    print(paste0("Processing variable ", col, "..."))
    properties_refill_df_[is.na(get(col)), paste0(col) := most_]
    properties_refill_df_[, paste0(col) := factor(get(col), levels=sort(properties_refill_df_[, unique(get(col))]))]
}

properties_refill_df <- merge(properties_refill_df, properties_refill_df_, by="parcelid")
rm(properties_refill_df_)

full_vars_to_keep <- c("parcelid", "logerror", "transactiondate", "split",
                        "freq_bin1", "freq_bin2", "freq_bin3", "freq_bin4",
                        "freq_bin5", "freq_bin6", "freq_bin7", "freq_bin8",
                        "freq_bin9", "freq_bin10", "logerror_pred")
full <- full[, full_vars_to_keep, with =F]
gc()
full <- merge(full, properties_refill_df, by="parcelid")


###########################  Add Transformation of Variables  ###########################
# Process some character variables
full <- char2num_dt(full, "rawcensustractandblock")
full[, censustractandblock := as.character(censustractandblock)]
full <- char2num_dt(full, "censustractandblock")
full <- substring_dt(full, "censustractandblock", range=c(1,4))
full <- substring_dt(full, "censustractandblock", range=c(1,8))
full <- substring_dt(full, "censustractandblock", range=c(1,12))
full <- substring_dt(full, "propertycountylandusecode", range=c(1,2))

# Add Date info from transaction date
full[, transactionMonth := as.factor(month(as.Date(as.character(transactiondate), "%Y-%m-%d")))]
full[, transactionDayofYear := yday(as.Date(as.character(transactiondate), "%Y-%m-%d"))]
full[, transactionDayofYear2 := transactionYear + transactionDayofYear/366]

# Log the numerical variables
num_vars_to_loop_ <- setdiff(colnames(full), c("parcelid", "logerror", "transactiondate", "split", "logerror_pred"))
for (col in num_vars_to_loop_) {
    class_ <- full[, class(get(col))]
    if (class_ %in% c("numeric", "integer")) {
        print(paste0("Taking log of variable ", col))
        min_ <- min(full[, get(col)], na.rm=T)
        full[, paste0(col,"_log") := log(get(col) - min_ + 1)]
    }
}
for (col in colnames(full)) {
    if (class(full[, get(col)]) == "character") {
        full[, paste0(col) := as.factor(get(col))]
    }
}

predictors3 <- copy(setdiff(colnames(full), c("parcelid", "logerror", "logerror_pred", "split", "transactiondate")))
predictors3 <- copy(predictors3)  # just in case

full[, rowidx := seq(1, nrow(full))]
full[, yearbuilt := as.factor(yearbuilt)]
#spline_vars <- c("finishedsquarefeet12", "taxamount", "structuretaxvaluedollarcnt",
#                "calculatedfinishedsquarefeet", "landtaxvaluedollarcnt", "taxvaluedollarcnt",
#                "finishedfloor1squarefeet", "finishedsquarefeet15", "lotsizesquarefeet")
spline_vars <- c("structuretaxvaluedollarcnt", "finishedsquarefeet12", "transactionDayofYear", "calculatedbathnbr_log", "taxamount")
spline_vars <- copy(intersect(spline_vars, colnames(properties_refill_df)))
#spline_vars <- c(spline_vars, paste0(spline_vars, "_log"))
spline_vars <- copy(intersect(spline_vars, colnames(full)))

new_cols_added <- c()
for (col in spline_vars) {
    class_ <- full[, class(get(col))]
    if (class_ %in% c("numeric", "integer")) {
        vec_ <- round(unique(full[, get(col)]), 7)
        m_ <- b_spline_creation2(vec_, col, knots_percentiles = c(0.03, 0.1, 0.25, 0.75, 0.9, 0.97))
        m_ <- m_$bspline
        cols <- copy(colnames(m_))
        for (col_ in cols) {
            if (length(unique(m_[, get(col_)])) == 1) {
                m_[, paste0(col_) := NULL]
            }
        }
        setnames(m_, col, "zzzz")
        m_[, zzzz := round(zzzz, 7)]
        full[, zzzz := round(get(col), 7)]
        full <- merge(full, m_, by="zzzz")
        setkey(full, rowidx)
        new_cols_added <- c(new_cols_added, copy(setdiff(colnames(m_), "zzzz")))
    }
}
rm(m_)

## Conver sparse matrix, since it's consumes too much memory, try two steps instead
all_vars_ <- unique(c(predictors3,new_cols_added))
types_ <- sapply(all_vars_, FUN=function(x) full[, class(get(x))])
names_ <- names(types_)
char_vars_ <- names_[types_ %in% c("character", "factor")]
num_vars_ <- unique(c(spline_vars, new_cols_added))
modelmatrix00 <- sparse.model.matrix(~.-1, full[, num_vars_, with=F])
gc()
modelmatrix01 <- sparse.model.matrix(~.-1, full[, char_vars_, with=F])
gc()
modelmatrix <- cbind(modelmatrix00, modelmatrix01)
gc()
names_ <- c(colnames(modelmatrix00), colnames(modelmatrix01))
rm(modelmatrix00, modelmatrix01); gc();
colnames(modelmatrix) <- names_


###########################  Model Building  ###########################

num_fields_glmnet <- c("censustractandblock60376513043011",
                            "structuretaxvaluedollarcnt",
                            "taxamount",
                            "finishedsquarefeet12",
                            "censustractandblock60375031032000",
                            "censustractandblock60372718012003",
                            "censustractandblock61110082012023",
                            "censustractandblock60372655101000",
                            "taxamount_s1",
                            "censustractandblock60376027004007",
                            "censustractandblock60371917102002",
                            "censustractandblock60372380005007",
                            "censustractandblock60374087032027",
                            "censustractandblock60371082021018",
                            "censustractandblock60371171022002",
                            "censustractandblock60372181101000",
                            "rawcensustractandblock60375990.002002",
                            "censustractandblock60377013041001",
                            "censustractandblock60591106041000",
                            "censustractandblock60375425021005",
                            "propertycountylandusecode0109",
                            "finishedsquarefeet12_s5",
                            "censustractandblock60372321202002",
                            "censustractandblock60372315001006",
                            "censustractandblock_1to12603780041010",
                            "censustractandblock61110009031003",
                            "yearbuilt1991.01110839844",
                            "structuretaxvaluedollarcnt_s2",
                            "taxamount_s4",
                            "censustractandblock60376705001016",
                            "finishedsquarefeet12_s1",
                            "censustractandblock60371112041015",
                            "censustractandblock60590626043023",
                            "taxamount_s6",
                            "censustractandblock60371891022004",
                            "rawcensustractandblock60371439.011",
                            "structuretaxvaluedollarcnt_s3",
                            "censustractandblock60372038003001",
                            "structuretaxvaluedollarcnt_s1",
                            "propertycountylandusecode1722",
                            "censustractandblock60372911102003",
                            "censustractandblock60371413031000",
                            "censustractandblock60371945002014",
                            "censustractandblock_1to860378004",
                            "regionidzip96013",
                            "yearbuilt1930",
                            "censustractandblock_1to860372657",
                            "taxamount_s3",
                            "structuretaxvaluedollarcnt_s5",
                            "structuretaxvaluedollarcnt_s4",
                            "regionidzip96951",
                            "regionidcity12447",
                            "taxamount_s2",
                            "censustractandblock60372079002009",
                            "rawcensustractandblock60376705.001016",
                            "censustractandblock60374077021000",
                            "censustractandblock_1to12605905241510",
                            "censustractandblock_1to12603762090420",
                            "censustractandblock60371200301002",
                            "transactionMonth4",
                            "yearbuilt1977",
                            "yearbuilt1961",
                            "censustractandblock60371397013006",
                            "regionidzip95984")
train_dm_reg <- xgb.DMatrix(
                    data = cbind(modelmatrix[full$split=="train", num_fields_glmnet], as.matrix(full[full$split=="train", logerror_pred])),
                    label = full[full$split=="train", logerror],
                    missing = NA
                )
gc()
test_dm_reg <- xgb.DMatrix(
                    data = cbind(modelmatrix[full$split=="test", num_fields_glmnet], as.matrix(full[full$split=="test", logerror_pred])),
                    missing = NA
        )
gc()
xgb2_glmnet <- xgb.train(
                    gblinear = "gblinear",
                    data = train_dm_reg,
                    #watchlist = list(train = train_dm_freq, valid = valid_dm_freq),
                    eval_metric = "mae",
                    #nfold = 8,
                    print_every_n = 20,
                    #early_stopping_rounds = 20,
                    objective = "reg:linear",
                    nthread = parallel::detectCores() -1,
                    alpha = 0.1,
                    lambda = 7,
                    nrounds = 26
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

write.csv(submission, file = "submission_0625_v1_glmnet_on_xgboost.csv", row.names = F)

if (FALSE) {
    # ensembel
    anothersubmission <- fread("./source_data/lgb_starter.csv", header = TRUE)
    weights <- c(0.67, 0.33)
    submission[, 2:7] <- ((weights[1]/sum(weights)) * anothersubmission[, 2:7] + (weights[2]/sum(weights)) * submission[, 2:7])
	submission[, 2:7] <- round(submission[, 2:7], 4)
}
