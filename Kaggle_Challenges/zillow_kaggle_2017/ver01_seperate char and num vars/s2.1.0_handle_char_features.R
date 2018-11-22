setwd("E:/zillow_kaggle_2017/")

require(dplyr)
require(data.table)
require(xgboost)
require(lightgbm)
require(glmnet)
require(readr)


###########################  Functions  ###########################
char2num_dt <- function(data_table, column) {
	# This is to convert the character variables to numerical variables
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
    # This function is to take the substring of the column
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

convert_2_char <- function(data_table, columns) {
	# This function converts the columns from whatever charcter
	for (col in columns) {
		if (col %in% colnames(data_table)) {
			if (class(data_table[1, get(col)]) != "character") {
				data_table[, paste0(col) := as.character(get(col))]
			}
		}
	}
	return(data_table)
}

mae <- function(preds, dtrain) {
	# customized mae function in the xgboost function
    labels <- getinfo(dtrain, "label")
    weight <- rep(1, length(labels))
    try({weight <- getinfo(dtrain, "weight")}, silent=T)
    err <- sum(weight * abs(labels-preds))/sum(weight)
    return(list(metric = "mae",  value=err))
}
# in case if the xgboost version is old, use this
# as the eval_metric, just use feval = mae in xgb


distinct_field <- function(x) {
    # x is the vector of variable values to be distincted with order
    x <- data.table(x)
    length_ <- nrow(x)
    x[, ids := seq(1, length_)]
    x2 <- x[, lapply(.SD, min, na.rm=T), by = "x", .SDcols = "ids"]
    setkey(x2, ids)
    return(x2$x)
}


label_encode_factor <- function(dt, col){
	# This funciton converts character column col to factor
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
	# This funciton converts character column col to factor in batch for all possible variables
    # dt is the data table
    for (col in colnames(dt)) {
        dt <- label_encode_factor(dt, col)
    }
    return(dt)
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



############################################################
############################################################
# Load in the data and specify the variable types
############################################################
############################################################
### Load in the files
full <- fread("./full.csv", na.strings = c("NA","na","NAN","","null", "NULL",
										"--", "-", "**", "*", "N/A","n/a", "Missing",
										"MISSING", "missing"))

# add some missing variables back
full[, numberofstories := as.integer(exp(numberofstories_log))]
full[, yardbuildingsqft17_log := log(yardbuildingsqft17)]
full[, finishedsquarefeet6_log := log(finishedsquarefeet6)]
######################
# Common Setting
non_predictors <- c("parcelid", "transactiondate", "logerror", "split", "random", "foldid",
                    "logerror_pred", "rowidx")

predictors <- fread("./Var_Importance.csv")$Feature
month_vars <- c("transactionMonth", "transactionYear")

# character columns
char_predictors <- fread("./char_vars.csv")$Feature
pure_char_predictors <- fread("./char_vars.csv")$Pure_Character
# numerical columns
num_predictors <- fread("./num_vars.csv")$Feature
# filter out the variables not in the full table
non_predictors <- copy(intersect(colnames(full), non_predictors))
predictors <- copy(intersect(colnames(full), predictors))
char_predictors <- copy(intersect(colnames(full), char_predictors))
pure_char_predictors <- copy(intersect(colnames(full), pure_char_predictors))
num_predictors <- copy(intersect(colnames(full), num_predictors))
#
cat(paste0("The varaibles that can be used as both character and numerical variables are:"))
cat(paste0(paste0("'"), intersect(char_predictors, num_predictors), collapse="', "), paste0("'"))
# Convert the character variables if they are loaded not as characters
full <- convert_2_char(full, pure_char_predictors)
full <- label_encode_factor2(full)
for (col in non_predictors) {if (class(full[, get(col)]) == "factor") {full[, paste0(col) := as.character(get(col))]}}
full <- factorNA(full)

# Create Sparse Matrix on Character or Numerical (but can be deemed as character) variables
previous_na_action <- options('na.action')
options(na.action='na.pass')
x_ <- c(); for (col in char_predictors) {if (length(unique(full[, get(col)]))==1) {x_ <- c(x_, col)}}
full_sparse <- sparse.model.matrix(~.-1, full[, setdiff(char_predictors, x_), with=F])
options(na.action=previous_na_action$na.action)
gc();

if (FALSE) {
	save.image("s2.1.0_lastrow.RData", compress=F)
	save(full_sparse, file="full_sparse.RData", compress=F)
}
