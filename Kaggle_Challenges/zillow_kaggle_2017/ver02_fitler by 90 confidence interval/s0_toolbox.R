
library(dplyr)
library(data.table)
library(xgboost)
library(gbm)
library(readr)


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


writecsv2h2o <- function(df) {
    readr::write_csv(df, "zzzzdf.csv")
    data_hex <- h2o.importFile(path = "zzzzdf.csv")
    vars_to_loop <- h2o.colnames(data_hex)[unlist(h2o.getTypes(data_hex)) == "enum"]
    for (c in vars_to_loop) {
        data_hex[data_hex[, c]=="NaN", c]=NA
    }
    file.remove("zzzzdf.csv")
    return(data_hex)
}

