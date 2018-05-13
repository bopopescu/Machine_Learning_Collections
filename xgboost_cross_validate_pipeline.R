library(xgboost)
library(data.table)
require(stringr)


###########################################################################
# COMMONLY USED FUNCTIONS
cross_validate_prediciton <- function(data_X,  params,  seed,  best_iter,  nfolds,
                                    labels,  weights, save_cv_models=TRUE,
                                    model_persist_dir = NULL,
                                    model_persist_name = NULL) {
    ############
    # ENSURE the
    #   data_X, params, seed, best_iter, nfolds,
    #   (labels, weights are the same label and weight, data_X is data X-Matrix)
    #   is used same as xgb.cv model tuning

    # For example
    # params <- list(
    #     objective = 'binary:logistic',
    #     eval_metric = 'auc',
    #     scale_pos_weight = pos_weight_adj, # adjust positive weight
    #     max_depth = 3,
    #     subsample = 0.6,
    #     colsample_bytree = 1,
    #     alpha = 1,
    #     lambda =100,
    #     eta = 0.1,
    #     maximize = TRUE)
    # set.seed(seed)
    # xgb_cv <- xgb.cv(
    #     data = d_train,
    #     nfold = nfolds,
    #     params = params,
    #     nrounds = 5000,
    #     early_stopping_rounds = 50,
    #     print_every_n = 50
    #     )
    ### Cross Validated Scores
    set.seed(seed)
    df_ = data.table(cv_foldid = runif(nrow(data_X)))
    df_[, cv_foldid := ceiling(cv_foldid * nfolds)]
    folds = df_[, sort(unique(cv_foldid))]
    scores = rep(-999999, nrow(df_))
    OOB_metric = NULL
    OOB_metric_db = list()
    for (foldid in folds) {
        print(paste0("Processing foldid ", foldid, " out of ", length(folds)," folds for CV predictions"))
        train_idx_ = df_[, which(cv_foldid != foldid)]
        test_idx_ = df_[, which(cv_foldid == foldid)]
        d_train_ <- xgb.DMatrix(data = data_X[train_idx_,],
                                label = labels[train_idx_],
                                weight = weights[train_idx_],
                                missing = NA)
        d_test_ <- xgb.DMatrix(data = data_X[test_idx_,],
                                label = labels[test_idx_],
                                weight = weights[test_idx_],
                                missing = NA)
        set.seed(seed)
        alliter = min(round(best_iter * 1.3), best_iter + 100)
        bst_ <- xgb.train(
            data = d_train_,
            params = params,
            watchlist = list(train=d_train_, test=d_test_),
            nrounds = alliter,
            print_every_n = 100
            )
        OOB_metric_db[[paste0(foldid)]] = bst_$evaluation_log
        scores[test_idx_] = predict(bst_, d_test_)
        if (save_cv_models) {
            if (is.null(model_persist_dir)) {
                dir.create("XGBoost_CV_Model_Save", showWarnings = F)
            } else {
                if (!file.exists(model_persist_dir)) {
                    dir.create(model_persist_dir, showWarnings = F)
                }
            }
            if (is.null(model_persist_name)) {
                cvmodelname = "xgb_cv_model"
            } else {
                cvmodelname = model_persist_name
            }
            foldid_new = stringr::str_pad(foldid, nchar(as.character(10)), pad = "0")
            xgb.save(bst_, file.path(model_persist_dir, paste0(cvmodelname, foldid_new,".model")))
            # load binary model to R
            #bst2 <- xgb.load("xgboost.model")
        }
        rm(bst_, train_idx_, test_idx_, d_train_, d_test_)
    }
    rm(df_)
    bestiter_ = best_iter
    if (is.null(params[['maximize']])) {maximize = TRUE} else {maximize=params[['maximize']]}
    if (maximize) {best_metric = -1 * (1e20)} else {best_metric = 1e20}
    for (i in seq(round(alliter*0.6), alliter)) {
        OOB_metric = NULL
        for (key in names(OOB_metric_db)) {
            dt = OOB_metric_db[[key]]
            OOB_metric = c(OOB_metric, dt[i, get(colnames(dt)[ncol(dt)])])
        }
        if (maximize) {
            if (mean(OOB_metric) > best_metric) {
                bestiter_ = i
                best_metric = mean(OOB_metric)
            }
        } else {
            if (mean(OOB_metric) < best_metric) {
                bestiter_ = i
                best_metric = mean(OOB_metric)
            }
        }
    }
    OOB_metric = NULL
    for (key in names(OOB_metric_db)) {
        dt = OOB_metric_db[[key]]
        OOB_metric = c(OOB_metric, dt[bestiter_, get(colnames(dt)[ncol(dt)])])
    }
    print(paste0("error_metric values on each OOB (CV oob fold) are:  ",
            paste0(OOB_metric, collapse=", "), " with best_iteration ", bestiter_))
    # save best iteration number
    if (save_cv_models) {
        write.csv(bestiter_,
                file=file.path(model_persist_dir, paste0(cvmodelname, "_best_iteration",".txt")),
                row.names = F, quote = F)
    }
    #attributes(x)$names
    return(scores)
}


function(full_list, pattern) {
    # sample usage
    # find_elements_with_pattern(c("ab", "cdef", "abcef"), "*a*b*")
    return(full_list[grepl(glob2rx(pattern), full_list)])
}


cross_validate_prediciton_OOB = function(model_persist_dir, model_persist_name,
                                        data_submit, best_iter) {
    #### This function is to make prediction based on average of
    #    N-fold xgboost model predictions
    # model_persist_dir is the directory saving all CV XGBoost models
    # model_persist_name is the common pattern of CV XGBoost model names
    # data_submit is the data matrix for CV prediction, the unseen data
    # best_iter is the best iteration of all CV XGBoost models calculated above
    files = list.files(path = model_persist_dir, pattern = paste0("*.model"))
    files = find_elements_with_pattern(files, paste0(model_persist_name, "*.model"))
    cvscores_ <- rep(0, nrow(data_submit))
    for (f_ in files) {
        bst_ = xgb.load(file.path(model_persist_dir, f_))
        d_score_ = xgb.DMatrix(data = data_submit, missing = NA)
        cvscores_ = cvscores_ + predict(bst_, d_score_, ntreelimit = best_iter)
    }
    cvscores_ = cvscores_ / length(files)  # average of cross validated scores
    return(cvscores_)
}

#  END OF FUNCTIONS
###########################################################################

load("test_geo_xgboost_purepremium_model.RData")

seed = 10038
exposure_field = "exp"
loss_field = "loss" # incurred loss
# the predicted pure premium from other variables used as offset here
pred_pp_offset_variable = "pp_pred_offset" # Rest of variables in data are census data for geo-spatial model

predictors = setdiff(colnames(data), c(loss_field))
data_X <- data.matrix(data[, predictors, with = F])
#data_X <- data.matrix(data[, c(exposure_field, pred_pp_offset_variable), with = F])

############################
# Define the binary cutoff
# of target y
Y_BINARY_CUTOFF = 0
N_FOLDS = 10
############################
print("The exp and pp_pred_offset gave max auc 0.725 for Y_BINARY_CUTOFF=0")
labels = data[, 1*(get(loss_field) > Y_BINARY_CUTOFF)]
weights = data[, get(exposure_field)]

d_train <- xgb.DMatrix(data = data_X,
                        label = labels,
                        weight = weights,
                        missing = NA)
params <- list(
    objective = 'binary:logistic',
    eval_metric = 'auc',
    #scale_pos_weight = pos_weight_adj,
    max_depth = 3,
    subsample = 0.6,
    colsample_bytree = 1,
    alpha = 1,
    lambda =100,
    eta = 0.1,
    maximize = TRUE)
set.seed(seed)
xgb_cv <- xgb.cv(
    data = d_train,
    nfold = N_FOLDS,
    params = params,
    nrounds = 5000,
    early_stopping_rounds = 50,
    print_every_n = 50
    )
best_iter = xgb_cv$best_iteration


###########################################################################
# Example of Use Cases
###########################################################################
########## Score on Train and New Data ##########
# data_X is the train data that we score on for validation
cv_predictions = cross_validate_prediciton(data_X,  params,  seed,  best_iter,  nfolds=N_FOLDS,
                                labels,  weights, save_cv_models=TRUE,
                                model_persist_dir = "E:/_temp_/trash/cvoutputs",
                                model_persist_name = "XGBoost_FirstCut")
# data_submit is brand new data
new_predictions = cross_validate_prediciton_OOB(model_persist_dir = "E:/_temp_/trash/cvoutputs",
                        model_persist_name = "XGBoost_FirstCut",
                        data_submit  = data_X,
                        best_iter = 253)

