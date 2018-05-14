library(xgboost)
library(data.table)
require(stringr)


###########################################################################
# COMMONLY USED FUNCTIONS
cross_validate_prediciton <- function(data_X,  params,  seed,  n_iters,  nfolds,
                                    labels,  weights, save_cv_models=TRUE,
                                    model_persist_dir = NULL,
                                    model_persist_name = NULL) {
    ############
    # ENSURE the
    #   data_X, params, seed, n_iters, nfolds,
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
    xgb_models_list = NULL
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
        alliter = min(round(n_iters * 1.3), n_iters + 100)
        bst_ <- xgb.train(
            data = d_train_,
            params = params,
            watchlist = list(train=d_train_, test=d_test_),
            nrounds = alliter,
            print_every_n = 100
            )
        OOB_metric_db[[paste0(foldid)]] = bst_$evaluation_log
        eval(parse(text=paste0("bst_",foldid, " <- bst_")))
        xgb_models_list <- c(xgb_models_list, paste0("bst_",foldid))
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
    bestiter_ = n_iters
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
    # Score on OOB folds
    for (foldid in folds) {
        eval(parse(text=paste0("bst_ <- bst_",foldid)))
        test_idx_ = df_[, which(cv_foldid == foldid)]
        d_test_ <- xgb.DMatrix(data = data_X[test_idx_,], missing = NA)
        scores[test_idx_] = predict(bst_, d_test_, ntreelimit=bestiter_)
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


find_elements_with_pattern = function(full_list, pattern) {
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
    prediction = TRUE,
    nrounds = 5000,
    early_stopping_rounds = 50,
    print_every_n = 50
    )
cv_predictions_by_xgbcv = xgb_cv$pred
gini = NormalizedWeightedGini(solution = labels, weights = weights, submission = cv_predictions_by_xgbcv)
print(paste0("Cross Validated AUC on Training Data is ", round(0.5*(1+gini),4)))
rm(gini)
best_iter = xgb_cv$best_iteration


###########################################################################
# Example of Use Cases
###########################################################################
########## Score on Train and New Data ##########
# data_X is the train data that we score on for validation
cv_predictions = cross_validate_prediciton(data_X,  params,  seed,  n_iters=best_iter,  nfolds=N_FOLDS,
                                labels,  weights, save_cv_models=TRUE,
                                model_persist_dir = "E:/_temp_/trash/cvoutputs",
                                model_persist_name = "XGBoost_FirstCut")
gini = NormalizedWeightedGini(solution = labels, weights = weights, submission = cv_predictions)
print(paste0("Cross Validated AUC by cross_validate_prediciton function on Training Data is ",
        round(0.5*(1+gini),4)))
# data_submit is brand new data
new_predictions = cross_validate_prediciton_OOB(model_persist_dir = "E:/_temp_/trash/cvoutputs",
                        model_persist_name = "XGBoost_FirstCut",
                        data_submit  = data_X,
                        best_iter = 253)

########## Pure Premium Model Checks ##########
data_X <- data.matrix(data[, setdiff(predictors, pred_pp_offset_variable), with = F])
# Create the Binary Classfication by Severity Cutoffs
cutoffs = c(0,  468, 734, 1309, 1969, 2920, 4100, 5781, 8940, 15324, 36859, 73377, 183975)
counter = 1
for (Y_BINARY_CUTOFF in cutoffs) {
    print(paste0("Creating the binary classfications by cutoff ",Y_BINARY_CUTOFF))
    # Define the binary cutoff of target y
    N_FOLDS = 10
    labels = data[, 1*(get(loss_field) > Y_BINARY_CUTOFF)]
    weights = data[, get(exposure_field)]
    d_train <- xgb.DMatrix(data = data_X,
                            label = labels,
                            weight = weights,
                            missing = NA)
    params <- list(
        objective = 'binary:logistic',
        eval_metric = 'auc',
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
        prediction = TRUE,
        nrounds = 5000,
        early_stopping_rounds = 50,
        print_every_n = 100
        )
    cv_predictions_by_xgbcv = xgb_cv$pred
    var_new = paste0("freq_pred_", counter)
    data[, paste0(var_new) := cv_predictions_by_xgbcv / weights]
    counter = counter + 1
}

freq_pred_var_names = paste0("freq_pred_", seq(1,counter-1))


######################################
# Run Elastic Net PP Model
set.seed(seed)
require(sqldf)
require(glmnet)
sample_size <- floor(0.5 * nrow(data))
train_index <- sample(seq_len(nrow(data)), size = sample_size)
train_index = sort(train_index)
test_index <- setdiff(seq_len(nrow(data)), train_index)

predictors =  freq_pred_var_names
x_matrix <- log(data.matrix(data[, predictors, with = F]))
y_matrix <- data[, get(loss_field)/get(exposure_field)]

fits <- cv.glmnet(x = x_matrix[train_index, ],
               y = y_matrix[train_index],
               family = "poisson",
               weights = weights[train_index],
               offset = data[train_index, log(get(pred_pp_offset_variable))],
               standardize = T,
               nfolds = 20,
               alpha = 1,
               thresh = 1e-5,
               keep = TRUE, # keeps the cross-validated scores instead, but large matrix, default is FALSE
               parallel = TRUE)
best_lambda = fits$lambda.min
preds = predict(fits, newx = x_matrix, type='response', s=best_lambda,
                offset=data[, log(get(pred_pp_offset_variable))])
# If we need to check model performance, the cv_preds is cross-valided scores that will be
# used for checking model performance through CV
cv_preds = exp(fits$fit.preval[, which(abs(fits$lambda - best_lambda)<1e-5)])

# Check the GINI and Lift After the Stacking
mylift2(preds[train_index], preds[train_index],
        data[train_index, get(loss_field)/get(exposure_field)],
        data[train_index,get(exposure_field)], 10,
        image_name="zzzzz_stacking_train.png",
        image_title = "train")
mylift2(preds[test_index], preds[test_index],
        data[test_index, get(loss_field)/get(exposure_field)],
        data[test_index,get(exposure_field)], 10,
        image_name="zzzzz_stacking_test.png",
        image_title = "test")

# Check the GINI and Lift before introducing new variables
mylift2(data[train_index, get(pred_pp_offset_variable)],
        data[train_index, get(pred_pp_offset_variable)],
        data[train_index, get(loss_field)/get(exposure_field)],
        data[train_index,get(exposure_field)], 10,
        image_name="zzzzz_train.png",
        image_title = "train")
mylift2(data[test_index, get(pred_pp_offset_variable)],
        data[test_index, get(pred_pp_offset_variable)],
        data[test_index, get(loss_field)/get(exposure_field)],
        data[test_index,get(exposure_field)], 10,
        image_name="zzzzz_test.png",
        image_title = "test")

