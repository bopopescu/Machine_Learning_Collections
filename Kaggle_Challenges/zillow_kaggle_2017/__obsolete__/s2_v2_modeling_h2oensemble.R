setwd("E:/zillow_kaggle_2017/")

library(dplyr)
library(data.table)
library(xgboost)
library(h2o)

load("s2_line415.RData")
# This is the .RData from the s2_modeling_xgboost


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


predictors_ <- intersect(setdiff(c(predictors, new_predictors), c(non_predictors)), colnames(full))
setkey(full, rowidx)
for (col in predictors_) {
    if (length(unique(full[split=="train", get(col)]))==1){
        full[, paste0(col) := NULL]
        print(paste0("Column ", col, " is removed due to unique distinct value."))
    }
}
predictors_ <- intersect(predictors_, colnames(full))
full_mm <- data.table(data.matrix(full) * 1.0)
full <- full[, intersect(non_predictors, copy(colnames(full))), with = F]  # save memory
gc()

# Start H2O Cluster
h2o.init(min_mem_size="70G")

#full_h2o <- writecsv2h2o(full_mm)
readr::write_csv(full_mm, "zzzzdf.csv")
rm(full_mm); gc();
full_h2o <- h2o.importFile(path = "zzzzdf.csv")
vars_to_loop <- h2o.colnames(full_h2o)[unlist(h2o.getTypes(full_h2o)) == "enum"]
for (c in vars_to_loop) {
    full_h2o[full_h2o[, c]=="NaN", c]=NA
}
file.remove("zzzzdf.csv")
gc();


############### Stack Ensemble ###########
randomseed <- 10011
nfolds <- 20

trainidx <- which(full$split=="train" & full$foldid >= 16)
valididx <- which(full$split=="train" & full$foldid < 16)
h2o_vars <- c(predictors_, "logerror")
h2o.impute(full_h2o, method ="median")
htrain <- full_h2o[trainidx, h2o_vars]
hvalid <- full_h2o[valididx, h2o_vars]

# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = predictors_,
                        y = "logerror",
                        training_frame = htrain,
                        distribution = "laplace",
                        stopping_metric = "MAE",
                        ntrees = 2000,
                        max_depth = 3,
                        min_rows = 2,
                        learn_rate = 0.03,
                        col_sample_rate_per_tree = 0.8,
                        nfolds = nfolds,
                        sample_rate = 0.632,
                        col_sample_rate = 0.8,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE,
                        stopping_rounds = 20,
                        seed = randomseed)

# Train & Cross-validate a Deep Learning
my_dl <- h2o.deeplearning(
                    training_frame = htrain,
                    x = predictors_,
                    y = "logerror",
                    hidden=c(200, 100, 50, 50),
                    epochs=10000,
                    nfolds = nfolds,
                    l1 = 0.07,
                    l2 = 0.001,
                    activation = "RectifierWithDropout",
                    hidden_dropout_ratios = c(0.5, 0.5, 0.5, 0.5),
                    standardize = TRUE,
                    loss = "Absolute",
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    stopping_rounds=20,
                    distribution = "laplace",
                    stopping_metric="MAE",
                    stopping_tolerance=0.0001,
                    seed = randomseed)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = predictors_,
                                y = "logerror",
                                training_frame = htrain,
                                model_id = "my_ensemble_1",
                                base_models = list(my_gbm@model_id, my_dl@model_id))
# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = hvalid)


############### Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = hvalid)  # 0.06692397
perf_dl_test <- h2o.performance(my_dl, newdata = hvalid)
baselearner_best_test <- min(h2o.mae(perf_gbm_test), h2o.mae(perf_dl_test))
ensemble_test <- h2o.mae(perf)
print(sprintf("Best Base-learner Test MAE:  %s", baselearner_best_test))
print(sprintf("Ensemble Test MAE:  %s", ensemble_test))


################ Create prediction
submission_df_h2o <- h2o.predict(object = ensemble, newdata = full_h2o[which(full$split=="test"), h2o_vars])




#############  LEGACY CODES   ###############
if (FALSE) {
    ########  GBM Hyperparamters  ########
    learn_rate_opt <- c(0.03)
    max_depth_opt <- c(2, 3, 4)
    sample_rate_opt <- c(0.632)
    col_sample_rate_opt <- c(0.8)
    hyper_params <- list(learn_rate = learn_rate_opt,
                        max_depth = max_depth_opt,
                        sample_rate = sample_rate_opt,
                        col_sample_rate = col_sample_rate_opt,
                        min_rows = 2,
                        stopping_rounds = 20,
                        stopping_metric = "MAE",
                        distribution = "laplace")
    search_criteria <- list(strategy = "RandomDiscrete",
                            max_models = 3,
                            seed = randomseed)

    my_gbm <- h2o.grid(algorithm = "gbm",
                        grid_id = "gbm_grid_mae",
                        x = predictors_,
                        y = "logerror",
                        training_frame = htrain,
                        ntrees = 2000,
                        seed = randomseed,
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE,
                        hyper_params = hyper_params,
                        search_criteria = search_criteria)


    ########  Deep Learning Model Tuning  ########
    hyper_params <- list(
        activation=c("Rectifier","RectifierWithDropout"),
        hidden=list(c(200, 200, 200)),
        input_dropout_ratio=c(0,0.05),
        l1= c(0.01, 0.1, 1, 10),
        l2= c(0.01, 0.1, 1, 10)
        )
    ### Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
    search_criteria = list(strategy = "RandomDiscrete",
                            max_runtime_secs = 3600,
                            max_models = 100,
                            seed=1234567,
                            stopping_rounds=8,
                            stopping_tolerance=1e-2)
    dl_random_grid <- h2o.grid(
                        algorithm="deeplearning",
                        grid_id = "dl_grid_random",
                        training_frame = htrain,
                        validation_frame = hvalid,
                        x = predictors_,
                        y = "logerror",
                        epochs=5000,
                        distribution = "laplace",
                        stopping_metric="MAE",  ## could be "MSE","logloss","r2"
                        stopping_tolerance=0.001,
                        seed = randomseed,
                        stopping_rounds=20,
                        # score_duty_cycle=0.025,  ## don't score more than 2.5% of the wall time
                        max_w2=10,  ## can help improve stability for Rectifier
                        hyper_params = hyper_params,
                        search_criteria = search_criteria
    )


    ########################
    x_ <- h2o.deeplearning(training_frame = htrain,
                    validation_frame = hvalid,
                    x = predictors_,
                    y = "logerror",
                    hidden=c(200, 100, 50, 50),
                    epochs=5000,
                    activation = "RectifierWithDropout",
                    l1 = 0.07,
                    l2 = 0.001,
                    hidden_dropout_ratios = c(0.5, 0.5, 0.5, 0.5),
                    standardize = TRUE,
                    loss = "Absolute",
                    mini_batch_size = 128,
                    #nfolds = nfolds,
                    #fold_assignment = "Modulo",
                    #keep_cross_validation_predictions = TRUE,
                    stopping_rounds=20,
                    distribution = "laplace",
                    stopping_metric="MAE", ## could be "MSE","logloss","r2"
                    stopping_tolerance=0.0001,
                    seed = randomseed)
    x_@model$training_metrics
    x_@model$validation_metrics

    ########################
    # Train & Cross-validate a RF
    my_rf <- h2o.randomForest(x = predictors_,
                        y = "logerror",
                        training_frame = htrain,
                        ntrees = 1000,
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE,
                        stopping_metric = "MAE",
                        stopping_rounds = 20,
                        seed = randomseed)
}
