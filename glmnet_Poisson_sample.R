load("PoissonExample.RData")

library(glmnet)
library(hydroGOF)

fit = glmnet(x, y, family = "poisson", standardize = TRUE, intercept=TRUE);
cvfit = cv.glmnet(x, y, family = "poisson", standardize = TRUE, intercept=TRUE, nfolds=10);
plot(cvfit);
opt.lam = c(cvfit$lambda.min, cvfit$lambda.1se);
coef(cvfit, s = opt.lam);
pred = predict(cvfit, newx = x, type = "response", s = c(opt.lam));
pred = cbind(y, pred);
colnames(pred) = c("y", "pred_lambda_min", "pred_lambda_lse");
rmse_lambda_min = rmse(pred[,1], pred[,2]);
rmse_lambda_lse = rmse(pred[,1], pred[,3]);

###########
# shorter x matrix for demo 
########### 

x = x[,c(1:2)];
fit = glmnet(x, y, family = "poisson");
cvfit = cv.glmnet(x, y, family = "poisson", nfolds=10);
plot(cvfit);
opt.lam = c(cvfit$lambda.min, cvfit$lambda.1se);
coef(cvfit, s = opt.lam);
pred = predict(cvfit, newx = x, type = "response", s = c(cvfit$lambda.min));
pred = cbind(y, pred);
colnames(pred) = c("y", "pred_lambda_min");
rmse_lambda_min = rmse(pred[,1], pred[,2]);
coef_lambda_min = coef(cvfit, s = c(cvfit$lambda.min));
coef_lambda_min;
coef_lambda_min = as.matrix(coef_lambda_min);
x_new = cbind(1,x);
pred_new = exp(x_new %*% coef_lambda_min) 

# This demostrates that glmnet uses link function log automatically # when loss distribution is poisson