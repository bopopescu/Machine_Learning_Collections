##  This Program has wMAPE value of 0.0218324 ##

library(gbm)
library(parallel)

rm(list=ls());


report = data.frame(zone=NA,month=NA,MAPE=NA,wMAPE=NA,OverEst=NA);
tempreport = report;
report = report[0,];

load(file.path(getwd(),'test_data/Data_for_ShortTerm_Forecast.RData'));

modeldata$Weekday = as.factor(modeldata$Weekday);

features_not_incl = c("Datenum","Date","Zone","TestMonth","Partition");
target =  "y";
features = setdiff(names(modeldata),c(target,features_not_incl));
allFieldsinModel = union(target,features);

myformula = paste(target," ~ ",paste(features,collapse=" + "),sep="");
myformula = as.formula(myformula);

newreport = tempreport;
stm = Sys.time();

gbmmodel = gbm(myformula,
               data=modeldata[modeldata$Partition=='Train',allFieldsinModel], 
               distribution = "laplace", interaction.depth = min(15,length(features)),
               bag.fraction = 0.5 , shrinkage = 0.005, n.minobsinnode =1, n.trees = 20000, 
               keep.data=FALSE, cv.folds = parallel::detectCores());
time_for_training = Sys.time() - stm;
time_for_training;

modelname = paste('gbmmodel',sep="");
cmd = paste(modelname,' = gbmmodel',sep="");
eval(parse(text=cmd));

# ###
# cmd = paste('save(',modelname,',file="',modelname,'.Rdata")',sep="");
# eval(parse(text=cmd));
# ###

bestiter <- gbm.perf(gbmmodel, method="cv");
var_sumry = summary(gbmmodel,n.trees=bestiter);

modeldata$pred = predict(gbmmodel,modeldata,bestiter);

testdata = modeldata[modeldata$Partition == 'Test',];
testdata$error = testdata$pred - testdata[,target];

mape = mean(abs(testdata$pred / testdata[,target] -1));

wmape = sum(abs(testdata$error))/sum(testdata[,target]);

testdata$overestind = 1 * (testdata$error >= 0);
overest_perc = sum(testdata$overestind) / length(testdata$overestind);

newreport$MAPE = mape;
newreport$wMAPE = wmape;
newreport$OverEst = overest_perc;

report <- rbind(report,newreport)
print(report)
