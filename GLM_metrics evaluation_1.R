mylift<-function(orderby, pred, actual, w, n) {
  if (length(w)==0) {
    w<-rep(1.0, length(pred))
  }
  v<-data.frame(o=orderby, p=pred, a=actual, w=w)
  v<-v[order(-v$o),]
  v2<-v[order(-v$a),]
  #print(head(v, 100))
  v$cumm_w<-cumsum(v$w)
  v$cumm_y<-cumsum(v$w*v$a)
  total_w<-sum(v$w)
  gini<-with(v,2* sum(cumm_y*w)/(sum(a*w)*total_w)-1)
  print(paste("gini=", gini))
  v$pidx<-round(v$cumm_w*n/total_w+0.5)
  v$pidx[v$pidx>n]<-n
  v$pidx[v$pidx<1]<-1
  orf = sum(v$a*v$w)/sum(v$p*v$w)
  sum1<-sqldf("select pidx, sum(w) as w, min(o) as min, max(o) as max, sum(p*w)/sum(w) as p, sum(a*w)/sum(w) as a, min(a) as min_a, max(a) as max_a, sum(a*w) as a_cnt from v group by pidx")
  sum1$p_orf<-sum1$p*orf
  sum1$err<-with(sum1, a/p_orf)
  sum1$avg_a <- sum(v$a*v$w)/sum(v$w)
  sum1$a_lift <- with(sum1,a_cnt/(avg_a*w))
  sum1$p_lift <- with(sum1,p_orf/avg_a)
  
  print(sum1)
  with(sum1,plot(p_orf, type='l', ylim=c(min(a,p_orf), max(a,p_orf)),xaxt='n', col="blue",lwd = 3,ylab='prediction',xlab='quantiles'))
  axis(1, at=1:20, 1:20) 
  
  #   legend("topright",("predicted", "actual"), lty=1:2,col=c( "blue","red"))
  legend("topright",legend=c("predicted","actual","mean"), lty=c(1,1,2), col=c( "blue","red","green"))
  
  v2$cumm_w<-cumsum(v2$w)
  v2$cumm_y<-cumsum(v2$w*v2$a)
  max_gini<-with(v2,2*sum(cumm_y*w)/(sum(a*w)*total_w)-1)
  normalized_gini<-gini/max_gini
  result<-c(gini,max_gini,normalized_gini,sum1$a_lift[1])
  
  legend("top", legend=c("gini:",round(normalized_gini,3)), cex = 0.8)
  
  lines(sum1$a_cnt/sum1$w, col="red",type='b',lwd = 3)
  abline(h=sum1$avg_a, col ="green",lty=2,lwd = 2)
  text(1, sum1$a[1],paste("lift =",round(sum1$a_lift[1],3)), pos=4,cex= 0.8)
  
  sum1
  
  
  return(result)
}

# lift chart example
#with(d1[d1$split1==0,], mylift(orderby=pred_cc, pred=pred_cc, actual=y, w=NULL, n=10))


# leave one out to convert categorical variable into numerical variables
#looTrain <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0)

#looTest <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0)


#### gini function
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


WeightedGini <- function(solution, weights, submission){
  df = data.frame(solution = solution, weights = weights, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = cumsum((df$weights/sum(df$weights)))
  totalPositive <- sum(df$solution * df$weights)
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / totalPositive
  n <- nrow(df)
  gini <- sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
  return(gini)
}

NormalizedWeightedGini <- function(solution, weights, submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}


# wrap up into a function to be called within xgboost.train
evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini(as.numeric(labels),as.numeric(preds))
  return(list(metric = "Gini", value = err))
}





rMSE = function(y_pred,y)
{
  sqrt(mean((y-y_pred)^2))
}

MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- nrow(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(nrow(act))      
  return(ll);
}