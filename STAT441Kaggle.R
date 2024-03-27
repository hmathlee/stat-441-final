X_train <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/X_train.csv", stringsAsFactors=TRUE)
X_test <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/X_test.csv", stringsAsFactors=TRUE)
y_train <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/y_train.csv")

library(data.table)
library(xgboost)
library(tidyverse)
set.seed(20843361)

imputeDK <- function(v, vDK){
  vmod <- vDK[which(v == -4)]
  v <- vmod
}

X_train$v176[which(X_train$v176 == -4)] <- imputeDK(X_train$v176, X_train$v176_DK)
X_train$v177[which(X_train$v177 == -4)] <- imputeDK(X_train$v177, X_train$v177_DK)
X_train$v178[which(X_train$v178 == -4)] <- imputeDK(X_train$v178, X_train$v178_DK)
X_train$v179[which(X_train$v179 == -4)] <- imputeDK(X_train$v179, X_train$v179_DK)
X_train$v180[which(X_train$v180 == -4)] <- imputeDK(X_train$v180, X_train$v180_DK)
X_train$v181[which(X_train$v181 == -4)] <- imputeDK(X_train$v181, X_train$v181_DK)
X_train$v182[which(X_train$v182 == -4)] <- imputeDK(X_train$v182, X_train$v182_DK)
X_train$v183[which(X_train$v183 == -4)] <- imputeDK(X_train$v183, X_train$v183_DK)
X_train$v221[which(X_train$v221 == -4)] <- imputeDK(X_train$v221, X_train$v221_DK)
X_train$v222[which(X_train$v222 == -4)] <- imputeDK(X_train$v222, X_train$v222_DK)
X_train$v223[which(X_train$v223 == -4)] <- imputeDK(X_train$v223, X_train$v223_DK)
X_train$v224[which(X_train$v224 == -4)] <- imputeDK(X_train$v224, X_train$v224_DK)

X_test$v176[which(X_test$v176 == -4)] <- imputeDK(X_test$v176, X_test$v176_DK)
X_test$v177[which(X_test$v177 == -4)] <- imputeDK(X_test$v177, X_test$v177_DK)
X_test$v178[which(X_test$v178 == -4)] <- imputeDK(X_test$v178, X_test$v178_DK)
X_test$v179[which(X_test$v179 == -4)] <- imputeDK(X_test$v179, X_test$v179_DK)
X_test$v180[which(X_test$v180 == -4)] <- imputeDK(X_test$v180, X_test$v180_DK)
X_test$v181[which(X_test$v181 == -4)] <- imputeDK(X_test$v181, X_test$v181_DK)
X_test$v182[which(X_test$v182 == -4)] <- imputeDK(X_test$v182, X_test$v182_DK)
X_test$v183[which(X_test$v183 == -4)] <- imputeDK(X_test$v183, X_test$v183_DK)
X_test$v221[which(X_test$v221 == -4)] <- imputeDK(X_test$v221, X_test$v221_DK)
X_test$v222[which(X_test$v222 == -4)] <- imputeDK(X_test$v222, X_test$v222_DK)
X_test$v223[which(X_test$v223 == -4)] <- imputeDK(X_test$v223, X_test$v223_DK)
X_test$v224[which(X_test$v224 == -4)] <- imputeDK(X_test$v224, X_test$v224_DK)
X_train$v228b_r[which(is.na(X_train$v228b_r))] <- 428
X_train$v231b_r[which(is.na(X_train$v231b_r))] <- -1
X_train$v233b_r[which(is.na(X_train$v233b_r))] <- -1
X_train$v251b_r[which(is.na(X_train$v251b_r))] <- -1
X_test$v233b_r[which(is.na(X_test$v233b_r))] <- 616

Xtrainfilter <- X_train[,!grepl("DK|DE|f",names(X_train))]
Xtestfilter <- X_test[,!grepl("DK|DE|f",names(X_test))]
head(X_train[,grepl("DK|DE|f",names(X_train))])
XtrainUA <- Xtrainfilter
XtestUA <- Xtestfilter
head(XtrainUA[,c(5:259, 261:263, 265:266, 268:306, 308:376, 379:390, 392:393)])

XtrainUAset <- XtrainUA[, c(5:259, 261:263, 265:266, 268:306,
                            308:376, 379:390, 392:393)]
XtestUAset <- XtestUA[, c(5:259, 261:263, 265:266, 268:306,
                          308:376, 379:390, 392:393)]

y_train$label <- as.numeric(y_train$label)
ytraintest <- y_train
ytraintest[which(ytraintest[,2] == -1),2] <- 0
ytt <- ytraintest[XtrainUA$id + 1,]

dtrain <- xgb.DMatrix(data = as.matrix(XtrainUAset), label = ytt$label)
sampleLabel <- floor(rnorm(dim(XtestUA)[1],mean(ytt$label)))
sampleLabel[which(sampleLabel < 0)] <- 0
sampleLabel[which(sampleLabel > 4)] <- 4
dtest <- xgb.DMatrix(data = as.matrix(XtestUAset))
watchlist <- list(train=dtrain)

params <- list(max_depth = 6, eta = 0.3)
bst <- xgb.train(params = params, data=dtrain, watchlist = watchlist, nrounds = 500, 
                 objective="multi:softprob", num_class = 5, save_period = NULL)

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
important_features <- importance_matrix %>% 
  filter(Gain >= 0.02)
important_feature_names <- important_features$Feature

important_feature_names

paramsBayes <- list(max_depth = 9,	min_child_weight = 15,
                    subsample = 1, colsample_bytree = 0.2868571,
                    eta = 0.0296193, objective = "multi:softprob", num_class = 5)

XtrainSelect <- Xtrainfilter[important_feature_names]
XtestSelect <- Xtestfilter[important_feature_names]
dtrainSelect <- xgb.DMatrix(data = as.matrix(XtrainSelect), label = ytt$label)
dtestSelect <- xgb.DMatrix(data = as.matrix(XtestSelect))
watchlistBayes <- list(train=dtrainSelect)



xgbBayes <- xgb.train(params = paramsBayes, data = dtrainSelect, 
                      watchlist = watchlistBayes, nrounds = 500)

pred <- predict(xgbBayes, dtestSelect)

xgb.cv(params = paramsBayes, data=dtrainSelect, watchlist = watchlistBayes, nrounds = 500, nfold = 10)

a <- 1:length(pred)
res <- data.frame(id=c(0:11437),"no answer" = pred[seq(1, length(a),5)],
                  "very important" = pred[seq(2, length(a),5)],
                  "quite important" = pred[seq(3, length(a),5)],
                  "not important" = pred[seq(4, length(a),5)],
                  "not at all important" = pred[seq(5, length(a),5)])
write.csv(res, file="stat441proj.csv", row.names=FALSE)

