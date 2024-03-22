X_train <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/X_train.csv", stringsAsFactors=TRUE)
X_test <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/X_test.csv", stringsAsFactors=TRUE)
y_train <- read.csv("C:/Users/bruce/Downloads/w2024-kaggle-contest/y_train.csv")

library(data.table)
library(xgboost)
library(tidyverse)
set.seed(20843361)
Xtrainfilter <- X_train[,!grepl("DE|f",names(X_train))]
Xtestfilter <- X_test[,!grepl("DE|f",names(X_test))]
head(X_train[,grepl("DE|f",names(X_train))])
head(XtrainUA[,5:264])
choice <- 1:10000
XtrainUA <- Xtrainfilter#[choice,]
XtestUA <- Xtestfilter

XtrainUAset <- XtrainUA[, c(5:264)]
XtestUAset <- XtestUA[, c(5:264)]

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

params <- list(max_depth = 3, eta = 0.01)
bst <- xgb.train(params = params, data=dtrain, watchlist = watchlist, nrounds = 500, 
                 objective="multi:softprob", num_class = 5)
pred <- predict(bst, dtest)

xgb.cv(params = params, data=dtrain, watchlist = watchlist, nrounds = 500, nfold = 10,
       objective="multi:softprob", num_class = 5)

a <- 1:length(pred)
res <- data.frame(id=c(0:11437),"no answer" = pred[seq(1, length(a),5)],
                  "very important" = pred[seq(2, length(a),5)],
                  "quite important" = pred[seq(3, length(a),5)],
                  "not important" = pred[seq(4, length(a),5)],
                  "not at all important" = pred[seq(5, length(a),5)])
write.csv(res, file="stat441proj.csv", row.names=FALSE)
