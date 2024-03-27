library(xgboost)
library(tidyverse)
library(caret)
library(Matrix)
library(dplyr)
library(rBayesianOptimization)

# train data
X_train = read.csv("X_train.csv", stringsAsFactors=TRUE)
y_train = read.csv("y_train.csv", stringsAsFactors=TRUE)
X_train = X_train[7:438]
X_train <- X_train %>%
  mutate(across(where(is.factor), as.numeric))
y_train$label[y_train$label == -1] = 0
label = y_train$label
train = cbind(label, X_train)
train = na.omit(train)
train_scaled <- scale(train[,-1])
# test data
X_test = read.csv("X_test.csv", stringsAsFactors=TRUE)
X_test = X_test[7:438]
test <- X_test %>%
  mutate(across(where(is.factor), as.numeric))

# base model
dtrain <- xgb.DMatrix(data = as.matrix(train_scaled[,-1]), 
                      label = train$label)
set.seed(123)
model <- xgb.train(data = dtrain, objective = "multi:softprob", 
                   nrounds = 500, num_class=5)
importance_matrix <- xgb.importance(model = model)
print(importance_matrix)
important_features <- importance_matrix %>% 
  filter(Gain >= 0.002)
important_feature_names <- important_features$Feature

# selected data
train_selected = train[c("label", important_feature_names)]
#set.seed(123)
#samples <- sample(nrow(train_selected), 5000)
#X_sample <- train_selected[samples,]
dtrain_selected <- xgb.DMatrix(data = as.matrix(train_selected[,-1]), 
                               label = train_selected$label)

# cv 
xgb_fun <- function(max_depth, min_child_weight, subsample, colsample_bytree, eta) {
  params <- list(
    booster = "gbtree",
    objective = "multi:softprob",
    num_class = 5,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    eta = eta,
    eval_metric = "mlogloss"
  )
  
  cv.nround = 500
  cv.nfold = 10
  set.seed(123)
  
  cv <- xgb.cv(params = params, data = dtrain_selected, 
               nrounds = cv.nround, nfold = cv.nfold, 
               showsd = TRUE, stratified = TRUE, verbose = FALSE)
  
  list(Score = -min(cv$evaluation_log$test_mlogloss_mean), 
       Pred = cv$evaluation_log$test_mlogloss_mean)
}
bounds <- list(max_depth = c(8L, 10L),
               min_child_weight = c(14L, 15L),
               subsample = c(0.95, 1),
               colsample_bytree = c(0.1, 0.3),
               eta = c(0.02, 0.03))
set.seed(123)
bayes_opt <- BayesianOptimization(xgb_fun, verbose = 1,
                                  bounds = bounds,
                                  init_points = 5,
                                  n_iter = 25, acq = "ucb")
print(bayes_opt$Best_Par)

# Best Parameters Found: 
# Round = 22	max_depth = 9.0000	min_child_weight = 15.0000	subsample = 1.0000	
# colsample_bytree = 0.2868571	eta = 0.0296193	Value = -0.8325565

best_para = bayes_opt$Best_Par
best = list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = 5,
  eval_metric = "mlogloss",
  max_depth = best_para["max_depth"],
  min_child_weight = best_para["min_child_weight"],
  subsample = best_para["subsample"],
  colsample_bytree = best_para["colsample_bytree"],
  eta = best_para["eta"]
)
xgb_model = xgb.train(params = best, data = dtrain_selected, nrounds = 500)
dtest_selected = test[important_feature_names]
dtest = xgb.DMatrix(data = as.matrix(dtest_selected))
predictions_prob = predict(xgb_model, dtest)
predictions_matrix = matrix(predictions_prob, ncol = 5, byrow = TRUE)
head(predictions_matrix)

# final output
result = data.frame(id = 0:(nrow(predictions_matrix) - 1))
result = cbind(result, predictions_matrix)
colnames(result) = c("id", "no answer", 
                      "very important", "quite important", 
                      "not important", "not at all important")
write.csv(result, "prob.csv", row.names = FALSE)
``
