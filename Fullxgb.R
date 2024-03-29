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
# test data
X_test = read.csv("X_test.csv", stringsAsFactors=TRUE)
X_test = X_test[7:438]
test <- X_test %>%
  mutate(across(where(is.factor), as.numeric))
# weight
num_samples <- length(train$label)
num_classes <- 5
### two weight method
class_weights <- table(train$label) / num_samples
# class_weights <- length(train$label) / (5 * table(train$label))
sample_weights <- sapply(train$label, function(x) class_weights[x+1])

# dtrain
dtrain <- xgb.DMatrix(data = as.matrix(train[,-1]), 
                      label = train$label, 
                      weight = sample_weights
                      )

# select
set.seed(123)
model <- xgb.train(data = dtrain, objective = "multi:softprob", 
                   nrounds = 500, num_class=5, 
                   tree_method = "hist", device = "cuda")
importance_matrix <- xgb.importance(model = model)
print(importance_matrix)
important_features <- importance_matrix %>% 
  filter(Gain >= 0.002)
important_feature_names <- important_features$Feature
sum(important_features$Gain)

# selected data
train_selected = train[c("label", important_feature_names)]
dtrain_selected <- xgb.DMatrix(data = as.matrix(train_selected[,-1]), 
                               label = train_selected$label, 
                               weight = sample_weights
                               )

# cv 
xgb_fun <- function(max_depth, min_child_weight, subsample, colsample_bytree, eta, gamma) {
  params <- list(
    booster = "gbtree",
    objective = "multi:softprob",
    num_class = 5,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    eta = eta,
    gamma = gamma,
    eval_metric = "mlogloss"
  )
  
  cv.nround = 500
  cv.nfold = 5
  set.seed(123)
  
  cv <- xgb.cv(params = params, data = dtrain_selected, 
               nrounds = cv.nround, nfold = cv.nfold, 
               showsd = TRUE, stratified = TRUE, 
               tree_method = "hist", device = "cuda", 
               verbose = 0)
  
  list(Score = -min(cv$evaluation_log$test_mlogloss_mean))
}
bounds <- list(max_depth = c(3L, 5L),
               min_child_weight = c(1L, 20L),
               subsample = c(0.5, 1),
               colsample_bytree = c(0.2, 1),
               eta = c(0.01, 0.3),
               gamma = c(0, 2))
set.seed(123)
bayes_opt <- BayesianOptimization(xgb_fun, verbose = 1,
                                  bounds = bounds,
                                  init_points = 5,
                                  n_iter = 15, acq = "ucb")
print(bayes_opt$Best_Par)

### best
# Best Parameters Found: 
# max_depth = 6.0000	min_child_weight = 28.0000	subsample = 1.0000	
# colsample_bytree = 0.2720138	eta = 0.03492437	gamma = 0.0412249	
# Value = -0.8283446

# max_depth = 7.0000	min_child_weight = 22.0000	subsample = 0.9580844	
# colsample_bytree = 0.2188373	eta = 0.03540979	gamma = 1.0000	
# Value = -0.8299262

# max_depth = 7.0000    min_child_weight = 26.0000    subsample = 0.9572633    
# colsample_bytree = 0.1655841    eta = 0.0398854    gamma = 0.8674792    
# Value = -0.8302205

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
  eta = best_para["eta"],
  gamma = best_para["gamma"]
)

try = list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = 5,
  eval_metric = "mlogloss",
  max_depth = 6,
  min_child_weight = 28,
  subsample = 1,
  colsample_bytree = 0.2720138,
  eta = 0.03492437,
  gamma = 0.0412249
)
xgb.cv(params = try, data = dtrain_selected, 
       nrounds = 500, nfold = 5, 
       tree_method = "hist", device = "cuda", 
       verbose = 1)

xgb_model = xgb.train(params = try, data = dtrain_selected, 
                      nrounds = 500, verbose = 1, 
                      tree_method = "hist", device = "cuda")
dtest_selected = test[important_feature_names]
dtest_selected = xgb.DMatrix(data = as.matrix(dtest_selected))
dtest = xgb.DMatrix(data = as.matrix(test))

# pred
pred_prob = predict(xgb_model, dtest_selected)
pred_matrix = matrix(pred_prob, ncol = 5, byrow = TRUE)
head(pred_matrix)
pred_class <- max.col(pred_matrix)-1
hist(pred_class, col = "lightblue", labels = TRUE)
hist(y_train$label, labels = TRUE)

#
pred_vs_true <- cbind(pred_class, train$label)
pred_for_0 <- pred_vs_true[pred_vs_true[, 2] == 0]
plot(hist(pred_for_0))


# final output
result = data.frame(id = 0:(nrow(pred_matrix) - 1))
result = cbind(result, pred_matrix)
colnames(result) = c("id", "no answer", 
                      "very important", "quite important", 
                      "not important", "not at all important")
write.csv(result, "-0.8283446.csv", row.names = FALSE)

