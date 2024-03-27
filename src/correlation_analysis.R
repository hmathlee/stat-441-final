# Read pre-processed train/test sets
train <- read.csv("path_to_preprocessed_X_train_csv", stringsAsFactors=TRUE)
test <- read.csv("path_to_preprocessed_X_test_csv", stringsAsFactors=TRUE)

# Drop id column from train and test
train <- train[, 2:ncol(train)]
test <- test[, 2:ncol(test)]

# Convert train/test to numeric so we can call cor() on them
train <- apply(train, 2, function(x){as.numeric(x)})
test <- apply(test, 2, function(x){as.numeric(x)})

# Correlation matrix
corrMat <- cor(train)

# Variate 2 was country abbreviation; remove it
corrMat <- corrMat[-c(2), -c(2)]

# Get the 25 most correlated variable pairs 
library(lares)
top_corrs <- corr_cross(as.data.frame(train), top=25)
top_corrs_data <- top_corrs$data
write.csv(top_corrs_data, file="top_25_correlations.csv")

