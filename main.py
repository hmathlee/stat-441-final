import pandas as pd
from utils import data_split
from models import XGBoost, Logistic

X_TRAIN_PATH = "data/X_train.csv"
Y_TRAIN_PATH = "data/y_train.csv"

X_TEST_PATH = "data/X_test.csv"

VAL_SIZE = 0.2
RANDOM_STATE = 420

if __name__ == "__main__":
    # 80% of the data is training; to get (VAL_SIZE * 100)% validation data, divide by 0.8
    val_sample = VAL_SIZE / 0.8
    datasets = data_split(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, val_sample, RANDOM_STATE)

    val_metrics = {}

    models = [Logistic(), XGBoost()]
    enc = True
    for model in models:
        # Dynamically generate the required version of the dataset: reduce runtime
        # Generate the one-hot version of the datasets
        if model.needs_one_hot and not enc:
            datasets = data_split(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, val_sample, RANDOM_STATE, True)
        # Generate the categorical version of the datasets
        elif not model.needs_one_hot and enc:
            datasets = data_split(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, val_sample, RANDOM_STATE, False)
        enc = model.needs_one_hot

        # Train each model and report validation metrics
        X_train, y_train = datasets["train"]
        X_val, y_val = datasets["val"]
        val_metrics[model.model] = model.train(X_train, y_train, X_val, y_val)

    print(val_metrics)
