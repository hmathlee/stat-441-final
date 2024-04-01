import os
from yaml import safe_load
from datetime import datetime
from time import time

import numpy as np
import matplotlib.pyplot as plt

from utils import read_csv_data, preprocess, select_features, write_pred_csv, pandas2numpy, report_validation_metrics
from legacy.utils import pandas2torch

from models import XGBoost, ModelStack, RandomForest, KNearest, NeuralNetwork

# Config
cfg = safe_load(open("config.yaml", "r"))
model_out = cfg["MODEL_OUT"]


if __name__ == "__main__":
    start = time()

    # Read data
    X_train, y_train, X_test = read_csv_data(cfg["X_TRAIN"], cfg["Y_TRAIN"], cfg["X_TEST"])

    # Get datetime for saving preprocessed datasets
    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")
    preproc_dir = os.path.join("preprocessed", datetime.now().strftime("%X").replace(":", "_"))
    X_train, y_train, X_val, y_val, X_test = preprocess(X_train, y_train, X_test, preproc_dir)

    # Model dir
    if not os.path.exists(model_out):
        os.makedirs(model_out)

    # Confusion matrix dir
    if not os.path.exists("confusion"):
        os.makedirs("confusion")

    # XGBoost for feature selection
    xgb_select = XGBoost(cfg["XGBOOST"]["SELECTION"])
    print("Fitting XGBoost (selection). {} features".format(len(X_train.columns)))
    xgb_select_metrics = xgb_select.train(X_train, y_train, X_val, y_val, pre_test=cfg["PRE_TEST"])
    print("XGBoost log loss:", xgb_select_metrics["logloss"])
    print("XGBoost accuracy:", xgb_select_metrics["accuracy"])

    # If we did a pre-test
    if cfg["PRE_TEST"] > 0:
        print("XGBoost log loss (pre-test):", xgb_select_metrics["pre_test_logloss"])
        print("XGBoost accuracy (pre-test):", xgb_select_metrics["pre_test_accuracy"])

    # Prediction
    print("Predicting with XGBoost...")
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    pred_df = xgb_select.predict(X_test)
    pred_dir = os.path.join("predictions", datetime.now().strftime("%X").replace(":", "_"))
    write_pred_csv(pred_df, pred_dir)

    # Select important features from XGBoost
    X_train, X_val, X_test = select_features(X_train, X_val, X_test, xgb_select.trained)

    # XGBoost for classification
    xgb_predict = XGBoost(cfg["XGBOOST"]["PREDICTION"])
    print("Fitting XGBoost (prediction). {} features".format(len(X_train.columns)))
    xgb_predict_metrics = xgb_predict.train(X_train, y_train, X_val, y_val, pre_test=cfg["PRE_TEST"])
    print("XGBoost log loss:", xgb_predict_metrics["logloss"])
    print("XGBoost accuracy:", xgb_predict_metrics["accuracy"])

    if not os.path.exists("preprocessed_numpy"):
        os.makedirs("preprocessed_numpy")
    preproc_dir_np = os.path.join("preprocessed_numpy", datetime.now().strftime("%X").replace(":", "_"))
    X_train_np, X_val_np, X_test_np = pandas2numpy(X_train, X_val, X_test, preproc_dir=None)
    X_train_np, X_val_np = X_train_np.astype(float), X_val_np.astype(float)

    # Random forest
    rf = RandomForest()
    print("Fitting random forest (prediction). {} features".format(X_train_np.shape[1]))
    rf_metrics = rf.train(X_train_np, y_train, X_val_np, y_val)
    print("Random forest log loss:", rf_metrics["logloss"])
    print("Random forest accuracy:", rf_metrics["accuracy"])

    # K-Nearest Neighbors
    knn = KNearest()
    print("Fitting k-nearest neighbors (prediction). {} features".format(X_train_np.shape[1]))
    knn_metrics = knn.train(X_train_np, y_train, X_val_np, y_val)
    print("KNN log loss:", knn_metrics["logloss"])
    print("KNN accuracy:", knn_metrics["accuracy"])

    # Stacking
    stack = ModelStack(prefit=False)
    X_train_np, X_val_np = X_train_np.astype(float), X_val_np.astype(float)
    print("Fitting stacked model (prediction). {} features".format(X_train_np.shape[1]))
    stack_metrics = stack.train(X_train_np, y_train, X_val_np, y_val)
    print("Stacked log loss:", stack_metrics["logloss"])
    print("Stacked accuracy:", stack_metrics["accuracy"])

    end = time()
    print("Runtime: {}".format(end - start))

    # Check if stacking model load works
    stack_from_pickle = ModelStack(from_pickle=True)
    val_metrics = report_validation_metrics(stack_from_pickle, X_val_np, y_val)
    print("Stacked log loss (from load):", val_metrics["logloss"])
    print("Stacked accuracy (from load):", val_metrics["accuracy"])

    # Prediction
    print("Predicting with model stacking...")
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    pred_df = stack_from_pickle.predict(X_test_np)
    pred_dir = os.path.join("predictions", datetime.now().strftime("%X").replace(":", "_"))
    write_pred_csv(pred_df, pred_dir)

    # Neural network training
    nn = NeuralNetwork()
    print("Fitting MLP (prediction). {} features".format(len(X_train.columns)))
    dtrain = pandas2torch(X_train, y_train)
    dval = pandas2torch(X_val, y_val)
    nn_metrics = nn.train(X_train, y_train, X_val, y_val, return_history=True)
    print("MLP log loss:", nn_metrics["logloss"])
    print("MLP accuracy:", nn_metrics["accuracy"])

    # Plot neural net train/val loss curves
    t = np.arange(nn.epochs) + 1
    plt.plot(t, nn_metrics["loss_history"]["train"], label="train loss")
    plt.plot(t, nn_metrics["loss_history"]["val"], label="val loss")

    # Plot XGBoost loss for reference
    xgb_loss = np.repeat(xgb_select_metrics["logloss"], nn.epochs)
    plt.plot(t, xgb_loss, label="XGBoost loss", linestyle="dashed")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Log Loss (Train/Val)")
    plt.savefig("nn_loss.png")
