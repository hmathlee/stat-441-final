import os
from yaml import safe_load
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from utils import read_csv_data, preprocess, select_features, write_pred_csv, pandas2numpy, pandas2torch
from models import XGBoost, RandomForest, NeuralNetwork


# Config
cfg = safe_load(open("config.yaml", "r"))


if __name__ == "__main__":
    # Read data
    X_train, y_train, X_test, val_idx = read_csv_data(cfg["X_TRAIN"], cfg["Y_TRAIN"], cfg["X_TEST"], cfg["VAL"])

    # Get datetime for saving preprocessed datasets
    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")
    preproc_dir = os.path.join("preprocessed", datetime.now().strftime("%X").replace(":", "_"))
    X_train, y_train, X_val, y_val, X_test = preprocess(X_train, y_train, X_test, val_idx, preproc_dir)

    # XGBoost for feature selection
    xgb_select = XGBoost()
    print("Fitting XGBoost (selection). {} features".format(len(X_train.columns)))
    xgb_select_metrics = xgb_select.train(X_train, y_train, X_val, y_val, cfg["XGBOOST"]["SELECTION"], cv=False,
                                          pre_test=cfg["PRE_TEST"])
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

    # Random forest
    rf = RandomForest()
    print("Fitting random forest (prediction). {} features".format(len(X_train.columns)))
    X_train_np, X_val_np, X_test_np = pandas2numpy(X_train, X_val, X_test)
    rf_metrics = rf.train(X_train_np, y_train, X_val_np, y_val)
    print("Random forest log loss:", rf_metrics["logloss"])
    print("Random forest accuracy:", rf_metrics["accuracy"])

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
