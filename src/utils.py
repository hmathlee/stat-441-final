# Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# File utils
import os
from yaml import safe_load
from joblib import dump

from collections import Counter
import random

cfg = safe_load(open("config.yaml", 'r'))
model_out = cfg["MODEL_OUT"]

# Set seed
rng = np.random.default_rng(cfg["SEED"])
random.seed(cfg["SEED"])


def stratified_train_val_split(X, y):
    full_df = pd.concat([X, y], axis=1)
    full_df = full_df.reset_index(drop=True)
    dtrain, dval = None, None
    labels = y.unique()

    for label in labels:
        df_with_label = full_df[full_df["label"] == label]
        train, val = train_test_split(df_with_label, test_size=int(len(df_with_label) * cfg["VAL"]),
                                      random_state=cfg["SEED"])
        dtrain = train if dtrain is None else pd.concat([dtrain, train])
        dval = val if dval is None else pd.concat([dval, val])

    X_train = dtrain.drop(columns=["label"])
    X_val = dval.drop(columns=["label"])

    y_train = dtrain["label"]
    y_val = dval["label"]

    return X_train, y_train, X_val, y_val


def read_csv_data(X_train_path, y_train_path, X_test_path):
    # Read in the dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)["label"]

    # Remove any rows from the train set that contain NA values
    X_train = X_train.dropna(axis=0)
    y_train = y_train.iloc[X_train.index]

    return X_train, y_train, X_test


def train_val_split(X_train, y_train, val_idx):
    X_train = X_train.reset_index(drop=True)

    v = X_train.index.isin(val_idx)

    X_val = X_train[v]
    y_val = y_train[v]

    X_train = X_train[~v]
    y_train = y_train[~v]

    return X_train, y_train, X_val, y_val


def eliminate_variables(X, group_start, group_end, new_ftr_name):
    idx = list(X.columns).index(group_start)
    curr_var = group_start
    to_drop = []
    while curr_var != group_end:
        to_drop.append(curr_var)
        idx += 1
        curr_var = list(X.columns)[idx]
    if new_ftr_name:
        X[new_ftr_name] = X[to_drop].sum(axis=1)
    X = X.drop(columns=to_drop)
    return X


def zero_out_negatives(X, ftr_name):
    vals = X[ftr_name].unique()
    for v in vals:
        if v < 0:
            X[ftr_name] = X[ftr_name].replace(v, 0)
    return X


def feature_transform(X):
    # Rewrite fieldwork start and end dates as numbers
    fw_start = X["fw_start"].apply(lambda x: x // 100 + (x % 100) / 12)
    fw_end = X["fw_end"].apply(lambda x: x // 100 + (x % 100) / 12)

    # Replace field work start and end with duration
    fw_duration = fw_end - fw_start
    X = X.assign(fw_duration=fw_duration)
    X = X.drop(columns=["fw_start", "fw_end"])
    X["fw_duration"] = X["fw_duration"].astype(float)

    return X


def drop(X):
    # Drop the id and country columns (redundant information)
    vars_to_drop = ["id", "country"]
    for var in X.columns:
        vals = X[var].unique()

        # Remove variables with only one unique value
        if len(vals) == 1 or X[var].dtype != "O" and len(vals[vals >= 0]) == 1:
            vars_to_drop.append(var)

        # Drop modified and/or flagged variables
        elif "DE" in var:
            vars_to_drop.append(var)
        elif "f" in var and "fw" not in var:
            vars_to_drop.append(var)

    X = X.drop(columns=vars_to_drop)
    return X


def preprocess(X_train, y_train, X_test, preproc_dir=None):
    # Replace -1's in y_train with 0's
    y_train = y_train.replace(-1, 0)

    # The drop() function removes variables with only one unique value; this is checked across all observations,
    # i.e. both train and test sets
    print("Removing constant, modified, and flagged variables...")
    X = pd.concat([X_train, X_test])
    X = drop(X)

    # Retrieve the train and test sets
    X_train = X.iloc[:len(X_train)]
    X_test = X.iloc[len(X_train):]

    # Do some feature engineering
    print("Applying feature engineering...")
    X_train = feature_transform(X_train)
    X_test = feature_transform(X_test)

    # Handle the categorical variables for XGBoost
    print("Ensuring categorical variable types...")
    X_train = categorical_dmatrix(X_train)
    X_test = categorical_dmatrix(X_test)

    # Train-val split
    print("Generating a train-val data split...")
    X_train, y_train, X_val, y_val = stratified_train_val_split(X_train, y_train)

    # Write the dfs to .csv files if applicable
    if len(os.listdir("preprocessed")) == 0 and preproc_dir:
        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)

            X_train_path = os.path.join(preproc_dir, "X_train.csv")
            X_val_path = os.path.join(preproc_dir, "X_val.csv")
            X_test_path = os.path.join(preproc_dir, "X_test.csv")

            X_train.to_csv(X_train_path, index=False)
            X_val.to_csv(X_val_path, index=False)
            X_test.to_csv(X_test_path, index=False)

            print("Preprocessed datasets saved to")
            print("-- X_train: {}".format(X_train_path))
            print("-- X_val: {}".format(X_val_path))
            print("-- X_test: {}".format(X_test_path))

    return X_train, y_train, X_val, y_val, X_test


def categorical_dmatrix(X):
    for var in X.columns:
        if X[var].dtype == "O":
            X[var] = X[var].astype("category")
        elif X[var].dtype == "bool":
            X[var] = X[var].astype(int)
    return X


def select_features(X_train, X_val, X_test, xgb_model):
    # Get feature importance from XGBoost model (descending order)
    imp = xgb_model.feature_importances_
    ftr_idx = np.where(imp > cfg["XGBOOST"]["SELECTION"]["IMP_THRESH"])

    print("Selecting features based on gain...")
    ftr_select = xgb_model.feature_names_in_[ftr_idx]
    scores = xgb_model.feature_importances_[ftr_idx]

    df_rows = [[ftr, score] for ftr, score in zip(list(ftr_select), list(scores))]

    importance_df = pd.DataFrame(df_rows, columns=["feature_name", "importance_gain"])
    importance_df.to_csv("importance_matrix.csv")

    ftr_select = list(ftr_select)
    print("{} features selected from XGBoost.".format(len(ftr_select)))

    # Feature selection
    X_train = X_train[ftr_select]
    X_val = X_val[ftr_select]
    X_test = X_test[ftr_select]

    return X_train, X_val, X_test


def write_pred_csv(pred_df, dest):
    assert not os.path.exists(dest), "The prediction directory {} already exists.".format(dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
        pred_path = os.path.join(dest, "pred.csv")
        pred_df.to_csv(pred_path, index_label="id")


def pandas2numpy(X_train, X_val, X_test, preproc_dir):
    # label-encode
    X = pd.concat([X_train, X_val, X_test])
    le = LabelEncoder()
    for var in X.columns:
        if var == "fw_duration":
            break
        if X[var].dtype in ["category", "object"]:
            X[var] = le.fit_transform(X[var])

    n_train, n_val = len(X_train), len(X_val)

    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]

    if len(os.listdir("preprocessed_numpy")) == 0 and preproc_dir:
        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)

            X_train_path = os.path.join(preproc_dir, "X_train.csv")
            X_val_path = os.path.join(preproc_dir, "X_val.csv")
            X_test_path = os.path.join(preproc_dir, "X_test.csv")

            X_train.to_csv(X_train_path, index=False)
            X_val.to_csv(X_val_path, index=False)
            X_test.to_csv(X_test_path, index=False)

    return np.array(X_train), np.array(X_val), np.array(X_test)


# Convenience function for saving models_1
def save_model_as_joblib(model):
    model_path = os.path.join(model_out, model.model + ".pkl")
    dump(model.trained, model_path)


# Convenience function for saving confusion matrix .csv
def save_confusion_matrix_csv(y_true, y_pred, out_path):
    assert y_true.shape == y_pred.shape, \
        "y_true and y_pred are not the same shape; got {} and {}.".format(y_true.shape, y_pred.shape)
    y_true_s = pd.Series(y_true, name="Label")
    y_pred_s = pd.Series(y_pred, name="Prediction")
    df = pd.crosstab(y_true_s, y_pred_s)
    df.to_csv(out_path)


def report_validation_metrics(model, X_val, y_val):
    val_pred = model.trained.predict_proba(X_val)
    val_pred_class = np.argmax(val_pred, axis=1)
    conf_mat_out = os.path.join("confusion", model.model + "_confusion_matrix.csv")
    save_confusion_matrix_csv(y_val, val_pred_class, conf_mat_out)

    return {
        "logloss": log_loss(y_val, val_pred),
        "accuracy": accuracy_score(y_val, val_pred_class)
    }


def preprocess_xgboost(X_train, y_train, X_test, val_idx, preproc_dir=None):
    # Replace -1's in y_train with 0's
    y_train = y_train.replace(-1, 0)

    # The drop() function removes variables with only one unique value; this is checked across all observations,
    # i.e. both train and test sets
    print("Removing constant, modified, and flagged variables...")
    X = pd.concat([X_train, X_test])
    X = drop(X)

    dk_vars = ["v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v221", "v222", "v223", "v224"]
    for var in dk_vars:
        dk = X[var + "_DK"]
        dk_response_idx = (dk != -4)
        X[var][dk_response_idx] = dk[dk_response_idx]

    X = X.drop(columns=[var + "_DK" for var in dk_vars])

    # Retrieve the train and test sets
    X_train = X.iloc[:len(X_train)]
    X_test = X.iloc[len(X_train):]

    # Train-val split
    print("Generating a train-val data split...")
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_idx)

    # Write the dfs to .csv files if applicable
    if len(os.listdir("preprocessed")) == 0 and preproc_dir:
        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)

            X_train_path = os.path.join(preproc_dir, "X_train.csv")
            X_val_path = os.path.join(preproc_dir, "X_val.csv")
            X_test_path = os.path.join(preproc_dir, "X_test.csv")

            X_train.to_csv(X_train_path, index=False)
            X_val.to_csv(X_val_path, index=False)
            X_test.to_csv(X_test_path, index=False)

            print("Preprocessed datasets saved to")
            print("-- X_train: {}".format(X_train_path))
            print("-- X_val: {}".format(X_val_path))
            print("-- X_test: {}".format(X_test_path))

    return X_train, y_train, X_val, y_val, X_test


def undersample_majority_class(X, y):
    y_counts = Counter(y).items()
    y_counts = sorted(y_counts, key=lambda x: x[1])
    min_count_label, min_count = y_counts[0]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    full_df = pd.concat([X, y], axis=1)
    for label, count in y_counts[1:]:
        curr_label = full_df[full_df["label"] == label]
        others = full_df[full_df["label"] != label]
        curr_label_keep = curr_label.sample(frac=min_count/count)
        full_df = pd.concat([others, curr_label_keep])

    new_y_counts = sorted(Counter(full_df["label"]).items())
    print(new_y_counts)

    return full_df.drop(columns=["label"]), full_df["label"]
