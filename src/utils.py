# Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

# File utils
import os
from yaml import safe_load
from joblib import dump, load

# Torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.nn.functional import normalize

cfg = safe_load(open("config.yaml", 'r'))
model_out = cfg["MODEL_OUT"]

# Set seed
rng = np.random.default_rng(cfg["SEED"])


def read_csv_data(X_train_path, y_train_path, X_test_path, val):
    # Read in the dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)["label"]

    # Remove any rows from the train set that contain NA values
    X_train = X_train.dropna(axis=0)
    y_train = y_train.iloc[X_train.index]

    # (val)% of all data set aside for validation
    n_val = int(val * (len(X_train) + len(X_test)))
    val_idx = rng.choice(np.arange(len(X_train)), n_val, replace=False)

    return X_train, y_train, X_test, list(val_idx)


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

    # Aggregate some integer-encoded categorical variables (not of type "object")
    X = eliminate_variables(X, "v22", "v31", "neighbor_discrim")
    X = eliminate_variables(X, "v31", "v38", "distrust")
    X = eliminate_variables(X, "v40", "v45a", "job_standard")
    X = eliminate_variables(X, "v46", "v51", "work_ethic")
    X = eliminate_variables(X, "v72", "v79", "patriarch_belief")
    X = eliminate_variables(X, "v83", "v96", "children_virtues")
    X = eliminate_variables(X, "v97", "v102", "poli_activism")
    X = eliminate_variables(X, "v115", "v133", "society_conf")
    X = eliminate_variables(X, "v133", "v144", "democracy")
    X = eliminate_variables(X, "v146", "v149", "poli_system")
    X = eliminate_variables(X, "v164", "v171", "community")

    # Subtract each of the "none mentioned" variables from their relevant groups
    X = zero_out_negatives(X, "v45a")
    X["job_standard"] = X["job_standard"] - X["v45a"]
    X = X.drop(columns=["v45a"])

    X = zero_out_negatives(X, "v96")
    X["children_virtues"] = X["children_virtues"] - X["v96"]
    X = X.drop(columns=["v96"])

    X = X.drop(columns=["v108", "v109", "v110", "v111"])

    dk_vars = ["v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183", "v221", "v222", "v223", "v224"]
    for var in dk_vars:
        dk = X[var + "_DK"]
        dk_response_idx = (dk != -4)
        X[var][dk_response_idx] = dk[dk_response_idx]

    X = X.drop(columns=[var + "_DK" for var in dk_vars])

    # Handle the continuous income variable separately
    income_continuous = X["v261_ppp"]
    X = X.drop(columns=["v261_ppp"])
    mean, std = income_continuous.mean(), income_continuous.std()
    income_continuous = (income_continuous - mean) / std
    X["v261_ppp"] = income_continuous

    # Add/subtract DK (don't know) values based on topic: subtract if higher rating means individual perceives a
    # high level of corruption in their country; add otherwise
    to_subtract = ["v176", "v180", "v181"]
    to_add = ["v177", "v178", "v179", "v182", "v183"]
    corruption = X[to_add].sum(axis=1) - X[to_subtract].sum(axis=1)

    # Replace these variables with the corruption feature
    # X = X.assign(corruption=(corruption + corruption_dk))
    X = X.assign(corruption=corruption)
    X = X.drop(columns=to_add)
    X = X.drop(columns=to_subtract)

    # Select variables up to 224_DK, along with the new variables we selected
    # Specify the new feature name as None to not make a new aggregate variable
    X = eliminate_variables(X, "v225", "fw_duration", None)

    return X


def drop(X):
    """
    Drop variables with zero variance and/or whose values are modified/flagged
    :param X: pd.DataFrame; original dataset.
    :return: pd.DataFrame, with columns dropped.
    """
    # Drop the id and country columns (redundant information)
    vars_to_drop = ["country"]
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


def preprocess(X_train, y_train, X_test, val_idx, preproc_dir=None):
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


def categorical_dmatrix(X):
    for var in X.columns:
        if X[var].dtype == "O":
            X[var] = X[var].astype("category")
        elif X[var].dtype == "bool":
            X[var] = X[var].astype(int)
    return X


def select_features(X_train, X_val, X_test, xgb_model):
    """
    Apply a feature selection transformation (based on importance matrix from XGBoost) on train dnd test sets.
    :param X_train: pd.DataFrame; train set
    :param X_test: pd.DataFrame; test set
    :param xgb_model: XGBoost Booster model.
    :return: (pd.DataFrame, pd.DataFrame) (with only the selected features from XGBoost)
    """
    # Get feature importance from XGBoost model (descending order)
    imp = xgb_model.feature_importances_
    ftr_idx = np.where(imp > cfg["XGBOOST"]["SELECTION"]["IMP_THRESH"])
    ftr_select = []

    # Get features with non-zero importance
    print("Selecting features based on gain...")
    for idx in ftr_idx:
        ftr_select.append(xgb_model.feature_names_in_[idx])

    ftr_select = list(ftr_select[0])
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


def pandas2numpy(X_train, X_val, X_test):
    # One-hot-encode
    X = pd.concat([X_train, X_val, X_test])
    X = pd.get_dummies(X)

    n_train, n_val = len(X_train), len(X_val)

    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]

    return np.array(X_train), np.array(X_val), np.array(X_test)


# Convenience function for saving models
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


