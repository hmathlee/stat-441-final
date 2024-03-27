import pandas as pd
from yaml import safe_load

import numpy as np

cfg = safe_load(open("../config.yaml", 'r'))

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
    fw_duration = fw_start - fw_end
    X = X.assign(fw_duration=fw_duration)
    X = X.drop(columns=["fw_start", "fw_end"])

    # Aggregate some integer-encoded categorical variables (not of type "object")
    X = eliminate_variables(X, "v22", "v31", "neighbor_discrim")
    X = eliminate_variables(X, "v31", "v38", "distrust")
    X = eliminate_variables(X, "v164", "v171", "community")
    X = eliminate_variables(X, "v40", "v45a", "job_standard")
    X = eliminate_variables(X, "v46", "v51", "work_ethic")
    X = eliminate_variables(X, "v72", "v79", "patriarch_belief")
    X = eliminate_variables(X, "v83", "v96", "children_virtues")
    X = eliminate_variables(X, "v97", "v102", "poli_activism")
    X = eliminate_variables(X, "v115", "v132", "society_conf")
    X = eliminate_variables(X, "v133", "v144", "democracy")

    # Subtract each of the "none mentioned" variables from their relevant groups
    X = zero_out_negatives(X, "v45a")
    X["job_standard"] = X["job_standard"] - X["v45a"]
    X = X.drop(columns=["v45a"])

    X = zero_out_negatives(X, "v96")
    X["children_virtues"] = X["children_virtues"] - X["v96"]
    X = X.drop(columns=["v96"])

    X = X.drop(columns=["v108", "v109", "v110", "v111"])

    # Add/subtract DK (don't know) values based on topic: subtract if higher rating means individual perceives a
    # high level of corruption in their country; add otherwise
    to_subtract = ["v176", "v180", "v181"]
    to_add = ["v177", "v178", "v179", "v182", "v183"]
    corruption = X[to_add].sum(axis=1) - X[to_subtract].sum(axis=1)

    # Compute corruption values for both the original and "don't know" responses
    to_subtract = [var + "_DK" for var in to_subtract]
    to_add = [var + "_DK" for var in to_add]
    corruption_dk = X[to_add].sum(axis=1) - X[to_subtract].sum(axis=1)

    # Replace these variables with the corruption feature
    X = X.assign(corruption=(corruption + corruption_dk))
    X = X.drop(columns=to_add)
    X = X.drop(columns=to_subtract)
    
    # Select variables up to 224_DK, along with the new variables we selected
    # Specify the new feature name as None to not make a new aggregate variable
    X = eliminate_variables(X, "v225", "fw_duration", None)

    n_col_new = len(X.columns)
    print("n_features: {}".format(n_col_new))

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


def preprocess(X_train, y_train, X_test, val_idx):
    # Replace -1's in y_train with 0's
    y_train = y_train.replace(-1, 0)

    # The drop() function removes variables with only one unique value; this is checked across all observations,
    # i.e. both train and test sets
    X = pd.concat([X_train, X_test])
    X = drop(X)

    # Retrieve the train and test sets
    X_train = X.iloc[:len(X_train)]
    X_test = X.iloc[len(X_train):]

    # Do some feature engineering
    X_train = feature_transform(X_train)
    X_test = feature_transform(X_test)

    # Handle the categorical variables for XGBoost
    X_train = categorical_dmatrix(X_train)
    X_test = categorical_dmatrix(X_test)

    # Train-val split
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_idx)
    return X_train, y_train, X_val, y_val, X_test


def categorical_dmatrix(X):
    for var in X.columns:
        if X[var].dtype == "O":
            X[var] = X[var].astype("category")
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
    ftr_idx = np.where(imp >= 0.002)
    ftr_select = []

    # Get features with non-zero importance
    print("Selecting features based on gain...")
    for idx in ftr_idx:
        ftr_select.append(xgb_model.feature_names_in_[idx])

    print("{} features selected from XGBoost".format(len(ftr_select)))

    # Feature selection
    X_train = X_train[ftr_select]
    X_val = X_val[ftr_select]
    X_test = X_test[ftr_select]

    return X_train, X_val, X_test
