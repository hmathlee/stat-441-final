import pandas as pd
from yaml import safe_load

import numpy as np
from sklearn.linear_model import Lasso

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch import nn

cfg = safe_load(open("config.yaml", 'r'))

# Set seed
rng = np.random.default_rng(cfg["SEED"])


def read_csv_data(X_train_path, y_train_path, X_test_path, test_size):
    # Read in the dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)["label"]

    # Remove any rows from the dataset that contain NA values
    X_train = X_train.dropna(axis=0)
    X_test = X_test.dropna(axis=0)
    y_train = y_train.iloc[X_train.index]

    n = len(X_train)
    val_idx = rng.choice(np.arange(n), int(test_size * n), replace=False)

    return X_train, y_train, X_test, list(val_idx)


def df_copy(dfs):
    return [df.copy() for df in dfs]


def train_val_split(X_train, y_train, X_test, val_idx):
    X_train = X_train.reset_index(drop=True)

    v = X_train.index.isin(val_idx)

    X_val = X_train[v]
    y_val = y_train[v]

    X_train = X_train[~v]
    y_train = y_train[~v]

    return X_train, y_train, X_val, y_val, X_test


def feature_transform(X):
    # Rewrite fieldwork start and end dates as numbers
    X["fw_start"] = X["fw_start"].apply(lambda x: x // 100 + (x % 100) / 12)
    X["fw_end"] = X["fw_end"].apply(lambda x: x // 100 + (x % 100) / 12)

    # For integer-encoded categorical variables, map k distinct values to {0, ..., k}
    for var in X.columns:
        if X[var].dtype == "integer":
            unique_vals = X[var].unique()
            map_vals = dict([(i, v) for i, v in enumerate(unique_vals)])
            X[var] = X[var].map(map_vals)

    return X


def drop_and_scale(X, scale=True):
    # Remove variables with only one unique value
    vars_to_drop = ["country"]
    for var in X.columns:
        if X[var].nunique() == 1:
            vars_to_drop.append(var)
    X = X.drop(columns=vars_to_drop)

    # Variable scaling
    if scale:
        for var in X.columns:
            if X[var].dtype != "O":
                X[var] = (X[var] - X[var].mean()) / X[var].std()

    return X


def preprocess(X_train, y_train, X_test, val_idx, one_hot_encode=True, feature_select=True, scale=True):
    # Replace -1 in y_train with 0
    y_train = y_train.replace(-1, 0)

    X_train = feature_transform(X_train)
    X_test = feature_transform(X_test)

    if one_hot_encode:
        # One-hot encode X-variables (need to combine X_train and X_test to get all possible categories)
        X = pd.concat([X_train, X_test])
        X = pd.get_dummies(X)
        X_train = X.iloc[:len(X_train)]
        X_test = X.iloc[len(X_train):]

    X_train = drop_and_scale(X_train, scale=scale)
    X_test = drop_and_scale(X_test, scale=scale)

    if feature_select:
        print("Running lasso for feature selection")
        lasso_feature_select = select_features(X_train, y_train)

        # Reduce dimensionality
        X_train = X_train.iloc[:, lasso_feature_select]
        X_test = X_test.iloc[:, lasso_feature_select]

    X_train, y_train, X_val, y_val, X_test = train_val_split(X_train, y_train, X_test, val_idx)

    if not one_hot_encode:
        # Enable categorical
        X_train = categorical_dmatrix(X_train)
        X_val = categorical_dmatrix(X_val)
        X_test = categorical_dmatrix(X_test)

    return X_train, y_train, X_val, y_val, X_test


def categorical_dmatrix(X):
    for var in X.columns:
        if X[var].dtype == "O":
            X[var] = X[var].astype("category")
    return X


def select_features(X, y):
    """
    Selects features from X-matrix using lasso regression; returns indices of nonzero coefficients.
        :param X: matrix of input variables.
        :param y: response vector.
        :param alpha: lasso regression penalization parameter.
        :return: np.array (indices of nonzero coefficients).
    """
    # Lasso regression
    lasso = Lasso(alpha=cfg["LASSO"]["ALPHA"])
    lasso.fit(X, y)
    return list(lasso.coef_.nonzero()[0])


def pandas2torch(X, y):
    # Cast booleans in X to ints (for numpy conversion)
    for var in X.columns:
        if X[var].dtype == "bool":
            X[var] = X[var].astype(int)
    X_tensor = Tensor(X.to_numpy())
    y_tensor = Tensor(y.to_numpy())
    return TensorDataset(X_tensor, y_tensor)


def L2_regularize(params, const):
    L2 = 0
    loss_fn = nn.MSELoss(reduction="sum")
    for param in params:
        zero = torch.zeros(param.size()).to(param.get_device())
        L2 += loss_fn(param, zero)
    return const * L2