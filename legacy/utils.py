# Torch
from torch import Tensor
from torch.utils.data import TensorDataset

# Data
import pandas as pd
import numpy as np


def df_copy(dfs):
    return [df.copy() for df in dfs]


def pandas2numpy(X_train, X_val, X_test):
    # One-hot-encode
    X = pd.concat([X_train, X_val, X_test])
    X = pd.get_dummies(X)

    n_train, n_val = len(X_train), len(X_val)

    # Scale features between 0 and 1
    mean = np.tile(np.mean(X, axis=1), (X.shape[1], 1)).transpose()
    std = np.tile(np.std(X, axis=1), (X.shape[1], 1)).transpose()
    X = X * mean / std

    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]

    return X_train, X_val, X_test


def pandas2torch(X, y):
    # Cast booleans in X to ints (for numpy conversion)
    for var in X.columns:
        if X[var].dtype == "bool":
            X[var] = X[var].astype(int)
    X_tensor = Tensor(X.to_numpy())
    y_tensor = Tensor(y.to_numpy())
    return TensorDataset(X_tensor, y_tensor)