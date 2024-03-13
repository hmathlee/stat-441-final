import pandas as pd
from sklearn.model_selection import train_test_split


def categorical_dmatrix(X):
    res = X.copy(deep=True)
    for var in res.columns:
        if res[var].dtype == "O":
            res[var] = res[var].astype("category")
    return res


def data_split(X_train_path, y_train_path, X_test_path, test_size, random_state, one_hot_encode=True):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["label"]

    X_test = pd.read_csv(X_test_path)

    X_train = X_train.drop("id", axis=1)
    X_test = X_test.drop("id", axis=1)

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    y_train = y_train.iloc[X_train.index]

    if one_hot_encode:
        X_train = pd.get_dummies(X_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": X_test
    }


def preprocess(dataset):
    return dataset.dropna(axis=0)
