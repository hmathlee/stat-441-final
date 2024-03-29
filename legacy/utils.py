# File utils
from yaml import safe_load

from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.nn.functional import normalize

# Config
cfg = safe_load(open("config.yaml", "r"))


def df_copy(dfs):
    return [df.copy() for df in dfs]


def label_encode(X):
    le = LabelEncoder()
    for var in X.columns:
        if X[var].dtype in ["object", "category"]:
            X[var] = le.fit_transform(X[var])
    return X


def pandas2torch(X, y):
    X = label_encode(X)
    # Cast booleans in X to ints (for numpy conversion)
    for var in X.columns:
        if X[var].dtype == "bool":
            X[var] = X[var].astype(int)

    X_tensor = Tensor(X.to_numpy())
    y_tensor = Tensor(y.to_numpy())
    return TensorDataset(normalize(X_tensor), y_tensor)