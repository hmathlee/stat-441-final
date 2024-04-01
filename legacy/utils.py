# File utils
from yaml import safe_load

from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.nn.functional import normalize

from src.utils import eliminate_variables, zero_out_negatives

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


def feature_transform(X):
    # Rewrite fieldwork start and end dates as numbers
    fw_start = X["fw_start"].apply(lambda x: x // 100 + (x % 100) / 12)
    fw_end = X["fw_end"].apply(lambda x: x // 100 + (x % 100) / 12)

    # Replace field work start and end with duration
    fw_duration = fw_end - fw_start
    X = X.assign(fw_duration=fw_duration)
    X = X.drop(columns=["fw_start", "fw_end"])
    X["fw_duration"] = X["fw_duration"].astype(float)

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

    for var in ["v20a", "v20b", "v45b", "v45c"]:
        response_idx = X[var][X[var] != -4]
        X.loc[response_idx, var[:-1]] = X[var][response_idx]

    X = X.drop(columns=["v20a", "v20b", "v45b", "v45c"])

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
        X.loc[dk_response_idx, var] = dk[dk_response_idx]

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

    X = X.drop(columns=["v45"])

    return X