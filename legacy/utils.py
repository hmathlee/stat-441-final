# File utils
from yaml import safe_load

# Config
cfg = safe_load(open("config.yaml", "r"))


def df_copy(dfs):
    return [df.copy() for df in dfs]
