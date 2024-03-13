import pandas as pd
from xgboost import train, DMatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from utils import categorical_dmatrix


class XGBoost:
    def __init__(self):
        self.model = "xgboost"
        self.needs_one_hot = False

    def train(self, X_train, y_train, X_val, y_val):
        y_train = y_train.replace(-1, 0)
        y_val = y_val.replace(-1, 0)

        dtrain = DMatrix(categorical_dmatrix(X_train), label=y_train, enable_categorical=True)
        dval = DMatrix(categorical_dmatrix(X_val), label=y_val, enable_categoricall=True)

        params = {'max_depth': 3, 'eta': 1, 'num_class': 5, 'objective': 'multi:softmax', 'eval_metric': 'mlogloss'}
        evals = [(dval, "val")]
        evals_result = {}

        model = train(params, dtrain, evals=evals, evals_result=evals_result)
        return list(evals_result)


class Logistic:
    def __init__(self):
        self.model = "logistic"
        self.needs_one_hot = True

    def train(self, X_train, y_train, X_val, y_val):
        model = LogisticRegression().fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)
        return log_loss(y_val, val_pred)
