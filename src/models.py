from yaml import safe_load

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

cfg = safe_load(open("../config.yaml", 'r'))

rng = np.random.default_rng(cfg["SEED"])


class XGBoost():
    def __init__(self):
        super().__init__()
        self.model = "xgboost"
        self.trained = None

    def train(self, X_train, y_train, X_val, y_val, xgb_cfg):
        # Watchlist (monitor over-fitting)
        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Define the model with parameters from the config file
        model = XGBClassifier(n_estimators=xgb_cfg["ROUNDS"], objective="multi:softprob", eval_metric="mlogloss",
                              enable_categorical=xgb_cfg["CATEGORICAL"], tree_method="hist",
                              colsample_bytree=xgb_cfg["SUBSET"], n_jobs=cfg["N_JOBS"], early_stopping_rounds=50)

        # Grid search cross-validation
        params = {"max_depth": xgb_cfg["DEPTH"], "eta": xgb_cfg["LR"], "min_child_weight": xgb_cfg["MIN_CHILD_WGT"]}
        xgb_search = GridSearchCV(model, params, n_jobs=cfg["N_JOBS"])
        xgb_search.fit(X_train, y_train, eval_set=eval_set, verbose=2)

        best_model = xgb_search.best_estimator_
        self.trained = best_model

        # Output best set of parameters
        for param in xgb_search.best_params_:
            print("{}: {}".format(param, xgb_search.best_params_[param]))

        # Evaluate best model on preemptive test set
        val_pred = self.trained.predict_proba(X_val)
        pred_class = np.argmax(val_pred, axis=1)

        return {
            "logloss":  log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }

    def predict(self, X_test):
        pred = self.trained.predict_proba(X_test)

        # Retrieve column names for prediction dataframe
        names_file = open("../names.txt", "r")
        columns = names_file.readlines()
        names_file.close()

        # Remove the newline characters
        for i in range(len(columns) - 1):
            columns[i] = columns[i][:-1]

        # Construct prediction dataframe
        pred_df = pd.DataFrame(pred, columns=columns)

        return pred_df
