# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Data
import pandas as pd
import numpy as np

# File utils
from yaml import safe_load

# Config
cfg = safe_load(open("config.yaml", "r"))


class Logistic:
    def __init__(self):
        super().__init__()
        self.model = "logistic"
        self.trained = None

    def train(self, X_train, y_train, X_val, y_val):
        self.trained = LogisticRegression().fit(X_train, y_train)
        val_pred = self.trained.predict_proba(X_val)
        pred_class = np.argmax(val_pred, axis=1)
        return {
            "logloss": log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }


class KNearest:
    def __init__(self):
        self.model = "k_nearest"
        self.trained = None

    def train(self, X_train, y_train, X_val, y_val):
        self.trained = KNeighborsClassifier().fit(X_train, y_train)
        val_pred = self.trained.predict_proba(X_val)
        pred_class = np.argmax(val_pred, axis=1)
        return {
            "logloss": log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }


class SVM:
    def __init__(self):
        super().__init__()
        self.model = "svm"
        self.trained = None

    def train(self, X_train, y_train, X_val, y_val):
        # Ensemble SVMs on subsets of data to speed up training
        model = BaggingClassifier(SVC(gamma="auto", kernel="rbf", decision_function_shape="ovo"),
                                  n_estimators=cfg["SVM"]["ROUNDS"], max_samples=cfg["SVM"]["SUBSAMPLE"],
                                  n_jobs=cfg["N_JOBS"])

        model.fit(X_train, y_train)
        self.trained = model

        print("Calibrating probabilities...")
        calibrated = CalibratedClassifierCV(model, n_jobs=cfg["N_JOBS"], cv="prefit")
        calibrated.fit(X_train, y_train)

        # Training metrics
        train_score = calibrated.predict_proba(X_train)
        train_pred = 1 / (1 + np.exp(-train_score))
        pred_class = np.argmax(train_pred, axis=1)
        train_loss = log_loss(y_train, train_pred)
        train_acc = accuracy_score(y_train, pred_class)

        # Val metrics
        val_score = calibrated.predict_proba(X_val)
        val_pred = 1 / (1 + np.exp(-val_score))
        pred_class = np.argmax(val_pred, axis=1)
        val_loss = log_loss(y_val, val_pred)
        val_acc = accuracy_score(y_val, pred_class)

        print("Training loss: {} | Val loss: {}".format(train_loss, val_loss))
        print("Training acc: {} | Val acc: {}".format(train_acc, val_acc))

        return {
            "logloss": val_loss,
            "accuracy": val_acc
        }

    def predict(self, X_test):
        pred = self.trained.predict_proba(X_test)

        # Retrieve column names for prediction dataframe
        names_file = open("names.txt", "r")
        columns = names_file.readlines()
        names_file.close()

        # Remove the newline characters
        for i in range(len(columns) - 1):
            columns[i] = columns[i][:-1]

        # Construct prediction dataframe
        pred_df = pd.DataFrame(pred, columns=columns)

        return pred_df
