import os
from yaml import safe_load
from joblib import load

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import Rprop
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Custom
from utils import save_model_as_joblib, report_validation_metrics, categorical_dmatrix
from legacy.utils import pandas2torch

cfg = safe_load(open("config.yaml", "r"))
model_out = cfg["MODEL_OUT"]

rng = np.random.default_rng(cfg["SEED"])


class XGBoost:
    def __init__(self, xgb_cfg, from_pickle=False):
        super().__init__()
        self.model = "xgboost"
        self.xgb_cfg = xgb_cfg

        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            self.trained = XGBClassifier(n_estimators=self.xgb_cfg["ROUNDS"], objective="multi:softprob",
                                         eval_metric="mlogloss", enable_categorical=self.xgb_cfg["CATEGORICAL"],
                                         tree_method="hist", colsample_bytree=self.xgb_cfg["SUBSET"],
                                         n_jobs=cfg["N_JOBS"], gamma=1.54, random_state=cfg["SEED"])

            # Set tuning parameters
            self.trained.set_params(max_depth=self.xgb_cfg["DEPTH"][0], learning_rate=self.xgb_cfg["LR"][0],
                                    min_child_weight=self.xgb_cfg["MIN_CHILD_WGT"][0])

    def train(self, X_train, y_train, X_val, y_val, pre_test=None):
        X_train = categorical_dmatrix(X_train)
        X_val = categorical_dmatrix(X_val)

        X_pre_test, y_pre_test = None, None

        # If we want to set aside a preemptive test set for further evaluation purposes
        if pre_test > 0:
            n_train = len(X_train)
            pre_test_idx = rng.choice(np.arange(n_train), size=int(pre_test * n_train), replace=False)
            pre_test_data = X_train.index.isin(pre_test_idx)
            X_pre_test = X_train.iloc[pre_test_data]
            y_pre_test = y_train.iloc[pre_test_data]
            X_train_new = X_train.iloc[~pre_test_data]
            y_train_new = y_train.iloc[~pre_test_data]
        else:
            X_train_new = X_train
            y_train_new = y_train

        # Watchlist (monitor over-fitting)
        eval_set = [(X_train_new, y_train_new), (X_val, y_val)]

        # Single train-test pass
        assert len(self.xgb_cfg["DEPTH"]) == 1, \
            "Depth parameter has length {} when 1 is required.".format(len(self.xgb_cfg["DEPTH"]))
        assert len(self.xgb_cfg["LR"]) == 1, \
            "Learning rate parameter has length {} when 1 is required.".format(len(self.xgb_cfg["LR"]))
        assert len(self.xgb_cfg["MIN_CHILD_WGT"]) == 1, \
            "Min child wgt parameter has length {} when 1 is required.".format(len(self.xgb_cfg["MIN_CHILD_WGT"]))

        self.trained = self.trained.fit(X_train_new, y_train_new, eval_set=eval_set, early_stopping_rounds=10)

        save_model_as_joblib(self)
        eval_metrics = report_validation_metrics(self, X_val, y_val)

        # Evaluation metrics on the preemptive test set, if applicable
        if pre_test > 0:
            pre_test_pred = self.trained.predict_proba(X_pre_test)
            pre_test_pred_class = np.argmax(pre_test_pred, axis=1)
            eval_metrics["pre_test_logloss"] = log_loss(y_pre_test, pre_test_pred)
            eval_metrics["pre_test_accuracy"] = accuracy_score(y_pre_test, pre_test_pred_class)

        return eval_metrics

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


class RandomForest:
    def __init__(self, from_pickle=False):
        rf = cfg["RF"]
        self.model = "random_forest"
        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            self.trained = RandomForestClassifier(n_estimators=rf["N_TREES"], criterion="log_loss",
                                                  max_depth=rf["MAX_DEPTH"], min_samples_leaf=rf["MIN_CHILD_WGT"],
                                                  max_features=rf["MAX_FTRS"], n_jobs=cfg["N_JOBS"], verbose=1,
                                                  max_samples=rf["MAX_SAMPLES"], random_state=cfg["SEED"])

    def train(self, X_train, y_train, X_val, y_val):
        self.trained.fit(X_train, y_train)

        train_pred = self.trained.predict_proba(X_train)
        train_pred_class = np.argmax(train_pred, axis=1)
        print("Random forest log loss (train):", log_loss(y_train, train_pred))
        print("Random forest accuracy (train):", accuracy_score(y_train, train_pred_class))

        save_model_as_joblib(self)
        eval_metrics = report_validation_metrics(self, X_val, y_val)
        
        return eval_metrics


class NN(nn.Module):
    def __init__(self, dims):
        super(NN, self).__init__()

        self.b1 = nn.BatchNorm1d(dims[0])
        self.b2 = nn.BatchNorm1d(dims[1])
        self.b3 = nn.BatchNorm1d(dims[2])
        self.b4 = nn.BatchNorm1d(dims[3])
        self.b5 = nn.BatchNorm1d(dims[4])
        self.b6 = nn.BatchNorm1d(dims[5])

        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])
        self.fc5 = nn.Linear(dims[4], dims[5])
        self.fc6 = nn.Linear(dims[5], dims[6])

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)

    def forward(self, x):
        x = self.b1(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.b2(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.b3(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc3(x)
        x = F.relu(x)

        x = self.b4(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc4(x)
        x = F.relu(x)

        x = self.b5(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc5(x)
        x = F.relu(x)

        x = self.b6(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)

        return x


class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.net = None
        self.epochs = cfg["NN"]["EPOCHS"]
        self.model = "nn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_one_epoch(self, train_loader, optim, loss_fn):
        loss_sum = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Data
            x, y = data
            x = x.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)
            optim.zero_grad()

            # Train
            pred = self.net(x)

            # Training loss with L2 regularization
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()

        # Report the average training loss across the batches
        return loss_sum / len(train_loader)

    def train(self, X_train, y_train, X_val, y_val, return_history=False):
        torch.set_num_threads(6)

        hidden_dims = cfg["NN"]["H_DIM"]
        dims = [len(X_train.columns)] + hidden_dims + [y_train.nunique()]
        self.net = NN(dims).to(self.device)

        dtrain = pandas2torch(X_train, y_train)
        dval = pandas2torch(X_val, y_val)

        train_loader = DataLoader(dtrain, batch_size=cfg["NN"]["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(dval, batch_size=cfg["NN"]["BATCH_SIZE"], shuffle=False)

        optim = Rprop(self.net.parameters(), lr=cfg["NN"]["RPROP_LR"])

        loss_fn = nn.NLLLoss().to(self.device)

        # TensorBoard writer
        writer = SummaryWriter()

        val_loss_sum = 0
        best_val_loss = 10000
        curr_acc = 0
        train_history, val_history = [], []
        for epoch in range(self.epochs):
            print("Epoch {}:".format(epoch + 1))

            self.net.train(True)
            # scheduler.step()
            train_loss = self.train_one_epoch(train_loader, optim, loss_fn)
            train_history.append(train_loss)
            writer.add_scalar("logloss_train", train_loss, epoch)
            print("Training loss: {}".format(train_loss))

            self.net.eval()
            correct = 0
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    x, y = data
                    x = x.to(self.device)
                    y = y.type(torch.LongTensor).to(self.device)
                    pred = self.net(x)
                    loss = loss_fn(pred, y)
                    val_loss_sum += loss.item()

                    # Validation accuracy
                    pred_class = torch.argmax(pred, dim=1)
                    correct += (pred_class == y).float().sum()

            val_loss = val_loss_sum / len(val_loader)
            val_loss_sum = 0
            val_acc = correct / len(y_val)
            val_history.append(val_loss)
            writer.add_scalar("logloss_val", val_loss, epoch)
            print("Validation loss: {}".format(val_loss))
            print("Validation accuracy: {}".format(val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                curr_acc = val_acc
                model_path = 'model.pt'.format(epoch + 1)
                torch.save(self.net.state_dict(), model_path)

        writer.flush()
        writer.close()

        eval_metrics = {
            "logloss": best_val_loss,
            "accuracy": curr_acc
        }

        if return_history:
            eval_metrics["loss_history"] = {
                "train": train_history,
                "val": val_history
            }

        return eval_metrics

    def predict(self, X_test):
        pass


class Logistic:
    def __init__(self, from_pickle=False):
        super().__init__()
        self.model = "logistic"

        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            self.trained = LogisticRegression()

    def train(self, X_train, y_train, X_val, y_val):
        self.trained.fit(X_train, y_train)
        
        save_model_as_joblib(self)
        eval_metrics = report_validation_metrics(self, X_val, y_val)
        
        return eval_metrics


class KNearest:
    def __init__(self, from_pickle=False):
        self.model = "k_nearest"

        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            self.trained = KNeighborsClassifier(n_neighbors=cfg["KNN"]["NEIGHBORS"], n_jobs=cfg["N_JOBS"])

    def train(self, X_train, y_train, X_val, y_val):
        self.trained.fit(X_train, y_train)

        train_pred = self.trained.predict_proba(X_train)
        train_pred_class = np.argmax(train_pred, axis=1)
        print("KNN log loss (train):", log_loss(y_train, train_pred))
        print("KNN accuracy (train):", accuracy_score(y_train, train_pred_class))
        
        save_model_as_joblib(self)
        eval_metrics = report_validation_metrics(self, X_val, y_val)
        
        return eval_metrics


class SVM:
    def __init__(self, from_pickle=False):
        super().__init__()
        self.model = "svm"

        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            # Ensemble SVMs on subsets of data to speed up training
            self.trained = BaggingClassifier(SVC(gamma="auto", kernel="rbf", decision_function_shape="ovo"),
                                             n_estimators=cfg["SVM"]["ROUNDS"], max_samples=cfg["SVM"]["SUBSAMPLE"],
                                             n_jobs=cfg["N_JOBS"])

    def train(self, X_train, y_train, X_val, y_val):
        self.trained.fit(X_train, y_train)
        save_model_as_joblib(self)

        print("Calibrating probabilities...")
        calibrated = CalibratedClassifierCV(self.trained, n_jobs=cfg["N_JOBS"], cv="prefit")
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


class ModelStack:
    def __init__(self, prefit=True, from_pickle=False):
        self.model = "stack"

        if from_pickle:
            self.trained = load(os.path.join(model_out, self.model + ".pkl"))
        else:
            xgb_cfg = cfg["XGBOOST"]["PREDICTION"]
            xgb = XGBoost(xgb_cfg)
            rf = RandomForest()
            knn = KNearest()

            self.estimators = [
                ("xgboost", xgb.trained),
                ("random forest", rf.trained),
                ("knn", knn.trained)
            ]

            # Use XGBoost as the Level 1 estimator
            xgb_cfg = cfg["XGBOOST"]["STACK"]
            final_estimator = XGBClassifier(n_estimators=xgb_cfg["ROUNDS"], objective="multi:softprob",
                                            eval_metric="mlogloss", enable_categorical=xgb_cfg["CATEGORICAL"],
                                            tree_method="hist", colsample_bytree=xgb_cfg["SUBSET"],
                                            n_jobs=cfg["N_JOBS"], random_state=cfg["SEED"])
            final_estimator.set_params(max_depth=xgb_cfg["DEPTH"][0], learning_rate=xgb_cfg["LR"][0],
                                       min_child_weight=xgb_cfg["MIN_CHILD_WGT"][0])

            cv = "prefit" if prefit else 3
            self.trained = StackingClassifier(estimators=self.estimators, final_estimator=final_estimator, cv=cv,
                                              verbose=2, n_jobs=cfg["N_JOBS"])

    def train(self, X_train, y_train, X_val, y_val):
        self.trained.fit(X_train, y_train)
        save_model_as_joblib(self)

        train_pred = self.trained.predict_proba(X_train)
        train_pred_class = np.argmax(train_pred, axis=1)
        print("Stacked log loss (train):", log_loss(y_train, train_pred))
        print("Stacked accuracy (train):", accuracy_score(y_train, train_pred_class))

        eval_metrics = report_validation_metrics(self, X_val, y_val)
        return eval_metrics

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
