# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

# Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
from torch import nn

# Data
import pandas as pd
import numpy as np

# File utils
from yaml import safe_load

# Custom
from utils import pandas2torch

# Config
cfg = safe_load(open("../config.yaml", "r"))


class Logistic():
    def __init__(self):
        super().__init__()
        self.model = "logistic"

    def train(self, X_train, y_train, X_val, y_val):
        model = LogisticRegression().fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)
        pred_class = np.argmax(val_pred, axis=1)
        return {
            "logloss": log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }


class SVM():
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


class NN(nn.Module):
    def __init__(self, dims):
        super(NN, self).__init__()
        # self.b1 = nn.BatchNorm1d(dims[0])
        # self.b2 = nn.BatchNorm1d(dims[1])
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])

    def forward(self, x):
        # x = self.b1(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc3(x)
        x = F.relu(x)

        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)

        return x


class NeuralNetwork():
    def __init__(self):
        super().__init__()
        self.net = None
        self.epochs = cfg["NN"]["EPOCHS"]
        self.model = "nn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.one_hot_encode = True
        self.feature_select = False
        self.scale = True
        self.feature_engineer = True
        self.correlation = True

    def train_one_epoch(self, train_loader, optim, loss_fn):
        loss_sum = 0

        for i, data in enumerate(train_loader):
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

    def train(self, X_train, y_train, X_val, y_val):
        torch.set_num_threads(6)

        input_dim = len(X_train.columns)
        hidden_dims = cfg["NN"]["H_DIM"]
        output_dim = y_train.nunique()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.net = NN(dims).to(self.device)

        dtrain = pandas2torch(X_train, y_train)
        dval = pandas2torch(X_val, y_val)

        train_loader = DataLoader(dtrain, batch_size=cfg["NN"]["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(dval, batch_size=cfg["NN"]["BATCH_SIZE"], shuffle=False)

        optim = SGD(self.net.parameters(), lr=cfg["NN"]["LR"], weight_decay=cfg["NN"]["L2_REG"])
        loss_fn = nn.NLLLoss().to(self.device)

        val_loss_sum = 0
        best_val_loss = 10000
        curr_acc = 0
        for epoch in range(self.epochs):
            print("Epoch {}:".format(epoch + 1))

            self.net.train(True)
            train_loss = self.train_one_epoch(train_loader, optim, loss_fn)
            print("Training loss: {}".format(train_loss))

            self.net.eval()
            correct = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    x, y = data
                    x = x.to(self.device)
                    y = y.type(torch.LongTensor).to(self.device)
                    pred = self.net(x)
                    loss = loss_fn(pred, y)
                    val_loss_sum += loss

                    # Validation accuracy
                    pred_class = torch.argmax(pred, dim=1)
                    correct += (pred_class == y).float().sum()

            val_loss = val_loss_sum / len(val_loader)
            val_loss_sum = 0
            val_acc = correct / len(y_val)
            print("Validation loss: {}".format(val_loss))
            print("Validation accuracy: {}".format(val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                curr_acc = val_acc
                model_path = 'model.pt'.format(epoch + 1)
                torch.save(self.net.state_dict(), model_path)

        return {
            "logloss": best_val_loss,
            "acc": curr_acc
        }

    def predict(self, X_test):
        pass