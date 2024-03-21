from yaml import safe_load

import numpy as np
from xgboost import train, DMatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.svm import SVC

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
from torch import nn

from utils import categorical_dmatrix, pandas2torch, L2_regularize

cfg = safe_load(open("config.yaml", 'r'))


class Model:
    def __init__(self):
        self.one_hot_encode = True
        self.feature_select = True
        self.scale = True

    def cross_val(self):
        pass

    def default_preprocessor(self):
        return self.one_hot_encode and self.feature_select and self.scale


class XGBoost(Model):
    def __init__(self):
        super().__init__()
        self.model = "xgboost"
        self.trained = None

        self.one_hot_encode = False
        self.feature_select = False
        self.scale = True

    def train(self, X_train, y_train, X_val, y_val):
        y_train = y_train.replace(-1, 0)
        y_val = y_val.replace(-1, 0)

        dtrain = DMatrix(categorical_dmatrix(X_train), label=y_train, enable_categorical=True)
        dval = DMatrix(categorical_dmatrix(X_val), label=y_val, enable_categorical=True)

        xgb = cfg["XGBOOST"]
        params = {'max_depth': xgb["DEPTH"], 'eta': xgb["LR"], 'num_class': xgb["N_CLASS"],
                  'objective': 'multi:softmax', 'eval_metric': 'mlogloss'}
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}

        model = train(params, dtrain, num_boost_round=xgb["ROUNDS"], evals=evals, evals_result=evals_result)
        self.trained = model

        val_pred = model.predict(dval)

        return {
            "logloss":  list(evals_result["val"]["mlogloss"])[-1],
            "accuracy": accuracy_score(y_val, val_pred)
        }


class Logistic(Model):
    def __init__(self):
        super().__init__()
        self.model = "logistic"
        self.one_hot_encode = True
        self.feature_select = True
        self.scale = True

    def train(self, X_train, y_train, X_val, y_val):
        model = LogisticRegression().fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)
        pred_class = np.argmax(val_pred, axis=1)
        return {
            "logloss": log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }


class SVM(Model):
    def __init__(self):
        super().__init__()
        self.model = "svm"
        self.one_hot_encode = True
        self.feature_select = True
        self.scale = True

    def train(self, X_train, y_train, X_val, y_val):
        model = SVC(gamma="auto", kernel="rbf", decision_function_shape="ovr")
        model.fit(X_train, y_train)
        val_score = model.decision_function(X_val)
        val_pred = 1 / (1 + np.exp(-val_score))
        pred_class = np.argmax(val_pred, axis=1)
        return {
            "logloss": log_loss(y_val, val_pred),
            "accuracy": accuracy_score(y_val, pred_class)
        }


class NN(nn.Module):
    def __init__(self, dims):
        super(NN, self).__init__()
        # self.b1 = nn.BatchNorm1d(dims[0])
        # self.b2 = nn.BatchNorm1d(dims[1])
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # x = self.b1(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc1(x)
        x = F.relu(x)

        # x = self.b2(x)
        x = nn.Dropout(cfg["NN"]["DROPOUT"])(x)
        x = self.fc2(x)
        x = F.softplus(x)

        return F.log_softmax(x, dim=1)


class NeuralNetwork(Model):
    def __init__(self):
        super().__init__()
        self.net = None
        self.epochs = cfg["NN"]["EPOCHS"]
        self.model = "nn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.one_hot_encode = True
        self.feature_select = False
        self.scale = True

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
        input_dim = len(X_train.columns)
        hidden_dim = cfg["NN"]["H_DIM"]
        output_dim = y_train.nunique()
        self.net = NN([input_dim, hidden_dim, output_dim]).to(self.device)

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
