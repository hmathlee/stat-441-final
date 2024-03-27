import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, log_loss

pred_dir = "02_42_47"

train_pred = pd.read_csv(os.path.join(pred_dir, "xgboost_train_pred.csv"))
y_train = pd.read_csv(os.path.join(pred_dir, "xgboost_train_label.csv"))

val_pred = pd.read_csv(os.path.join(pred_dir, "xgboost_val_pred.csv"))
y_val = pd.read_csv(os.path.join(pred_dir, "xgboost_val_label.csv"))

test_pred = pd.read_csv(os.path.join(pred_dir, "xgboost_pred.csv"))

pred = pd.concat([train_pred, val_pred])
y = pd.concat([y_train, y_val])

pred = pred.drop(columns=["id"]).to_numpy()
y = y.drop(columns=["Unnamed: 0"]).replace(-1, 0).to_numpy()

y = np.reshape(y, (len(y),))
print(np.bincount(y))

train_loss = log_loss(y, pred)
train_acc = accuracy_score(y, np.argmax(pred, axis=1))
print(train_loss, train_acc)

test_pred = test_pred.drop(columns=["id"]).to_numpy()
pseudo_labels = np.random.normal(2.3, 0.5, len(test_pred))
pseudo_labels = np.round(pseudo_labels)
test_loss = log_loss(pseudo_labels, test_pred)
test_acc = accuracy_score(pseudo_labels, np.argmax(test_pred, axis=1))

print(test_loss, test_acc)
