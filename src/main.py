from yaml import safe_load

from utils import read_csv_data, preprocess
from models import XGBoost

# Config
cfg = safe_load(open("../config.yaml", "r"))


if __name__ == "__main__":
    # Read data
    X_train, y_train, X_test, val_idx = read_csv_data(cfg["X_TRAIN"], cfg["Y_TRAIN"], cfg["X_TEST"], cfg["VAL"])
    X_train, y_train, X_val, y_val, X_test = preprocess(X_train, y_train, X_test, val_idx)

    # XGBoost for feature selection
    xgb_select = XGBoost()
    print("Fitting XGBoost. {} features".format(len(X_train.columns)))
    xgb_select_metrics = xgb_select.train(X_train, y_train, X_val, y_val, cfg["XGBOOST"])
    print("XGBoost log loss:", xgb_select_metrics["logloss"])
    print("XGBoost accuracy:", xgb_select_metrics["accuracy"])




#
# # Driver code
# if __name__ == "__main__":
#     # 80% of the data is training; to get (VAL_SIZE * 100)% validation data, divide by 0.8
#     val_sample = VAL_SIZE / 0.8
#
#     # Read data
#     dfs = read_csv_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, val_sample)
#
#     # Deep copy the dataframes so that the originals remain intact after preprocessing
#     X_train, y_train, X_test, val_idx = df_copy(dfs)
#
#     # Use XGBoost for feature selection, then run selected dataset on SVM
#     xgb_model = XGBoost()
#     dataset = preprocess2(X_train, y_train, X_test, val_idx)
#     X_train, y_train, X_val, y_val, X_test = dataset
#
#     # Train XGBoost and report validation metrics
#     print("+================================+")
#     print("Fitting " + xgb_model.model + " with:")
#     print("n_train = " + str(len(X_train)))
#     print("n_val = " + str(len(X_val)))
#     print("Training features = " + str(len(X_train.columns)))
#     xgboost_metrics = xgb_model.train(X_train, y_train, X_val, y_val, cfg["XGBOOST"]["SELECTION"])
#     print("XGBoost log loss:", xgboost_metrics["logloss"])
#     print("XGBoost accuracy:", xgboost_metrics["accuracy"])
#
#     # Prediction
#     pred_dir = datetime.now().strftime("%X").replace(':', '_')
#     os.makedirs(pred_dir)
#     print("Predicting with " + xgb_model.model)
#     pred_df = xgb_model.predict(X_test)
#     pred_path = os.path.join(pred_dir, xgb_model.model + '_pred.csv')
#     pred_df.to_csv(pred_path, index_label="id")
#
#     # Subset features based on XGBoost importance matrix
#     X_train, X_val, X_test = select_features(X_train, X_val, X_test, xgb_model.trained)
#
#     # Convert train, val, and test sets into numpy arrays for better use with other models
#     X_train, X_val, X_test = pandas2numpy(X_train, X_val, X_test)
#
#     models = [XGBoost()]
#     val_metrics = {}
#     for model in models:
#         # Train each model and report validation metrics
#         print("+================================+")
#         print("Fitting " + model.model + " with:")
#         print("n_train = " + str(len(X_train)))
#         print("n_val = " + str(len(X_val)))
#         print("Training features = " + str(len(X_train.columns)))
#         val_metrics[model.model] = model.train(X_train, y_train, X_val, y_val, cfg["XGBOOST"]["PREDICTION"])
#         print(val_metrics)
#
#         # Prediction
#         pred_dir = datetime.now().strftime("%X").replace(':', '_')
#         os.makedirs(pred_dir)
#
#         print("Predicting with " + model.model)
#         pred_df = model.predict(X_test)
#         pred_path = os.path.join(pred_dir, model.model + '_pred.csv')
#         pred_df.to_csv(pred_path, index_label="id")