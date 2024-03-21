from utils import read_csv_data, df_copy, preprocess
from models import XGBoost, Logistic, SVM, NeuralNetwork

# Paths
X_TRAIN_PATH = "data/X_train.csv"
Y_TRAIN_PATH = "data/y_train.csv"
X_TEST_PATH = "data/X_test.csv"

# Constants
VAL_SIZE = 0.2
RANDOM_STATE = 420

# Driver code
if __name__ == "__main__":
    # 80% of the data is training; to get (VAL_SIZE * 100)% validation data, divide by 0.8
    val_sample = VAL_SIZE / 0.8

    dfs = read_csv_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, val_sample)

    # Deep copy the dataframes so that the originals remain intact after preprocessing
    X_train, y_train, X_test, val_idx = df_copy(dfs)
    dataset = preprocess(X_train, y_train, X_test, val_idx)

    val_metrics = {}

    # models = [Logistic(), XGBoost(), SVM(), NeuralNetwork()]
    models = [XGBoost()]
    for model in models:
        # Call model's separate preprocessor function if applicable
        if model.default_preprocessor():
            dat = dataset
        else:
            X_train, y_train, X_test, val_idx = df_copy(dfs)
            dat = preprocess(X_train, y_train, X_test, val_idx,
                             model.one_hot_encode, model.feature_select, model.scale)

        # Train each model and report validation metrics
        X_train, y_train, X_val, y_val, X_test = dat
        print("+================================+")
        print("Fitting " + model.model + " with:")
        print("n_train = " + str(len(X_train)))
        print("n_val = " + str(len(X_val)))
        print("Training features = " + str(len(X_train.columns)))
        val_metrics[model.model] = model.train(X_train, y_train, X_val, y_val)

    print(val_metrics)
