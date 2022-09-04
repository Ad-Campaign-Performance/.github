import io
from mlflow_utils import fetch_logged_data
from pprint import pprint
import dvc.api
#model library
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
#mlflows library
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import os
import sys

path_parent = os.path.dirname(os.getcwd())

os.chdir(path_parent)
sys.path.insert(0, path_parent+'/scripts')


# path="gdrive://1K5jndf5P6ES1AxLJj69nbVYiVrYpkIJM"
path = "data/AdSmartABdata.csv"
repo = "/home/owon/Documents/10x/Week2/SmartAd_A-B_Testing_user_analysis/data"
version = "V4.0"


data_url = dvc.api.read(
    path=path,
    repo=repo,
    rev=version,
)
data_url2 = '../data/AdSmartABdata.csv'

mlflow.set_experiment('ab_xgboost')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


def main():

    np.random.seed(1996)

    data = pd.read_csv(io.StringIO(data_url), sep=",")
    print(data)
    data.drop(columns=['Unnamed: 0', 'date',
              'auction_id', 'yes', 'no'], inplace=True)

    # log data params
    # mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_colums', data.shape[1])

    train, test = train_test_split(data)

    x_train = train.drop(['yes'], axis=1)
    y_train = train[['yes']]
    X_test = test.drop(['yes'], axis=1)
    y_test = test[['yes']]

    # enable auto logging
    # this includes xgboost.sklearn estimators

    regressor = XGBRegressor()

    regressor.fit(x_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = regressor.predict(X_test)

    rmse, mae, r2 = eval_metrics(y_test, y_pred)

    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()
