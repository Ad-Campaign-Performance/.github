from pprint import pprint
import dvc.api

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import os, sys

path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
sys.path.insert(0, path_parent+'/scripts')

from mlflow_utils import fetch_logged_data

path="gdrive://1K5jndf5P6ES1AxLJj69nbVYiVrYpkIJM"
repo="../"
version="v1"

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version,
)

mlflow.set_experiment('ab_xgboost')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    return rmse, mae, r2

def main():
    # prepare example dataset
    data = pd.read_csv(data_url)
    
    #log data params
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_colums', data.shape[1])
    
    train, test = train_test_split(data)
    
    x_train = train.drop(['response'], axis=1)
    y_train = train[['response']]
    X_test = test.drop(['response'], axis=1)
    y_test = test[['response']]
    
    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    regressor = xgb.XGBRegressor(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3)
    
    regressor.fit(x_train, y_train, eval_set=[()])
    
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