from pprint import pprint
import dvc.api

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os, sys

path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
sys.path.insert(0, path_parent+'/scripts')

from mlflow_utils import fetch_logged_data

path="gdrive://1K5jndf5P6ES1AxLJj69nbVYiVrYpkIJM"
repo="../"
version="v4"

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version,
)

mlflow.set_experiment('ab_logistic')

# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
    
#     return rmse, mae, r2

def main():
    # prepare example dataset
    data = pd.read_csv(data_url)
    
    #log data params
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_colums', data.shape[1])
    
    train, test = train_test_split(data, test_size=0.30)
    
    x_train = train.drop(['response'], axis=1)
    y_train = train[['response']]
    X_test = test.drop(['response'], axis=1)
    y_test = test[['response']]
    

    lr = LogisticRegression()
    
    lr.fit(x_train, y_train)
    
    score = lr.score(X_test, y_test)
    
        
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


if __name__ == "__main__":
    main()