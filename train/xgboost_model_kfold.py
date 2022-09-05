from pprint import pprint
import dvc.api
from numpy import absolute
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import os, sys

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

path_parent = os.path.dirname(os.getcwd())
# <<<<<<< HEAD:train/xgboost_temp.py
# # os.chdir(path_parent)
# # sys.path.insert(0, path_parent+'/scripts')
# sys.path.append(os.path.abspath(os.path.join('..')))
# sys.path.insert(0,path_parent+'/scripts')
# =======
os.chdir(path_parent)
sys.path.insert(0, path_parent+'/scripts')
import io 


from mlflow_utils import fetch_logged_data

# path="gdrive://1K5jndf5P6ES1AxLJj69nbVYiVrYpkIJM"
path="data/AdSmartABdata.csv"
repo="C:/Users/user/Desktop/TenAcademy/SmartAd_A-B_Testing_user_analysis"
version="v5"


data_url = dvc.api.read(
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
    
    np.random.seed(1996)
    

    data = pd.read_csv(io.StringIO(data_url), sep=",")
   
    data.drop(columns=['Unnamed: 0', 'date', 'auction_id', 'yes', 'no'], inplace=True)
    data.drop(data[data['response'] == 2].index, inplace = True)
    print(data)
    #log data params
    # mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_colums', data.shape[1])
    
    X = data.drop(['response'], axis=1)
    Y = data[['response']]
   
    

    # mlflow.xgboost.autolog()

    regressor = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    
    # define model evaluation method
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate model
    results = cross_val_score(regressor, X, Y, scoring='neg_root_mean_squared_error', cv=cv)
    # results = absolute(results)
    print("RMSE: %f (%f)" % (results.mean()*100, results.std()*100))
    
    results = cross_val_score(regressor, X, Y, scoring='neg_mean_absolute_error', cv=cv)
    # results = absolute(results)
    print("RMSE: %f (%f)" % (results.mean()*100, results.std()*100))
    
    results = cross_val_score(regressor, X, Y, scoring='r2', cv=cv)
    # results = absolute(results)
    print("RMSE: %f (%f)" % (results.mean()*100, results.std()*100))
    
    # train, test = train_test_split(data)
    
    # X_test = test.drop(['response'], axis=1)
    # y_test = test[['response']]
    
    # y_pred = regressor.predict(X_test)
    
    # rmse, mae, r2 = eval_metrics(y_test, y_pred)
    
    # print("\n---------- Metrics ----------")
    # print("RMSE: ", rmse, '\n')
    # print("MAE: ", mae, '\n')
    # print("R2: ", r2, '\n')
    
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()