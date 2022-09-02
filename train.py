import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse




import DVC as dvc
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# get url from DVC

path = 'data/AdSmartABdata.csv'
repo = './'
version = 'v1'

data_url = dvc.API.GetURL(
    path = path,
    repo = repo,
    rev = version
    )

mlflow.set_experiment('demo')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(200)
    
    csv_url = (path)
    
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        
    with mlflow.start_run():
        mlflow.log_param('alpha',1)
        mlflow.log_metric('beta',2)