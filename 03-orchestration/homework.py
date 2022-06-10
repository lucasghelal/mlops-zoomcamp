import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, logging
from prefect.task_runners import SequentialTaskRunner
from prefect.orion.schemas.schedules import CronSchedule


log = logging.get_logger()

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        log.info(f"The mean duration of training is {mean_duration}")
    else:
        log.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    log.info(f"The shape of X_train is {X_train.shape}")
    log.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    log.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    log.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    layout_data = "data/fhv_tripdata_{}-{:02d}.parquet"
    if date:
        date_format = datetime.strptime(date, "%Y-%m-%d")
        
        date_train = date_format - relativedelta(months=2)
        date_val = date_format - relativedelta(months=1)
    else:
        date_train = datetime.today() - relativedelta(months=2)
        date_val = datetime.today() - relativedelta(months=1)
    
    path_data_train = layout_data.format(date_train.year, date_train.month)
    path_data_val = layout_data.format(date_val.year, date_val.month)

    return path_data_train, path_data_val

def save_object(obj, path):
    with open(path, 'wb') as f_out:
        pickle.dump(obj, f_out)

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    save_object(lr, f'models/model-{date}.pkl')
    save_object(dv, f'models/dv-{date}.pkl')

    run_model(df_val_processed, categorical, dv, lr)


# main(date="2021-08-15")
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *",
                          timezone="America/Sao_Paulo"),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)
