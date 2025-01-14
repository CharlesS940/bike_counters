import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import holidays
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

train_data = pd.read_parquet('/kaggle/input/train-and-test/train.parquet')
test_data = pd.read_parquet('/kaggle/input/train-and-test/final_test.parquet')

def merge_data(data):

    # creating date column in correct format to merge datasets
    weather_data = pd.read_csv('/kaggle/input/weather-data/weather_data_paris_daily.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['date2'] = data['date'].dt.strftime('%Y-%m-%d')
    relevant_data = weather_data[['temp', 'datetime', 'humidity', 'precip']]
    merged_data = pd.merge(data, relevant_data, left_on='date2', right_on='datetime', how='left')

    return merged_data.drop(columns=('date2'))

def is_within_curfew(data):
    indices = []
    curfews = [
    ('2020-10-17', '2020-10-29', '21:00', '06:00'),
    ('2020-10-30', '2020-12-15', '00:00', '23:59'),
    ('2020-12-15', '2021-05-19', '18:00', '06:00'),
    ('2021-05-19', '2021-06-20', '21:00', '06:00')
    ]   
    for start, end, start_time, end_time in curfews:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()
        start_time = pd.to_datetime(start_time).time()
        end_time = pd.to_datetime(end_time).time()

        # Separate date and time conditions
        date_condition = (data['date'].dt.date >= start) & (data['date'].dt.date <= end)
        time_condition = (data['date'].dt.time >= start_time) & (data['date'].dt.time <= end_time)

        # Combined condition
        condition = date_condition & time_condition

        # Append indices for rows that meet the condition
        indices.extend(data[condition].index.tolist())
        data['is_lockdown'] = data.index.isin(indices).astype(int)

    return data

def is_holiday(data):

    fr_holidays = holidays.France()
    data['is_holiday'] = data['date'].isin(fr_holidays).astype(int)

    return data

def _encode_dates(X):
    X = X.copy()  # modify a copy of X

    # Encode the date information from the "date" columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sin_transform = np.sin(2 * np.pi * X / self.period)
        cos_transform = np.cos(2 * np.pi * X / self.period)
        return np.column_stack((sin_transform, cos_transform))
    
def get_estimator():
    data_merge = FunctionTransformer(merge_data)

    lockdown_dates = FunctionTransformer(is_within_curfew)

    holiday_encoder = FunctionTransformer(is_holiday)

    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day', 'weekday', 'hour']

    #cycle_encoder = CyclicalEncoder()
    #cycle_cols = ['hour', 'weekday']

    num_encoder = StandardScaler()
    num_cols = ['latitude', 'longitude', 'temp', 'humidity', 'precip']

    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_cols = []

    binary_cols = ['is_lockdown', 'is_holiday']

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ('num', num_encoder, num_cols),
            ('cat', categorical_encoder, categorical_cols),
            ('binary', 'passthrough', binary_cols)
        ]
    )
    regressor = CatBoostRegressor(depth=11,
        iterations=1000,
        rsm=0.05,
        sampling_frequency="PerTree",
        subsample=0.9,
        verbose=0,)

    pipe = make_pipeline(
        data_merge,
        lockdown_dates,
        holiday_encoder,
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe

X_train, y_train = train_data[['longitude', 'latitude', 'date']], train_data[['log_bike_count']]
X_test = test_data[['longitude', 'latitude', 'date']]
pipe = get_estimator()

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

pred_df = pd.DataFrame({'Id': X_test.index,
                        'log_bike_count': pred})

pred_df.to_csv('submission.csv', index=False)