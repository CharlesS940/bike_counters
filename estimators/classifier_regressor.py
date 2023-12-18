import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import subprocess
subprocess.run(["pip", "install", "vacances_scolaires_france"])  # for kaggle
from vacances_scolaires_france import SchoolHolidayDates
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


# for kaggle submission
train_data = pd.read_parquet('/kaggle/input/train-and-test/train.parquet')
test_data = pd.read_parquet('/kaggle/input/train-and-test/final_test.parquet')

# for local tests
#train_data = pd.read_parquet('train.parquet')
#test_data = pd.read_parquet('final_test.parquet')

# data from https://www.visualcrossing.com/weather-history/Paris%2CFrance
def merge_data(data):

    data = data.copy()

    # creating date column in correct format to merge datasets
    weather_data = pd.read_csv('/kaggle/input/weather-data/weather_data_paris_daily.csv')
    
    # for local tests
    #weather_data = pd.read_csv('weather_data_paris_daily.csv')
    data.loc[:, 'date'] = pd.to_datetime(data['date'])
    data.loc[:, 'date2'] = data['date'].dt.strftime('%Y-%m-%d')
    relevant_data = weather_data[['humidity', 'datetime', 'precip', 'winddir', 'icon']]
    merged_data = pd.merge(data, relevant_data, left_on='date2', right_on='datetime', how='left')

    return merged_data.drop(columns=(['date2', 'datetime']))

#curfews = [
#    ('2020-10-17', '2020-10-29', '21:00', '06:00'),
#    ('2020-10-30', '2020-12-15', '00:00', '23:59'),
#    ('2020-12-15', '2021-05-19', '18:00', '06:00'),
#    ('2021-04-03', '2021-06-9', '00:00', '23:59')
#    ]   

def is_within_lockdown_curfew(data):
    indices = []
    lockdowns = [
        ('2020-10-28', '2020-12-15'),
        ('2021-04-03', '2021-06-09')
        ] 
    curfew = [
        ('2020-12-15', '2020-04-03'),
        ] 
    for start, end in lockdowns:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()

        # Separate date and time conditions
        condition = (data['date'].dt.date >= start) & (data['date'].dt.date <= end)

        # Append indices for rows that meet the condition
        indices.extend(data[condition].index.tolist())
        data['is_lockdown'] = data.index.isin(indices).astype(int)
    
    for start, end in curfew:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()

        # Separate date and time conditions
        condition = (data['date'].dt.date >= start) & (data['date'].dt.date <= end)

        # Append indices for rows that meet the condition
        indices.extend(data[condition].index.tolist())
        data['is_curfew'] = data.index.isin(indices).astype(int)

    return data

def is_holiday(data):

    fr_holidays = holidays.France()

    data['is_bank_holiday'] = data['date'].apply(lambda x: x in fr_holidays)

    # Initialize the SchoolHolidayDates object
    school_holidays = SchoolHolidayDates()

    # Retrieve school holidays for Paris (Zone C) for the desired years
    years = [2020, 2021]
    school_holidays = SchoolHolidayDates()
    paris_holidays_2020 = school_holidays.holidays_for_year_and_zone(2020, 'C')
    paris_holidays_2021 = school_holidays.holidays_for_year_and_zone(2021, 'C')

    # Generate a list of all individual holiday dates
    holiday_dates = [holiday['date'] for holiday in paris_holidays_2020.values()]
    holiday_dates.extend([holiday['date'] for holiday in paris_holidays_2021.values()])

    # Assuming df is your existing DataFrame
    data['is_school_holiday'] = data['date'].isin(holiday_dates)

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

def cyclical_encoding(data):
        
        period = np.max(data)
        sin_transform = np.sin(2 * np.pi * data / period)
        cos_transform = np.cos(2 * np.pi * data / period)
        return np.column_stack((sin_transform, cos_transform))

class ZeroClassifierRegressor(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self):
        self.classifier = CatBoostClassifier(verbose=0)
        self.regressor = CatBoostRegressor(subsample=0.9,
                                  learning_rate=0.25421,
                                  iterations=1500,
                                  l2_leaf_reg=1,
                                  depth=8,
                                  border_count=32,
                                  bootstrap_type='Bernoulli',
                                  sampling_frequency='PerTree',
                                  verbose=0)

    def fit(self, X, y):
        # Train the classifier
        is_non_zero = (y != 0).astype(int)
        self.classifier.fit(X, is_non_zero)

        # Train the regressor on non-zero values
        X_non_zero = X[is_non_zero == 1]
        y_non_zero = y[is_non_zero == 1]
        if len(X_non_zero) > 0:  # Check if there are non-zero values
            self.regressor.fit(X_non_zero, y_non_zero)

        return self

    def predict(self, X):
        # Predict zero vs non-zero
        is_non_zero = self.classifier.predict(X)

        # Initialize predictions array
        predictions = np.zeros(X.shape[0])

        # Predict with the regressor where classifier predicts non-zero
        if np.any(is_non_zero):
            X_non_zero = X[is_non_zero == 1]
            predictions[is_non_zero == 1] = self.regressor.predict(X_non_zero)

        return predictions
    
def get_estimator():
    data_merge = FunctionTransformer(merge_data)

    lockdown_dates = FunctionTransformer(is_within_lockdown_curfew)

    holiday_encoder = FunctionTransformer(is_holiday)

    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day']

    cycle_encoder = FunctionTransformer(cyclical_encoding)
    cycle_cols = ['hour', 'weekday']

    num_encoder = StandardScaler()
    num_cols = ['precip', 'humidity']

    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_cols = ['winddir', 'icon', 'counter_name']

    binary_cols = ['is_bank_holiday', 'is_lockdown', 'is_school_holiday', 'is_curfew']

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ('cycle', cycle_encoder, cycle_cols),
            ('num', num_encoder, num_cols),
            ('cat', categorical_encoder, categorical_cols),
            ('binary', 'passthrough', binary_cols)
        ]
    )
    classifier_regressor = ZeroClassifierRegressor()

    pipe = make_pipeline(
        data_merge,
        lockdown_dates,
        holiday_encoder,
        date_encoder,
        preprocessor,
        classifier_regressor,
    )

    return pipe

X_train, y_train = train_data[['counter_name', 'date']], train_data[['log_bike_count']]
X_test = test_data[['counter_name', 'date']]
pipe = get_estimator()

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

pred_df = pd.DataFrame({'Id': X_test.index,
                        'log_bike_count': pred})

# for kaggle submission
pred_df.to_csv('submission.csv', index=False)