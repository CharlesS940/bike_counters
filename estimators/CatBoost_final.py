# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import subprocess
subprocess.run(["pip", "install", "vacances_scolaires_france"])  # for kaggle
from vacances_scolaires_france import SchoolHolidayDates


# for kaggle submission
train_data = pd.read_parquet('/kaggle/input/train-and-test/train.parquet')
test_data = pd.read_parquet('/kaggle/input/train-and-test/final_test.parquet')

# for local tests
#train_data = pd.read_parquet('train.parquet')
#test_data = pd.read_parquet('final_test.parquet')

# importing weather data from 'url'
def merge_data(data):

    data = data.copy()

    # importing data
    weather_data = pd.read_csv('/kaggle/input/weather-data/weather_data_paris_daily.csv')
    
    # for local tests
    #weather_data = pd.read_csv('weather_data_paris_daily.csv')

    # creating date column in correct format to merge datasets
    data.loc[:, 'date'] = pd.to_datetime(data['date'])
    data.loc[:, 'date2'] = data['date'].dt.strftime('%Y-%m-%d')

    # selecting relevant data
    relevant_data = weather_data[['humidity', 'datetime', 'precip', 'winddir', 'icon']]
    merged_data = pd.merge(data, relevant_data, left_on='date2', right_on='datetime', how='left')

    return merged_data.drop(columns=(['date2', 'datetime'])) 

# creating is_lockdown and is_curfew variables
def is_within_lockdown_curfew(data):
    indices = []

    # lockdown dates from wikipedia page (we consider second lockdown ended once curfew was pushed to 11pm)
    lockdowns = [
        ('2020-10-28', '2020-12-15'),
        ('2021-04-03', '2021-06-09')
        ] 
    
    # although there was no lockdown over this period curfew was still in effect
    curfew = [
        ('2020-12-15', '2020-04-03'),
        ] 
    for start, end in lockdowns:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()

        # defining condition
        condition = (data['date'].dt.date >= start) & (data['date'].dt.date <= end)

        # append indices for rows that meet the condition
        indices.extend(data[condition].index.tolist())
        data['is_lockdown'] = data.index.isin(indices).astype(int)
    
    for start, end in curfew:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()

        # defining condition
        condition = (data['date'].dt.date >= start) & (data['date'].dt.date <= end)

        # append indices for rows that meet the condition
        indices.extend(data[condition].index.tolist())
        data['is_curfew'] = data.index.isin(indices).astype(int)

    return data

# creating variables for school and public holidays
def is_holiday(data):

    # importing public holidays
    fr_holidays = holidays.France()

    # mapping to data
    data['is_bank_holiday'] = data['date'].apply(lambda x: x in fr_holidays)

    # initializing SchoolHolidayDates object
    school_holidays = SchoolHolidayDates()

    # extract school holidays for Zone C
    years = [2020, 2021]
    school_holidays = SchoolHolidayDates()
    paris_holidays_2020 = school_holidays.holidays_for_year_and_zone(2020, 'C')
    paris_holidays_2021 = school_holidays.holidays_for_year_and_zone(2021, 'C')

    # generate list of all individual holiday dates
    holiday_dates = [holiday['date'] for holiday in paris_holidays_2020.values()]
    holiday_dates.extend([holiday['date'] for holiday in paris_holidays_2021.values()])

    # mapping to data
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

# we used cyclical encoding for hour and weekday variables
def cyclical_encoding(data):
        
        period = np.max(data)
        sin_transform = np.sin(2 * np.pi * data / period)
        cos_transform = np.cos(2 * np.pi * data / period)
        return np.column_stack((sin_transform, cos_transform))
    
def get_estimator():

    # merging weather data
    data_merge = FunctionTransformer(merge_data)

    # merging lockdown and curfew dates
    lockdown_dates = FunctionTransformer(is_within_lockdown_curfew)

    # extracting public and school holidays
    holiday_encoder = FunctionTransformer(is_holiday)

    # encoding dates
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day']

    # encoding weekday and hour
    cycle_encoder = FunctionTransformer(cyclical_encoding)
    cycle_cols = ['hour', 'weekday']

    # using RobustScaler() to prevent overfitting
    num_encoder = RobustScaler()
    num_cols = ['precip', 'humidity']

    # OneHotEncoding categorical variables 
    # (this is not actually necessary as CatBoost does this automatically)
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_cols = ['winddir', 'icon', 'counter_name']

    # passing through binary columns (no encoding)
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

    # initializing CatBoostRegressor with hyperparameters found through random search
    regressor = CatBoostRegressor(subsample=0.9,
                                  learning_rate=0.25421,
                                  iterations=1500,
                                  l2_leaf_reg=1,
                                  depth=8,
                                  border_count=32,
                                  bootstrap_type='Bernoulli',
                                  sampling_frequency='PerTree',
                                  verbose=0,
                                  loss_function='RMSE')

    # defining pipeline
    pipe = make_pipeline(
        data_merge,
        lockdown_dates,
        holiday_encoder,
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe

# selecting features from train and test data
X_train, y_train = train_data[['counter_name', 'date']], train_data[['log_bike_count']]
X_test = test_data[['counter_name', 'date']]
pipe = get_estimator()

# setting negative values to 0
def post_processing(pred):

    pred[(pred <= 0)] = 0

    return pred

# fitting pipeline
pipe.fit(X_train, y_train)

# predicting and post_processing
pred = post_processing(pipe.predict(X_test))

# storing in DataFrame object
pred_df = pd.DataFrame({'Id': X_test.index,
                        'log_bike_count': pred})

# for kaggle submission
pred_df.to_csv('submission.csv', index=False)