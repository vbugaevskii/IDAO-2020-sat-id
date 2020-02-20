#!/usr/bin/env python3

import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

COLS_TARGET = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
COLS_TARGET_DETREND = [col + '_detrend' for col in COLS_TARGET]
COLS_TARGET_PREDICT = [col + '_predict' for col in COLS_TARGET]
COLS_TARGET_SIM = [col + '_sim' for col in COLS_TARGET]


def pandas_vstack(objs, **params):
    return pd.DataFrame().append(objs)


def pandas_hstack(objs, **params):
    return pd.concat(objs, axis=1, **params)


class TimerContextManager(object):
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        print('Enter block: "{}".'.format(self._name), file=sys.stderr, flush=True)
        self._ts_start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ts_delta = time.time() - self._ts_start
        print('Exit block:  "{}". Elapsed in {:.2f} sec.'.format(self._name, ts_delta), file=sys.stderr, flush=True)
        del self._ts_start


def load_data():
    df_train = pd.read_csv("train.csv", index_col="id", parse_dates=["epoch"])
    df_test = pd.read_csv("test.csv", index_col="id", parse_dates=["epoch"])

    df = pandas_vstack([df_train, df_test], sort=False)
    df.sort_values(by=['sat_id', 'epoch'], inplace=True)
    return df, df_test.index


def prepare_group(group):
    group_new = group.drop_duplicates(subset=['ts_int'])

    period_mask = group[COLS_TARGET].notna().all(axis=1)
    period = np.median(np.diff(group_new.loc[period_mask, 'epoch']))
    period = int(period / np.timedelta64(1, 's'))
    period = '{}S'.format(period)

    epoch_new = pd.date_range(start=group_new['epoch'].iloc[0], periods=group_new.shape[0], freq=period)
    group_new.index = epoch_new

    return group_new


def calculate_predicted_features(df, models_all):
    df_list = []

    for sat_id, group in df.groupby('sat_id'):
        group = group.reset_index()
        group_new = prepare_group(group)

        coef, bias = models_all[sat_id]['trend']
        trend = group_new['ts'].values[:, np.newaxis] * coef[np.newaxis, :] + bias[np.newaxis, :]

        group_new_dt = pd.DataFrame(group_new[COLS_TARGET].values - trend,
                                    columns=COLS_TARGET_DETREND,
                                    index=group_new.index)
        group_new = pandas_hstack([group_new, group_new_dt])

        prediction_ = [ group_new[['ts_int']] ]
        for target in COLS_TARGET:
            target_predict = target + '_predict'
            target_detrend = target + '_detrend'

            mask = group_new.loc[:, target].notna()
            smooth = ExponentialSmoothing(group_new.loc[mask, target_detrend],
                                            trend=None, seasonal='add', seasonal_periods=24)
            smooth = smooth._predict(h=0, **models_all[sat_id]['smooth'][target])

            prediction = smooth.predict(start=group_new.index[0], end=group_new.index[-1])
            prediction = prediction.rename(target_predict)
            prediction_.append(prediction)
        prediction = pandas_hstack(prediction_)
        prediction = pd.merge(group[['ts_int', 'ts']], prediction, on='ts_int')

        trend = prediction['ts'].values[:, np.newaxis] * coef[np.newaxis, :] + bias[np.newaxis, :]            
        prediction[COLS_TARGET_PREDICT] = trend + prediction[COLS_TARGET_PREDICT].values
        prediction.sort_values(by='ts', inplace=True)

        group = pandas_hstack([group, prediction])
        mask = group[COLS_TARGET].isna().any(axis=1)
        group = group.loc[mask]

        df_list.append(group)

    df = pandas_vstack(df_list)
    return df


def split_train_test(df):
    mask = df[COLS_TARGET].isna().any(axis=1)
    df_train, df_test = df[~mask], df[mask]
    return df_train, df_test


if __name__ == '__main__':
    with TimerContextManager("Load extra data"):
        models_all = pickle.load(open("models_selected.pickle", 'rb'))

    with TimerContextManager("Load train and test data"):
        df, submission_index = load_data()

    with TimerContextManager("Calculate predicted features"):
        df['ts'] = df['epoch'].values.astype(np.int64) / 10 ** 9
        df['ts_int'] = df['ts'].astype(int)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = calculate_predicted_features(df, models_all)

    with TimerContextManager("Split data"):
        _, df_test = split_train_test(df)

    with TimerContextManager("Make prediction"):
        y_pred = df_test[COLS_TARGET_PREDICT].values

    with TimerContextManager("Save submission"):
        submission = pd.read_csv("test.csv", index_col="id", parse_dates=["epoch"])
        submission = submission[COLS_TARGET_SIM]
        submission[COLS_TARGET_SIM] = y_pred
        submission.rename(columns=dict(zip(COLS_TARGET_SIM, COLS_TARGET)), inplace=True)
        submission.to_csv("submission.csv", index=True)
