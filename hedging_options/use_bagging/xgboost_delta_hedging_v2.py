# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import copy
import math
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

print(xgb.__version__)

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    NORMAL_TYPE = 'mean_norm'
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    normal_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/normal_data.csv')
    no_need_columns = ['TradingDate']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)

    params = {
        'max_depth': 12,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_estimators': 10,
        # 'val_metric' : mean_squared_error,

    }
    target_fea = 'C_1'
    model = xgb.XGBRegressor(**params)
    model.fit(training_df.iloc[:, :-3].to_numpy(), np.array(training_df[target_fea]).reshape(-1, 1),
              eval_set=[(validation_df.iloc[:, :-3].to_numpy(), np.array(validation_df[target_fea]).reshape(-1, 1))],
              early_stopping_rounds=20, eval_metric='rmse')

    # Predict on x_test
    y_test_hat = model.predict(np.ascontiguousarray(testing_df.iloc[:, :-3].to_numpy()))

    error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'error_in_test : {error_in_test}')
    #  error_in_test : 0.04834734940222341
