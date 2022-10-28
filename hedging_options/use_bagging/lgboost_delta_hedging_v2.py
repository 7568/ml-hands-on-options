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
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    no_need_columns = ['TradingDate']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'MainSign_{i}')
    training_df = training_df.astype({j: int for j in cat_features})
    validation_df = training_df.astype({j: int for j in cat_features})
    testing_df = training_df.astype({j: int for j in cat_features})
    params = {'objective': 'regression',
              'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': -1,
              'num_leaves': 2 ** 8,
              'lambda_l1': 0.5,
              'lambda_l2': 0.5,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq': 20,
              'metric': {'l2'},
              'force_col_wise': True,
              }

    num_round = 5000
    early_s_n = 10
    target_fea = 'C_1'
    train_data = lgb.Dataset(training_df.iloc[:, :-3], training_df[target_fea])
    validation_data = lgb.Dataset(validation_df.iloc[:, :-3], validation_df[target_fea])

    bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data], verbose_eval=True,
                    callbacks=[early_stopping(early_s_n)], categorical_feature=cat_features)
    y_test_hat = bst.predict(testing_df.iloc[:, :-3], num_iteration=bst.best_iteration)

    error_in_test = mean_squared_error(y_test_hat, testing_df[target_fea])
    print(f'error_in_test : {error_in_test}')
    # result 0.0041245230662759974
