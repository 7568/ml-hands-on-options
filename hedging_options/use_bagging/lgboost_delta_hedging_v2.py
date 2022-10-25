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

    params = {'objective': 'regression',
              'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': -1,
              'num_leaves': 2 ** 8,
              'lambda_l1': 0.5,
              'lambda_l2': 0.5,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq ': 1,
              'metric': {'l2'}
              }

    train_data = lgb.Dataset(training_df.iloc[:, :-1],
                             np.array(training_df['target']).reshape(-1, 1))
    num_round = 50000
    validation_data = lgb.Dataset(validation_df.iloc[:, :-1],
                                  np.array(validation_df['target']).reshape(-1, 1))
    bst = lgb.train(params, train_data, num_round, valid_sets=validation_data, early_stopping_rounds=20)
    y_test_hat = bst.predict(testing_df.iloc[:, :-1], num_iteration=bst.best_iteration)

    error_in_test = mean_squared_error(y_test_hat, np.array(testing_df['target']).reshape(-1, 1))
    print(f'error_in_test : {error_in_test}')
    # result 0.4315025889686754
