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
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')

    params = {
        'iterations': 2000,
        'depth': 12,
        'learning_rate': 0.01,
        'loss_function': 'RMSE',
        'verbose': True,
        'task_type': "GPU",
        'devices': '0',
        'early_stopping_rounds': 20

    }
    train_pool = Pool(training_df.iloc[:, :-1],
                      np.array(training_df['target']).reshape(-1, 1),
                      cat_features=['CallOrPut', 'MainSign'])
    validation_pool = Pool(validation_df.iloc[:, :-1],
                      np.array(validation_df['target']).reshape(-1, 1),
                      cat_features=['CallOrPut', 'MainSign'])
    test_pool = Pool(testing_df.iloc[:, :-1],
                      cat_features=['CallOrPut', 'MainSign'])
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=validation_pool)
    # make the prediction using the resulting model
    y_test_hat = model.predict(test_pool)

    error_in_test = mean_squared_error(y_test_hat, np.array(testing_df['target']).reshape(-1, 1))
    print(f'error_in_test : {error_in_test}')
    # result 0.40784974689433334
