# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse

import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
import numpy as np
import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    return opt


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lgboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    LATEST_DATA_PATH = f'/home/liyu/data/hedging-option/latest-china-market/h_sh_300/'
    latest_df = pd.read_csv(f'{LATEST_DATA_PATH}/{NORMAL_TYPE}/predict_latest.csv')
    no_need_columns = ['TradingDate', 'C_1']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    latest_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y, latest_x, latest_y = util.reformat_data(
        training_df, testing_df, validation_df, latest_df)

    params = {'objective': 'binary',
              # 'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': -1,
              'num_leaves': 2 ** 8,
              'lambda_l1': 0.5,
              'lambda_l2': 0.5,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq': 20,
              'force_col_wise': True,
              # 'metric': 'multi_logloss',
              # 'num_classes': 3
              }

    num_round = 50000
    early_s_n = 10
    train_data = lgb.Dataset(train_x, train_y)
    validation_data = lgb.Dataset(validation_x, validation_y)
    bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data],
                    callbacks=[early_stopping(early_s_n), log_evaluation()])
    if opt.log_to_file:

        util.remove_file_if_exists(f'lgboostClassifier')
        bst.save_model('lgboostClassifier', num_iteration=bst.best_iteration)
        bst_from_file = lgb.Booster(model_file='lgboostClassifier')
    else:
        bst_from_file = bst
    y_validation_hat = bst_from_file.predict(validation_x, num_iteration=bst.best_iteration)
    y_test_hat = bst_from_file.predict(testing_x, num_iteration=bst.best_iteration)
    y_latest_hat = bst_from_file.predict(latest_x, num_iteration=bst.best_iteration)

    util.binary_eval_accuracy(validation_y, y_validation_hat > 0.5)
    util.binary_eval_accuracy(testing_y, y_test_hat > 0.5)
    util.binary_eval_accuracy(latest_y, y_latest_hat > 0.5)

