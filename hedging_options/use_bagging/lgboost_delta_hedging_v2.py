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
    no_need_columns = ['TradingDate']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    latest_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'MainSign_{i}')
    training_df = training_df.astype({j: int for j in cat_features})
    validation_df = validation_df.astype({j: int for j in cat_features})
    testing_df = testing_df.astype({j: int for j in cat_features})
    latest_df = latest_df.astype({j: int for j in cat_features})
    params = {'objective': 'multiclass',
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
              'metric': 'multi_logloss',
              'num_classes': 3
              }

    num_round = 50000
    early_s_n = 10
    target_fea = 'up_and_down'
    last_x_index = -6
    train_data = lgb.Dataset(training_df.iloc[:, :last_x_index], training_df[target_fea])
    validation_data = lgb.Dataset(validation_df.iloc[:, :last_x_index], validation_df[target_fea])
    bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data],
                    callbacks=[early_stopping(early_s_n), log_evaluation()])
    if opt.log_to_file:

        util.remove_file_if_exists(f'lgboostClassifier')
        bst.save_model('lgboostClassifier', num_iteration=bst.best_iteration)
        bst_from_file = lgb.Booster(model_file='lgboostClassifier')
    else:
        bst_from_file = bst
    y_validation_hat = bst_from_file.predict(validation_df.iloc[:, :last_x_index], num_iteration=bst.best_iteration)
    y_test_hat = bst_from_file.predict(testing_df.iloc[:, :last_x_index], num_iteration=bst.best_iteration)
    y_latest_hat = bst_from_file.predict(latest_df.iloc[:, :last_x_index], num_iteration=bst.best_iteration)

    util.eval_accuracy(validation_df[target_fea], np.argmax(y_validation_hat, axis=1))
    util.eval_accuracy(testing_df[target_fea], np.argmax(y_test_hat, axis=1))
    util.eval_accuracy(latest_df[target_fea], np.argmax(y_latest_hat, axis=1))

    """
    Early stopping, best iteration is:
[4771]	valid_0's auc: 0.99999
预测为1 且实际为1 ，看涨的准确率: 0.9997289421969235
预测为0中实际为1的概率，即期权实际是涨，但是被漏掉的概率 : 0.12194980822642508
    """
