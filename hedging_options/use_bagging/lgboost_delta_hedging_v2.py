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


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124_2/h_sh_300/'
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


    no_need_columns = ['TradingDate', 'NEXT_HIGH']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)

    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, testing_df, validation_df, not_use_pre_data=False)

    params = {'objective': 'binary',
              # 'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': 30,
              # 'num_leaves': 2 ** 8,
              'lambda_l1': 0.5,
              'lambda_l2': 0.5,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq': 20,
              # 'force_col_wise': True,
              # 'metric': 'binary_logloss',
              # 'num_classes': 3
              }

    num_round = 1000
    early_s_n = 20
    train_data = lgb.Dataset(train_x, train_y)
    validation_data = lgb.Dataset(validation_x, validation_y)
    bst = lgb.train(params, train_data, num_round,valid_sets=[validation_data],
                    callbacks=[early_stopping(early_s_n), log_evaluation()])
    if opt.log_to_file:

        util.remove_file_if_exists(f'lgboostClassifier')
        bst.save_model('lgboostClassifier', num_iteration=bst.best_iteration)
        bst_from_file = lgb.Booster(model_file='lgboostClassifier')
    else:
        bst_from_file = bst
    y_validation_hat = bst_from_file.predict(validation_x, num_iteration=bst.best_iteration)
    y_test_hat = bst_from_file.predict(testing_x, num_iteration=bst.best_iteration)


    util.binary_eval_accuracy(validation_y, y_validation_hat > 0.5)
    util.binary_eval_accuracy(testing_y, y_test_hat > 0.5)


"""
    0：不涨 ， 1：涨
tn, fp, fn, tp 11181 148 9596 345
test中为1的比例 : 0.4673718852844382
test中为0的比例 : 0.5326281147155618
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6997971602434077
查全率 - 实际为1，预测为1 : 0.03470475807262851
F1 = 0.06612995974698102
总体准确率：0.5418899858956276
"""
