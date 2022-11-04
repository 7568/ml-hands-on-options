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

    """
    Early stopping, best iteration is:
[225]	valid_0's binary_logloss: 0.633747
0：不涨 ， 1：涨
tn, fp, fn, tp 20500 4228 11265 8039
test中为1的比例 : 0.43840843023255816
test中为0的比例 : 0.5615915697674418
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6553354528409554
查全率 - 实际为1，预测为1 : 0.4164421881475342
F1 = 0.5092648316492984
总体准确率：0.6481422601744186
0：不涨 ， 1：涨
tn, fp, fn, tp 23228 3897 10931 8438
test中为1的比例 : 0.4165913881361036
test中为0的比例 : 0.5834086118638964
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6840697203080665
查全率 - 实际为1，预测为1 : 0.4356445867107233
F1 = 0.5322987635629574
总体准确率：0.6810771282315998
0：不涨 ， 1：涨
tn, fp, fn, tp 4249 852 2466 1551
test中为1的比例 : 0.4405571397236236
test中为0的比例 : 0.5594428602763764
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6454431960049938
查全率 - 实际为1，预测为1 : 0.3861090365944735
F1 = 0.48317757009345796
总体准确率：0.6361044088615925
    """
