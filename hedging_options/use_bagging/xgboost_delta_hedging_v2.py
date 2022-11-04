# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import copy
import math
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
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
        logger = util.init_log('xgboost_delta_hedging_v2')
    NORMAL_TYPE = 'mean_norm'
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
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


    params = {
        'n_estimators': 50000,
        'objective': 'binary:logistic',
        'max_depth': 12,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'use_label_encoder': False,
        'eval_metric': 'logloss'

    }

    model = xgb.XGBClassifier(**params)

    model.fit(train_x.to_numpy(), train_y,
              eval_set=[(validation_x.to_numpy(), np.array(validation_y))],
              early_stopping_rounds=20)
    if opt.log_to_file:

        util.remove_file_if_exists(f'XGBClassifier')
        model.save_model('XGBClassifier')
        model_from_file = xgb.XGBClassifier()
        model_from_file.load_model('XGBClassifier')
    else:
        model_from_file = model
    # Predict on x_test
    y_validation_hat = model_from_file.predict(np.ascontiguousarray(validation_x.to_numpy()))
    y_test_hat = model_from_file.predict(np.ascontiguousarray(testing_x.to_numpy()))
    y_latest_hat = model_from_file.predict(np.ascontiguousarray(latest_x.to_numpy()))
    util.binary_eval_accuracy(np.array(validation_y), y_validation_hat)
    util.binary_eval_accuracy(np.array(testing_y), y_test_hat)
    util.binary_eval_accuracy(np.array(latest_y), y_latest_hat)

    """
    0：不涨 ， 1：涨
tn, fp, fn, tp 19185 5543 9993 9311
test中为1的比例 : 0.43840843023255816
test中为0的比例 : 0.5615915697674418
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6268345226874916
查全率 - 实际为1，预测为1 : 0.48233526730211357
F1 = 0.5451724339832543
总体准确率：0.6471656976744186
0：不涨 ， 1：涨
tn, fp, fn, tp 21641 5484 9717 9652
test中为1的比例 : 0.4165913881361036
test中为0的比例 : 0.5834086118638964
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6376849894291755
查全率 - 实际为1，预测为1 : 0.49832206102534976
F1 = 0.5594551514273294
总体准确率：0.673054587688734
0：不涨 ， 1：涨
tn, fp, fn, tp 4115 986 2274 1743
test中为1的比例 : 0.4405571397236236
test中为0的比例 : 0.5594428602763764
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6386954928545254
查全率 - 实际为1，预测为1 : 0.4339058999253174
F1 = 0.5167506670619626
总体准确率：0.6424654529502084
    """
