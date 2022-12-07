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


# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20140101-20160229/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20160301-20190531/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190601-20221123/h_sh_300/'
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
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
    # LATEST_DATA_PATH = f'/home/liyu/data/hedging-option/latest-china-market/h_sh_300/'
    # latest_df = pd.read_csv(f'{LATEST_DATA_PATH}/{NORMAL_TYPE}/predict_latest.csv')
    no_need_columns = ['TradingDate', 'C_1']
                       # 'ImpliedVolatility',
                       # 'ImpliedVolatility_1',
                       # 'ImpliedVolatility_2',
                       # 'ImpliedVolatility_3',
                       # 'ImpliedVolatility_4']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    # latest_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y= util.reformat_data(
        training_df, testing_df, validation_df, not_use_pre_data=False)

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
    # y_latest_hat = model_from_file.predict(np.ascontiguousarray(latest_x.to_numpy()))
    util.binary_eval_accuracy(np.array(validation_y), y_validation_hat)
    util.binary_eval_accuracy(np.array(testing_y), y_test_hat)
    # util.binary_eval_accuracy(np.array(latest_y), y_latest_hat)

    """
    0：不涨 ， 1：涨
tn, fp, fn, tp 6024 341 1015 236
test中为1的比例 : 0.1642594537815126
test中为0的比例 : 0.8357405462184874
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.4090121317157712
查全率 - 实际为1，预测为1 : 0.18864908073541167
F1 = 0.25820568927789933
总体准确率：0.821953781512605
0：不涨 ， 1：涨
tn, fp, fn, tp 8212 149 1147 224
test中为1的比例 : 0.14087546239210852
test中为0的比例 : 0.8591245376078915
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6005361930294906
查全率 - 实际为1，预测为1 : 0.16338439095550694
F1 = 0.25688073394495414
总体准确率：0.8668310727496917
    
    
    0：不涨 ， 1：涨
tn, fp, fn, tp 12482 198 3657 277
test中为1的比例 : 0.23678825087275793
test中为0的比例 : 0.7632117491272421
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.5831578947368421
查全率 - 实际为1，预测为1 : 0.07041179461108286
F1 = 0.12565207530052167
总体准确率：0.7679667750090285
0：不涨 ， 1：涨
tn, fp, fn, tp 8416 129 458 113
test中为1的比例 : 0.06263712154453707
test中为0的比例 : 0.937362878455463
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.4669421487603306
查全率 - 实际为1，预测为1 : 0.1978984238178634
F1 = 0.2779827798277983
总体准确率：0.9356077226853883



0：不涨 ， 1：涨
tn, fp, fn, tp 9462 1409 7156 1645
test中为1的比例 : 0.44738714924766165
test中为0的比例 : 0.5526128507523383
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.5386378519973805
查全率 - 实际为1，预测为1 : 0.18691057834337008
F1 = 0.27752003374103756
总体准确率：0.564609597397316
0：不涨 ， 1：涨
tn, fp, fn, tp 8833 2570 7179 2532
test中为1的比例 : 0.4599317988064791
test中为0的比例 : 0.5400682011935208
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.49627597020776165
查全率 - 实际为1，预测为1 : 0.26073524868705594
F1 = 0.3418618780800648
总体准确率：0.5382684474756086

    """
