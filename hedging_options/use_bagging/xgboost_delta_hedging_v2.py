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
    normal_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/normal_data.csv')
    no_need_columns = ['TradingDate']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'MainSign_{i}')
    training_df = training_df.astype({j: int for j in cat_features})
    validation_df = validation_df.astype({j: int for j in cat_features})
    testing_df = testing_df.astype({j: int for j in cat_features})
    params = {
        'n_estimators': 500,
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
    target_fea = 'up_and_down'
    last_x_index = -6
    model = xgb.XGBClassifier(**params)
    model.fit(training_df.iloc[:, :last_x_index].to_numpy(), np.array(training_df[target_fea]),
              eval_set=[(validation_df.iloc[:, :last_x_index].to_numpy(), np.array(validation_df[target_fea]))],
              early_stopping_rounds=20)
    if opt.log_to_file:

        util.remove_file_if_exists(f'XGBClassifier')
        model.save_model('XGBClassifier')
        model_from_file = xgb.XGBClassifier()
        model_from_file.load_model('XGBClassifier')
    else:
        model_from_file = model
    # Predict on x_test
    y_validation_hat = model_from_file.predict(np.ascontiguousarray(validation_df.iloc[:, :last_x_index].to_numpy()))
    y_test_hat = model_from_file.predict(np.ascontiguousarray(testing_df.iloc[:, :last_x_index].to_numpy()))
    util.eval_accuracy(np.array(validation_df[target_fea]), y_validation_hat)
    util.eval_accuracy(np.array(testing_df[target_fea]), y_test_hat)
    # 预测为1 且实际为1 ，看涨的准确率: 0.9995863067535422
    # 预测为0中实际为1的概率，即期权实际是涨，但是被漏掉的概率 : 0.11983781907155139
