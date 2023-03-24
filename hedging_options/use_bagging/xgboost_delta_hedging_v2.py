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
from sklearn.metrics import mean_squared_error,f1_score,confusion_matrix,accuracy_score

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    return opt



def mae_loss(y_pred, y_val):
    # f(y_val) = abs(y_val-y_pred)
    grad = np.sign(y_val-y_pred)*np.repeat(1,y_val.shape[0])
    hess = np.repeat(0,y_val.shape[0])
    return grad, hess


def pseudo_huber_loss(y_pred, y_val):
    d = (y_val-y_pred)
    delta = 1
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def f1_eval(y_pred, y_val):
    y_pred = np.array([int(i>0) for i in y_pred])
    f_1 = f1_score(y_val.get_label(),y_pred, average="binary")
    return ('f1_error',1-f_1)

# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20140101-20160229/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20160301-20190531/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190601-20221123/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124_0.05/h_sh_300/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20140101-20220321/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('xgboost_delta_hedging_v2')
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    no_need_columns = ['TradingDate', 'NEXT_HIGH']
    # no_need_columns = ['TradingDate', 'C_1','SecurityID', 'Filling', 'ContinueSign', 'TradingDayStatusID']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y= util.reformat_data(
        training_df,validation_df, testing_df, not_use_pre_data=False)

    params = {
        'objective': 'binary:logistic',
        # 'objective': 'reg:squarederror',
        # 'objective': util.mse_loss,
        # 'objective': mae_loss,
        # 'objective': pseudo_huber_loss,
        'n_estimators': 500,
        'max_depth': 30,
        'learning_rate': 0.01,
        # 'tree_method': 'hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'use_label_encoder': False,
        # 'disable_default_eval_metric': 1
        # 'eval_metric': '1-f1'

    }

    model = xgb.XGBClassifier(**params)

    model.fit(train_x.to_numpy(), train_y,
              eval_set=[(validation_x.to_numpy(), np.array(validation_y))],
              early_stopping_rounds=20,eval_metric=f1_eval)
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
    print('======================')
    util.binary_eval_accuracy(np.array(testing_y), y_test_hat)
    # util.binary_eval_accuracy(np.array(latest_y), y_latest_hat)

"""
   

"""