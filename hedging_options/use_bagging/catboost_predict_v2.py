# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool,CatBoostRegressor
from sklearn.metrics import accuracy_score

import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    return parser.parse_args()


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124_0/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('catboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    training_df = testing_df.iloc[0:10,:]
    validation_df = testing_df.iloc[0:10,:]
    no_need_columns = ['TradingDate', 'NEXT_HIGH']
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    # cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y,testing_y_2 = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)


    test_pool = Pool(testing_x, cat_features=cat_features)

    from_file = CatBoostClassifier()

    from_file.load_model("CatBoostClassifier")
    # make the prediction using the resulting model

    y_test_hat = from_file.predict(test_pool)

    y_validation_true = np.array(validation_y).reshape(-1, 1)
    y_test_true = np.array(testing_y).reshape(-1, 1)


    # util.binary_eval_accuracy(y_validation_true, y_validation_hat)
    # print('==========================')
    util.binary_eval_accuracy(y_test_true, y_test_hat)
    util.detail_result_analysis(testing_y_2.to_numpy(), y_test_hat)


"""
0：不涨 ， 1：涨
tn, fp, fn, tp 9595 1734 7384 2557
test中为1的比例 : 0.4673718852844382
test中为0的比例 : 0.5326281147155618
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.5958983919832207
查全率 - 实际为1，预测为1 : 0.25721758374409015
F1 = 0.3593310848791456
总体准确率：0.5713211095439586
"""

