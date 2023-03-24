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
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    no_need_columns = ['TradingDate', 'NEXT_HIGH']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign', 'up_and_down']
    # cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'CallOrPut_{i}')
        cat_features.append(f'MainSign_{i}')
        cat_features.append(f'up_and_down_{i}')
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)

    params = {
        'iterations': 200,
        'depth': 16,
        'learning_rate': 0.01,
        # 'loss_function': '',
        # 'verbose': False,
        'task_type': "GPU",
        'logging_level': 'Verbose',
        'devices': '6',
        'early_stopping_rounds': 20,
        # 'eval_metric':'Accuracy'

    }

    train_pool = Pool(train_x, np.array(train_y).reshape(-1, 1), cat_features=cat_features)
    validation_pool = Pool(validation_x, np.array(validation_y).reshape(-1, 1), cat_features=cat_features)
    test_pool = Pool(testing_x, cat_features=cat_features)


    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validation_pool, log_cerr=sys.stderr, log_cout=sys.stdout)
    if opt.log_to_file:
        util.remove_file_if_exists(f'CatBoostClassifier')
        model.save_model("CatBoostClassifier")

        from_file = CatBoostClassifier()

        from_file.load_model("CatBoostClassifier")
    else:
        from_file = model
    # make the prediction using the resulting model
    y_validation_hat = from_file.predict(validation_pool)
    y_test_hat = from_file.predict(test_pool)

    y_validation_true = np.array(validation_y).reshape(-1, 1)
    y_test_true = np.array(testing_y).reshape(-1, 1)


    util.binary_eval_accuracy(y_validation_true, y_validation_hat)
    print('==========================')
    util.binary_eval_accuracy(y_test_true, y_test_hat)


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

