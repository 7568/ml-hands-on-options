# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    return parser.parse_args()


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('catboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
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

    params = {
        'iterations': 50000,
        'depth': 12,
        'learning_rate': 0.01,
        # 'loss_function': '',
        # 'verbose': False,
        'task_type': "GPU",
        'logging_level': 'Verbose',
        'devices': '6',
        'early_stopping_rounds': 5,
        # 'eval_metric':'Accuracy'

    }

    train_pool = Pool(train_x, np.array(train_y).reshape(-1, 1), cat_features=cat_features)
    validation_pool = Pool(validation_x, np.array(validation_y).reshape(-1, 1), cat_features=cat_features)
    test_pool = Pool(testing_x, cat_features=cat_features)
    latest_pool = Pool(latest_x, cat_features=cat_features)

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
    y_latest_hat = from_file.predict(latest_pool)
    y_validation_true = np.array(validation_y).reshape(-1, 1)
    y_test_true = np.array(testing_y).reshape(-1, 1)
    y_latest_true = np.array(latest_y).reshape(-1, 1)

    util.binary_eval_accuracy(y_validation_true, y_validation_hat)
    util.binary_eval_accuracy(y_test_true, y_test_hat)
    util.binary_eval_accuracy(y_latest_true, y_latest_hat)

    """
    bestTest = 0.6277074592
bestIteration = 165
Shrink model to first 166 iterations.
0：不涨 ， 1：涨
tn, fp, fn, tp 19658 5070 10267 9037
test中为1的比例 : 0.43840843023255816
test中为0的比例 : 0.5615915697674418
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6406039554830935
查全率 - 实际为1，预测为1 : 0.4681413178615831
F1 = 0.5409595642153782
总体准确率：0.6516851380813954
0：不涨 ， 1：涨
tn, fp, fn, tp 22370 4755 10132 9237
test中为1的比例 : 0.4165913881361036
test中为0的比例 : 0.5834086118638964
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6601629502572899
查全率 - 实际为1，预测为1 : 0.47689607104135473
F1 = 0.553760378885525
总体准确率：0.6798081472878221
0：不涨 ， 1：涨
tn, fp, fn, tp 4177 924 2377 1640
test中为1的比例 : 0.4405571397236236
test中为0的比例 : 0.5594428602763764
查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.6396255850234009
查全率 - 实际为1，预测为1 : 0.40826487428429176
F1 = 0.49840449779668744
总体准确率：0.6379688528186006
    """

