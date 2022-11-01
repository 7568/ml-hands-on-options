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
    testing_df = pd.read_csv(f'{LATEST_DATA_PATH}/{NORMAL_TYPE}/predict_latest.csv')
    no_need_columns = ['TradingDate']
    training_df.drop(columns=no_need_columns, axis=1, inplace=True)
    validation_df.drop(columns=no_need_columns, axis=1, inplace=True)
    testing_df.drop(columns=no_need_columns, axis=1, inplace=True)
    params = {
        'iterations': 50000,
        'depth': 12,
        'learning_rate': 0.01,
        # 'loss_function': '',
        # 'verbose': False,
        'task_type': "GPU",
        'logging_level':'Verbose',
        'devices': '6',
        'early_stopping_rounds': 5,
        # 'eval_metric':'Accuracy'

    }

    cat_features = ['CallOrPut', 'MainSign']
    # cat_features = ['CallOrPut']
    day_num=4

    for i in range(1, (day_num+1)):
        cat_features.append(f'MainSign_{i}')
    training_df = training_df.astype({j: int for j in cat_features})
    validation_df = validation_df.astype({j: int for j in cat_features})
    testing_df = testing_df.astype({j: int for j in cat_features})
    target_fea = 'up_and_down'
    last_x_index = 36+31*day_num
    print(training_df.columns.size - (36+31*day_num))
    train_pool = Pool(training_df.iloc[:, :last_x_index],
                      np.array(training_df[target_fea]).reshape(-1, 1),
                      cat_features=cat_features)
    validation_pool = Pool(validation_df.iloc[:, :last_x_index],
                           np.array(validation_df[target_fea]).reshape(-1, 1),
                           cat_features=cat_features)
    test_pool = Pool(testing_df.iloc[:, :last_x_index],
                     cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validation_pool,log_cerr=sys.stderr,log_cout=sys.stdout)
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
    y_validation_true=np.array(validation_df[target_fea]).reshape(-1, 1)
    y_test_true=np.array(testing_df[target_fea]).reshape(-1, 1)

    util.eval_accuracy(y_validation_true,y_validation_hat)
    util.eval_accuracy(y_test_true,y_test_hat)
    # acc = accuracy_score(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    # print(f'accuracy_score : {acc}')
    """
    day0:查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.9422520984988332
    查全率 - 实际为1，预测为1 : 0.46299845969536196
    F1 = 0.6209017569226886
    
    day4:查准率 - 预测为1 且实际为1 ，看涨的准确率: 0.985047371473994
    查全率 - 实际为1，预测为1 : 0.7847167550915626
    F1 = 0.8735437286262705
    
    """
