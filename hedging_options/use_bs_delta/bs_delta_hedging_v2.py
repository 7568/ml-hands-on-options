# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/28
Description:
"""
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from hedging_options.use_bagging import util

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20190701-20221124/h_sh_300/'
NORMAL_TYPE = 'mean_norm'
if __name__ == '__main__':

    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    no_need_columns = ['TradingDate', 'C_1']
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

    min_error = math.inf
    scale_rate = 0
    search_list = np.arange(-2, -1, 0.001)
    for i in tqdm(search_list,total=len(search_list)):
        target_hat = i * train_x['OpenPrice'] * train_x['Delta'] * train_x['CallOrPut']
        error = np.mean(np.array(target_hat - train_y) ** 2)
        if error < min_error:
            min_error = error
            scale_rate = i

    print(f'scale_rate ： {scale_rate} , min_error : {min_error}')
    target_hat = scale_rate * testing_x['OpenPrice'] * testing_x['Delta'] * testing_x['CallOrPut']
    error_in_test = mean_squared_error(target_hat, testing_y)
    print(f'error_in_test : {error_in_test}')
    util.binary_eval_accuracy(np.array(testing_y), np.array(target_hat))
    # result scale_rate ： -1.2179000000000861 , min_error : 0.8927224927963194 , error_in_test : 0.93137765978801
