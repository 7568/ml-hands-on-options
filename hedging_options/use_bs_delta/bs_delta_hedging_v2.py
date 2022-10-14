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

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))

# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/h_sh_300/'
if __name__ == '__main__':
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')

    min_error = math.inf
    scale_rate = 0
    search_list = np.arange(-2, -1, 0.001)
    for i in search_list:
        target_hat = i * training_df['OpenPrice'] * training_df['Delta'] * training_df['CallOrPut']
        error = np.mean(np.array(target_hat - training_df['target']) ** 2)
        if error < min_error:
            min_error = error
            scale_rate = i

    print(f'scale_rate ： {scale_rate} , min_error : {min_error}')
    target_hat = scale_rate * testing_df['OpenPrice'] * testing_df['Delta'] * testing_df['CallOrPut']
    error_in_test = mean_squared_error(target_hat , testing_df['target'])
    print(f'error_in_test : {error_in_test}')
    # result scale_rate ： -1.2179000000000861 , min_error : 0.8927224927963194 , error_in_test : 0.93137765978801
