# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/28
Description:
"""
import os
import sys

import torch

# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

import random

from hedging_options.library import dataset

from tqdm import tqdm
import resource


def linear_regression_hedge_put_clean_data(_data):
    pos = _data['M'] > 1.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (np.exp(-_data['RisklessRate'] / 100 * _data['RemainingTerm']) *
          _data['StrikePrice'] - _data['UnderlyingScrtClose'] >= _data[
              'ClosePrice'])
    _data = _data.loc[~bl]
    return _data


def linear_regression_hedge_call_clean_data(_data):
    pos = _data['M'] < 0.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (_data['UnderlyingScrtClose'] - np.exp(-_data['RisklessRate'] / 100 *
                                                _data['RemainingTerm']) *
          _data['StrikePrice'] >= _data['ClosePrice'])
    _data = _data.loc[~bl]
    return _data


def add_extra_feature(_data):
    # ClosePrice ,StrikePrice,'Vega', 'Theta', 'Rho','Vega_1', 'Theta_1','Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'
    _scale_rate = _data['UnderlyingScrtClose'] / 100
    for _name in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
                  'UnderlyingScrtClose_1']:
        _data[_name] = _data[_name] / _scale_rate
    _data['UnderlyingScrtClose'] = 100
    _data['delta_bs'] = _data['Delta']
    _data['S0_n'] = 100
    _data['S1_n'] = _data['UnderlyingScrtClose_1']
    _data['V0_n'] = _data['ClosePrice']
    _data['V1_n'] = _data['ClosePrice_1']
    _data['on_ret'] = 1 + _data['RisklessRate'] / 100 * (1 / 253)
    return _data


def get_data(clean_data):
    # _columns = ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
    #             'Delta',
    #             'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #             'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
    train_put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/train_put_data.csv', date_parser=[''])
    train_put_data['Delta'] = -train_put_data['Delta'] - 1

    train_call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/train_call_data.csv')

    test_put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/test_put_data.csv')
    test_put_data['Delta'] = -test_put_data['Delta'] - 1

    test_call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/test_call_data.csv')

    if clean_data:
        train_put_data = linear_regression_hedge_put_clean_data(train_put_data)
        train_call_data = linear_regression_hedge_call_clean_data(train_call_data)
        test_put_data = linear_regression_hedge_put_clean_data(test_put_data)
        test_call_data = linear_regression_hedge_call_clean_data(test_call_data)
    print(test_put_data.shape, test_call_data.shape)
    train_put_data = add_extra_feature(train_put_data)
    train_call_data = add_extra_feature(train_call_data)
    test_put_data = add_extra_feature(test_put_data)
    test_call_data = add_extra_feature(test_call_data)
    return train_put_data, train_call_data, test_put_data, test_call_data


def no_hedge_in_test(clean_data):
    # put_results, call_results = get_test_data(clean_data)
    train_put_data, train_call_data, test_put_data, test_call_data = get_data(clean_data)
    put_results = test_put_data
    call_results = test_call_data
    delta = 0
    put_mshes = np.power((100 * (delta * put_results['S1_n'] + put_results['on_ret'] * (
            put_results['V0_n'] - delta * put_results['S0_n']) - put_results['V1_n'])) / put_results['S1_n'], 2).mean()
    delta = 0
    call_mshes = np.power((100 * (delta * call_results['S1_n'] + call_results['on_ret'] * (
            call_results['V0_n'] - delta * call_results['S0_n']) - call_results['V1_n'])) / call_results['S1_n'], 2).mean()

    print(round(call_mshes, 3), '\t', round(put_mshes, 3), '\t', round((put_mshes + call_mshes) / 2, 3))


def reset_features(df):
    df['M'] = df['UnderlyingScrtClose'] / df['StrikePrice']
    df = df[
        ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm', 'Delta',
         'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
         'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']]
    df = df.replace({'CallOrPut': {'C': 0, 'P': 1}})
    return df


def prepare_data():
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/training.csv')
    valid_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/validation.csv')
    test_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/testing.csv')
    # train_data = train_data.iloc[:500, :]
    # valid_data_index = valid_data_index[:3000, :]
    # test_data_index = test_data_index[:500, :]

    train_data = reset_features(train_data)
    valid_data = reset_features(valid_data)
    test_data = reset_features(test_data)
    train_put_data = train_data[train_data['CallOrPut'] == 1]
    train_call_data = train_data[train_data['CallOrPut'] == 0]
    valid_put_data = valid_data[valid_data['CallOrPut'] == 1]
    valid_call_data = valid_data[valid_data['CallOrPut'] == 0]
    test_put_data = test_data[test_data['CallOrPut'] == 1]
    test_call_data = test_data[test_data['CallOrPut'] == 0]
    train_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/train_put_data.csv', index=False)
    train_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/train_call_data.csv', index=False)
    valid_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/valid_put_data.csv', index=False)
    valid_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/valid_call_data.csv', index=False)
    test_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/test_put_data.csv', index=False)
    test_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/test_call_data.csv', index=False)


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    prepare_data()
    CLEAN_DATA = True
    print(f'no_hedge_in_test , clean={CLEAN_DATA}')
    no_hedge_in_test(CLEAN_DATA)
