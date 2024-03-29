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

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)

def distill_put_clean_data(_data):
    pos = _data['Mo'] > 1.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (np.exp(-_data['RisklessRate'] / 100 * _data['RemainingTerm']) *
          _data['StrikePrice'] - _data['UnderlyingScrtClose'] >= _data[
              'ClosePrice'])
    _data = _data.loc[~bl]
    return _data


def distill_call_clean_data(_data):
    pos = _data['Mo'] < 0.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (_data['UnderlyingScrtClose'] - np.exp(-_data['RisklessRate'] / 100 *
                                                _data['RemainingTerm']) *
          _data['StrikePrice'] >= _data['ClosePrice'])
    _data = _data.loc[~bl]
    return _data

def go_back_old_value(normal_type, column, _data):
    if normal_type == 'mean_norm':
        mean_normalization_info = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/mean_normalization_info.csv')
        _close_old_info = mean_normalization_info[mean_normalization_info['column'] == column]
        std = _close_old_info['std'].values[0]
        mean = _close_old_info['mean'].values[0]
        underlying_scrt_close_old = _data[column] * std + mean
        return underlying_scrt_close_old
    else:
        mean_normalization_info = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/min_max_normalization_info.csv')
        _close_old_info = mean_normalization_info[mean_normalization_info['column'] == column]
        max = _close_old_info['max'].values[0]
        min = _close_old_info['min'].values[0]
        underlying_scrt_close_old = _data[column] * (max - min) + min
        return underlying_scrt_close_old


def add_extra_feature(normal_type, _data):
    # ClosePrice ,StrikePrice,'Vega', 'Theta', 'Rho','Vega_1', 'Theta_1','Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'
    underlying_scrt_close_old = go_back_old_value(normal_type, 'UnderlyingScrtClose', _data)
    _scale_rate = underlying_scrt_close_old / 100
    # for _name in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
    #               'UnderlyingScrtClose_1']:
    #     _data[_name] = _data[_name] / _scale_rate
    # _data['UnderlyingScrtClose'] = 100
    _data['S0_n'] = 100
    _data['S1_n'] = go_back_old_value(normal_type, 'UnderlyingScrtClose_1', _data) / _scale_rate
    _data['V0_n'] = go_back_old_value(normal_type, 'ClosePrice', _data) / _scale_rate
    _data['V1_n'] = go_back_old_value(normal_type, 'ClosePrice_1', _data) / _scale_rate
    # _data['on_ret'] = 1 + _data['RisklessRate'] / 100 * (1 / 253)
    _data['Delta'] = go_back_old_value(normal_type, 'Delta', _data)
    return _data


def get_data(normal_type, tag, clean_data):
    # _columns = ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
    #             'Delta',
    #             'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #             'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
    _put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/{tag}_put_data.csv', date_parser=[''])
    _put_data['Delta'] = -_put_data['Delta'] - 1

    _call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/{tag}_call_data.csv')

    if clean_data:
        _put_data = distill_put_clean_data(_put_data)
        _call_data = distill_call_clean_data(_call_data)
    _put_data = add_extra_feature(normal_type, _put_data)
    _call_data = add_extra_feature(normal_type, _call_data)
    return _put_data, _call_data


def no_hedge_result(NORMAL_TYPE,tag, clean_data):
    put_data, call_data = get_data(NORMAL_TYPE,tag, clean_data)
    show_hedge_result(put_data, 0, call_data, 0)

def get_hedge_result(_results, _delta):
    put_mshes = np.power((100 * (_delta * _results['S1_n'] + _results['on_ret'] * (
            _results['V0_n'] - _delta * _results['S0_n']) - _results['V1_n'])) / _results['S1_n'],
                         2).mean()

    return round(put_mshes, 3)

def show_hedge_result(put_results, put_delta, call_results, call_delta):
    put_mshes = np.power((100 * (put_delta * put_results['S1_n'] + put_results['on_ret'] * (
            put_results['V0_n'] - put_delta * put_results['S0_n']) - put_results['V1_n'])) / put_results['S1_n'],
                         2).mean()
    call_mshes = np.power((100 * (call_delta * call_results['S1_n'] + call_results['on_ret'] * (
            call_results['V0_n'] - call_delta * call_results['S0_n']) - call_results['V1_n'])) / call_results['S1_n'],
                          2).mean()

    print('call_mshes',round(call_mshes, 3), '\t', 'put_mshes',round(put_mshes, 3), '\t', 'mean',round((put_mshes + call_mshes) / 2, 3))

# def show_hedge_result2(_results, _delta):
#     put_mshes = np.power((100 * (_delta * _results['S1_n'] + _results['on_ret'] * (
#             _results['V0_n'] - _delta * _results['S0_n']) - _results['V1_n'])) / _results['S1_n'],
#                          2).mean()
#
#     print( '_mshes',round(put_mshes, 3))

def reset_features(df):
    df['M'] = df['UnderlyingScrtClose'] / df['StrikePrice']
    df = df[
        ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm', 'Delta',
         'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
         'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']]
    # df = df.replace({'CallOrPut': {'C': 0, 'P': 1}})
    return df


# 数据集总共时长为 599 天，那么训练集数据为 599*0.8 , 验证集和测试集分别为60天
# 于是我们将599天分成120份，则每份有5天(最后一份4天)，在这5天中，最后一天为训练集或者测试集，所以总共有119天是训练集和测试集
# 然后从191天中随机取出一半为验证集，剩余的为测试集，得到59天验证集，60天测试集
def split_training_validation_test():
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/two_day_all_clean_data.csv')
    print(len(df['SecurityID'].unique()))
    trading_date = df.sort_values(by=['TradingDate'])['TradingDate'].unique()

    print(len(trading_date))
    validation_testing_index = np.arange(4, 599, 5)
    np.random.shuffle(validation_testing_index)
    print(validation_testing_index)
    validation_index = validation_testing_index[:int(len(validation_testing_index) / 2)]
    testing_index = validation_testing_index[int(len(validation_testing_index) / 2):]
    training_index = np.arange(0, 600).reshape(120, 5)[:, :4].flatten()
    traning_day = trading_date[training_index]
    validation_day = trading_date[validation_index]
    testing_day = trading_date[testing_index]
    training_set = df[df['TradingDate'].isin(traning_day)]
    validation_set = df[df['TradingDate'].isin(validation_day)]
    test_set = df[df['TradingDate'].isin(testing_day)]

    print(training_set.shape[0])
    print(validation_set.shape[0])
    print(test_set.shape[0])
    if (training_set.shape[0] + validation_set.shape[0] + test_set.shape[0]) != df.shape[0]:
        print('error')
    training_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/training.csv', index=False)
    validation_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/validation.csv', index=False)
    test_set.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/testing.csv', index=False)


def prepare_data(normal_type):
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training.csv')
    valid_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation.csv')
    test_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing.csv')
    # train_data = train_data.iloc[:500, :]
    # valid_data_index = valid_data_index[:3000, :]
    # test_data_index = test_data_index[:500, :]

    # train_data = reset_features(train_data)
    # valid_data = reset_features(valid_data)
    # test_data = reset_features(test_data)
    train_put_data = train_data[train_data['CallOrPut'] == 1]
    train_call_data = train_data[train_data['CallOrPut'] == 0]
    valid_put_data = valid_data[valid_data['CallOrPut'] == 1]
    valid_call_data = valid_data[valid_data['CallOrPut'] == 0]
    test_put_data = test_data[test_data['CallOrPut'] == 1]
    test_call_data = test_data[test_data['CallOrPut'] == 0]
    train_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training_put_data.csv', index=False)
    train_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training_call_data.csv', index=False)
    valid_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation_put_data.csv', index=False)
    valid_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation_call_data.csv', index=False)
    test_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing_put_data.csv', index=False)
    test_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing_call_data.csv', index=False)



PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'mean_norm'
    # prepare_data()
    CLEAN_DATA = False
    print(f'no_hedge_in_test , clean={CLEAN_DATA}')
    # no_hedge_in_training(CLEAN_DATA)
    no_hedge_result(NORMAL_TYPE,'training', CLEAN_DATA)
    no_hedge_result(NORMAL_TYPE,'validation', CLEAN_DATA)
    no_hedge_result(NORMAL_TYPE,'testing', CLEAN_DATA)
