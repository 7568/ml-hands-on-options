# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/5/31
Description:
"""
import os
import sys

# Append the library path to PYTHONPATH, so library can be imported.
sys.path.append(os.path.dirname(os.getcwd()))
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import random
from hedging_options.library import dataset

from tqdm import tqdm

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)


# /home/liyu/data/hedging-option/china-market/parquet

# 将训练数据存放成 parquet 格式，方便训练的时候获取

def prepare_dataset(tag):
    save_path = f'/home/liyu/data/hedging-option/china-market/parquet/{tag}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_index = pd.read_csv(f'/home/liyu/data/hedging-option/china-market/h_sh_300_{tag}_index.csv')
    train_data = pd.read_csv('/home/liyu/data/hedging-option/china-market/h_sh_300.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])

    _dataset = dataset.PrepareChineDataSet(train_data, data_index, 15, save_path)

    # Create data loaders.
    dataloader = DataLoader(_dataset, num_workers=50, batch_size=1)
    for ii, (_, __) in tqdm(enumerate(dataloader), total=data_index.shape[0]):
        continue


def test_001():
    prepare_dataset('training')
    prepare_dataset('validation')
    prepare_dataset('test')

def add_extra_feature(_data):
    # ClosePrice ,StrikePrice,'Vega', 'Theta', 'Rho','Vega_1', 'Theta_1','Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'
    _scale_rate = _data['UnderlyingScrtClose'] / 100
    # for _name in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
    #               'UnderlyingScrtClose_1']:
    #     _data[_name] = _data[_name] / _scale_rate
    # _data['UnderlyingScrtClose'] = 100
    _data['S0_n'] = 100
    _data['S1_n'] = _data['UnderlyingScrtClose_1'] / _scale_rate
    _data['V0_n'] = _data['ClosePrice'] / _scale_rate
    _data['V1_n'] = _data['ClosePrice_1'] / _scale_rate
    _data['On_ret'] = 1 + _data['RisklessRate'] / 100 * (1 / 253)
    return _data


# 将每一天的数据存放成 parquet 格式
def prepare_dataset_for_panel_data(normal_type, tag):
    if not os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{normal_type}/{tag}/'):
        os.makedirs(f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{normal_type}/{tag}/')

    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/{normal_type}/{tag}.csv', parse_dates=['TradingDate'])
    days = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    all_nums = []
    for day in tqdm(days, total=len(days)):
        _options = df[df['TradingDate'] == day]
        all_nums.append(_options.shape[0])
        add_extra_feature(_options)
        _options = _options.drop(columns=['SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode', 'UnderlyingSecurityID',
                                          'UnderlyingSecuritySymbol', 'ShortName', 'DataType', 'HistoricalVolatility',
                                          'ImpliedVolatility', 'TheoreticalPrice', 'ExerciseDate',
                                          'ImpliedVolatility_1',
                                          'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1', 'Rho_1', 'index', 'ClosePrice_1',
                                          'UnderlyingScrtClose_1', ])
        _options.to_parquet(
            f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{normal_type}/{tag}/{str(day)[:10]}_datas.parquet')

    print(np.array(all_nums).min())


def test_002():
    normal_type = 'min_max_norm'
    prepare_dataset_for_panel_data(normal_type, 'training')
    prepare_dataset_for_panel_data(normal_type, 'validation')
    prepare_dataset_for_panel_data(normal_type, 'testing')

    normal_type = 'mean_norm'
    prepare_dataset_for_panel_data(normal_type, 'training')
    prepare_dataset_for_panel_data(normal_type, 'validation')
    prepare_dataset_for_panel_data(normal_type, 'testing')


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    test_002()
