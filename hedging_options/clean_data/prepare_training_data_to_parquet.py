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


# 将每一天的数据存放成 parquet 格式
def prepare_dataset_for_panel_data(tag):
    if not os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{tag}/'):
        os.makedirs(f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{tag}/')

    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/{tag}.csv', parse_dates=['TradingDate'])
    days = df.sort_values(by=['TradingDate'])['TradingDate'].unique()
    for day in tqdm(days, total=len(days)):
        _options = df[df['TradingDate'] == day]
        _options.drop(columns=['SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode', 'UnderlyingSecurityID',
                               'UnderlyingSecuritySymbol', 'ShortName', 'DataType', 'HistoricalVolatility',
                               'ImpliedVolatility','TheoreticalPrice'])
        _options.to_parquet(
            f'{DATA_HOME_PATH}/h_sh_300/panel_parquet/{tag}/{str(day)[:10]}_datas.parquet')


def test_002():
    prepare_dataset_for_panel_data('training')
    prepare_dataset_for_panel_data('validation')
    prepare_dataset_for_panel_data('testing')


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    test_002()
