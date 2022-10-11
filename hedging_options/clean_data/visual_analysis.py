# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/11
Description:
"""

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from hedging_options.library import common as cm


def test001():
    '''
    显示新开期权合约的日期与数量的关系
    :return: 生成一个柱状图，横轴日期，纵轴新开期权的数量，并保存
    '''
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    option_ids = df['SecurityID'].unique()
    start_dic = {}
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        sorted_options = _options.sort_values(by='TradingDate')
        start_date = str(sorted_options.iloc[0]['TradingDate'])
        if start_dic.get(start_date) is None:
            start_dic[start_date] = 1
        else:
            start_dic[start_date] += 1
    fig = plt.figure(figsize=(100, 50))
    ax = fig.add_subplot()
    ax.set_title(f'aaa')
    ax.set_xlabel('date')
    ax.set_ylabel('start options number')
    keys = []
    values = []
    for i in start_dic.keys():
        keys.append(i)
        values.append(start_dic[i])
    ax.bar(keys, values)
    ax.tick_params(axis='x', labelrotation=90)
    plt.savefig(f'{DATA_HOME_PATH}/h_sh_300/start_date.png')

    print('save start_date done!')


def test002():
    '''
    显示期权合约的日期与存在期权数量的关系
    :return: 生成一个柱状图，横轴日期，纵轴当前日期存在的期权的数量，并保存
    '''
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    trading_dates = df['TradingDate'].unique()
    start_dic = {}
    for trading_date in tqdm(trading_dates, total=len(trading_dates)):
        _options = df[df['TradingDate'] == trading_date]
        trading_date_str = str(trading_date)
        if start_dic.get(trading_date_str) is None:
            start_dic[trading_date_str] = _options.shape[0]
    fig = plt.figure(figsize=(100, 50))
    ax = fig.add_subplot()
    ax.set_title(f'aaa')
    ax.set_xlabel('date')
    ax.set_ylabel('start options number')
    keys = []
    values = []
    for i in start_dic.keys():
        keys.append(i)
        values.append(start_dic[i])
    ax.bar(keys, values)
    ax.tick_params(axis='x', labelrotation=90)
    plt.savefig(f'{DATA_HOME_PATH}/h_sh_300/date_trade_num.png')

    print('save date_trade_num done!')


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    # test001()
    test002()
