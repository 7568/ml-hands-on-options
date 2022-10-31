# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/11
Description:
"""
import math
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
    plt.close()
    print('save start_date done!')


def test002():
    """
    显示不同日期与存在期权数量的关系
    :return: 生成一个柱状图，横轴日期，纵轴当前日期存在的期权的数量，并保存
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    # df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    trading_dates = df['TradingDate'].unique()
    start_dic = {}
    nums=[]
    for trading_date in tqdm(trading_dates, total=len(trading_dates)):
        _options = df[df['TradingDate'] == trading_date]
        trading_date_str = str(trading_date)
        if start_dic.get(trading_date_str) is None:
            start_dic[trading_date_str] = _options.shape[0]
            nums.append(_options.shape[0])
    nums = np.array(nums)
    print(nums.max())
    print(nums>100)
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
    plt.close()
    print('save date_trade_num done!')


def test003():
    """
    显示期权合约的剩余年限与持仓量量的关系
    :return: 生成一个柱状图，横轴剩余年限，纵轴当剩余年限的期权的持仓量，并根据不同的连续合约标识分别保存
    """
    continue_signs = ['P9801', 'P9802', 'P9803', 'P9804', 'P9805']
    for continue_sign in continue_signs:
        if not os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/{continue_sign}'):
            os.mkdir(f'{DATA_HOME_PATH}/h_sh_300/{continue_sign}')

    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    # df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    option_ids = df['SecurityID'].unique()
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        continue_sign= _options['ContinueSign'].unique()
        if continue_sign.shape[0] > 1:
            raise RuntimeError('continue_sign error')

        sorted_options = _options.sort_values(by='TradingDate')
        position = sorted_options['Position']
        remaining_term = sorted_options['RemainingTerm']
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_title(f'aaa')
        ax.set_xlabel('date')
        ax.set_ylabel('start options number')
        ax.plot(position, remaining_term)
        # ax.tick_params(axis='x', labelrotation=90)
        plt.savefig(f'{DATA_HOME_PATH}/h_sh_300/{continue_sign[0]}/{option_id}_position_remaining.png')
        plt.close()
    print('save position_remaining done!')

def test004():
    """
    分析每个期权有效交易的天数
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    # df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)

    option_ids = df['SecurityID'].unique()

    trade_days = []
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        trade_days.append(_options.shape[0])
    print(trade_days)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_title(f'aaa')
    ax.set_xlabel('date')
    ax.set_ylabel('start options number')
    ax.hist(trade_days,bins=100)

    plt.show()

DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    # test001()
    test002()
    # test003()
    # test004()
