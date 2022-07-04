# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/27
Description:
"""

from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from hedging_options.library import common as cm

from hedging_options.use_bs_delta import get_bS_delta_hedge

from hedging_options.library import dataset

"""
对比两个数据中的两个特征，例如2020年之前的数据和之后的数据
ClosePrice和ClosePrice_1 ，显示他们的柱状图，查看数据的大概分布
查看这两个属性数据的差值的分布
"""


def test_001(df, df1, df2, feature1, feature2):
    fig = plt.figure(figsize=(50, 30))
    i = 3
    j = 5
    k = 1
    _df = {'all_data': df, 'data_before_2020': df1, 'data_after_2020': df2}
    for _k in _df:
        __df = _df[_k]
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.hist(__df[feature1], 100, density=False, facecolor='g', alpha=0.75)
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature2}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.hist(__df[feature2], 100, density=False, facecolor='r', alpha=0.3)
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature1} and {feature2}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.hist(__df[feature1], 100, density=False, facecolor='g', alpha=0.75)
        ax.hist(__df[feature2], 100, density=False, facecolor='r', alpha=0.3)

        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature2}-{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.hist(__df[feature2] - __df[feature1], 100, density=False, facecolor='g', alpha=0.75)

        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_({feature2}-{feature1})/{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.hist((__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01), 100, density=False, facecolor='g',
                alpha=0.75)
        print(((__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01)).to_numpy().max())
    plt.show()


def test_002(df, df1, df2, feature1, feature2):
    fig = plt.figure(figsize=(50, 30))
    i = 3
    j = 5
    k = 1
    _df = {'all_data': df, 'data_before_2020': df1, 'data_after_2020': df2}
    for _k in _df:
        __df = _df[_k]
        data_numbers = __df.shape[0]
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.plot(range(data_numbers), __df[feature1])
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature2}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.plot(range(data_numbers), __df[feature2])
        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature1} and {feature2}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.plot(range(data_numbers), __df[feature1])
        ax.plot(range(data_numbers), __df[feature2])

        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_{feature2}-{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.plot(range(data_numbers), __df[feature2] - __df[feature1])
        # ax.plot(__df[feature2] - __df[feature1], 100, density=False, facecolor='g', alpha=0.75)

        ax = fig.add_subplot(i, j, k)
        k += 1
        ax.set_title(f'{_k}_({feature2}-{feature1})/{feature1}')
        ax.set_xlabel('value')
        ax.set_ylabel('number')
        ax.plot(range(data_numbers), (__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01))
        # ax.plot((__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01), 100, density=False, facecolor='g',
        #         alpha=0.75)
        t = __df[(((__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01)) > 1000)]
        t['rate_payoff'] = (__df[feature2] - __df[feature1]) / (__df[feature1] + 1e-6)
        print(((__df[feature2] - __df[feature1]) / (__df[feature1] + 0.01)).to_numpy().max())
    plt.show()


"""
输出每个期权的走势图
"""

from mpl_toolkits.axes_grid1 import host_subplot
def get_data_by_securityids(param):
    df = param['df']
    feature1 = param['feature1']
    feature2 = param['feature2']
    option_ids = df['SecurityID'].unique()
    if not os.path.exists(f'{SAVE_ROOT_PATH}/(1&_1)/'):
        os.makedirs(f'{SAVE_ROOT_PATH}/(1&_1)/')
    if not os.path.exists(f'{SAVE_ROOT_PATH}/(2-1)/'):
        os.makedirs(f'{SAVE_ROOT_PATH}/(2-1)/')
    if not os.path.exists(f'{SAVE_ROOT_PATH}/(2-1)_1/'):
        os.makedirs(f'{SAVE_ROOT_PATH}/(2-1)_1/')
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _option_datas = df[df['SecurityID'] == option_id]
        data_numbers = _option_datas.shape[0]
        fig = plt.figure(figsize=(10, 10))
        ax = host_subplot(111)
        ax2 = ax.twinx()
        ax.set_xlabel("Day")
        ax.set_ylabel("Value")
        ax2.set_ylabel("Value")
        ax.set_title(f'{option_id}_{_option_datas.iloc[0]["TradingDate"]}')
        p1, = ax.plot(range(data_numbers), _option_datas[feature1]+_option_datas['StrikePrice'], color='red', label=f'{feature1}')
        ax.tick_params(axis='y', labelcolor='red')
        p2, = ax2.plot(range(data_numbers), _option_datas['UnderlyingScrtClose'], color='green',
                       label='UnderlyingScrtClose')
        ax2.tick_params(axis='y', labelcolor='green')
        leg = plt.legend()

        ax.yaxis.get_label().set_color(p1.get_color())
        leg.texts[0].set_color(p1.get_color())

        ax2.yaxis.get_label().set_color(p2.get_color())
        leg.texts[1].set_color(p2.get_color())

        plt.savefig(f'{SAVE_ROOT_PATH}/(1&_1)/{str(option_id)[-5:]}.png')
        plt.close(fig)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _option_datas = df[df['SecurityID'] == option_id]
        data_numbers = _option_datas.shape[0]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'{option_id}_{_option_datas.iloc[0]["TradingDate"]}')
        ax.set_xlabel('day')
        ax.set_ylabel('value')
        ax.plot(range(data_numbers), (_option_datas[feature2] - _option_datas[feature1]))
        plt.savefig(f'{SAVE_ROOT_PATH}/(2-1)/{str(option_id)[-5:]}.png')
        plt.close(fig)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _option_datas = df[df['SecurityID'] == option_id]
        data_numbers = _option_datas.shape[0]
        fig = plt.figure(figsize=(10, 10))


        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_title(f'{option_id}_{_option_datas.iloc[0]["TradingDate"]}')
        # ax.set_xlabel('day')
        # ax.set_ylabel('value')
        # ax.plot(range(data_numbers),
        #         (_option_datas[feature2] - _option_datas[feature1]) / (_option_datas[feature1] + 1e-6), color='red',
        #         label=f'{feature1}')
        # ax.tick_params(axis='y', labelcolor='red')
        # plt.legend()
        # ax2 = ax.twinx()
        # ax2.plot(range(data_numbers),
        #          (_option_datas['UnderlyingScrtClose_1'] - _option_datas['UnderlyingScrtClose']), color='green',
        #          label=f'UnderlyingScrtClose')
        # ax2.tick_params(axis='y', labelcolor='green')
        # # print(_option_datas['UnderlyingScrtClose_1'] - _option_datas['UnderlyingScrtClose'])
        # # print(((_option_datas[feature2] - _option_datas[feature1]) / (_option_datas[feature1] + 1e-6)).to_numpy().max())
        # plt.legend()

        ax = host_subplot(111)
        ax2 = ax.twinx()
        ax.set_xlabel("Day")
        ax.set_ylabel("Value")
        ax2.set_ylabel("Value")
        ax.set_title(f'{option_id}_{_option_datas.iloc[0]["TradingDate"]}')
        p1, = ax.plot(range(data_numbers),
                (_option_datas[feature2] - _option_datas[feature1]) / (_option_datas[feature1] + 1e-6), color='red',
                label=f'(price1-price)/price')
        ax.tick_params(axis='y', labelcolor='red')
        p2, = ax2.plot(range(data_numbers),
                 (_option_datas['UnderlyingScrtClose_1'] - _option_datas['UnderlyingScrtClose']), color='green',
                 label=f'Underprince-Under1price')
        ax2.tick_params(axis='y', labelcolor='green')
        leg = plt.legend()

        ax.yaxis.get_label().set_color(p1.get_color())
        leg.texts[0].set_color(p1.get_color())

        ax2.yaxis.get_label().set_color(p2.get_color())
        leg.texts[1].set_color(p2.get_color())
        plt.savefig(f'{SAVE_ROOT_PATH}/(2-1)_1/{str(option_id)[-5:]}.png')
        plt.close(fig)


def test_003(df, feature1, feature2):
    option_ids = df['SecurityID'].unique()
    cpu_num = cpu_count() - 6
    # cpu_num = 1
    if cpu_num > len(df):
        cpu_num = len(df)
    data_chunks = cm.chunks_np(option_ids, cpu_num)
    param = []
    for data_chunk in data_chunks:
        # print(df[df['SecurityID'].isin(data_chunk)].shape)
        ARGS_ = dict(df=df[df['SecurityID'].isin(data_chunk)], feature1=feature1, feature2=feature2)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(get_data_by_securityids, param)
        p.close()
        p.join()
    print('run done!')


def test_strike_price(df):
    option_ids = df['SecurityID'].unique()
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _option_datas = df[df['SecurityID'] == option_id]
        first_day_data = _option_datas.iloc[0:1]
        # print('======1======')
        # print(f"StrikePrice:{(first_day_data['StrikePrice'])} , UnderlyingScrtClose : {first_day_data['UnderlyingScrtClose']}")
        # print('======2======')
        # print((first_day_data['StrikePrice'] - first_day_data['UnderlyingScrtClose']))
        print('======3======')
        print((100 * (int(first_day_data['UnderlyingScrtClose'] / 100)) - first_day_data['StrikePrice']) / 100)


"""
['index', 'SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode',
       'UnderlyingSecurityID', 'UnderlyingSecuritySymbol', 'ShortName',
       'CallOrPut', 'StrikePrice', 'ExerciseDate', 'ClosePrice',
       'UnderlyingScrtClose', 'RemainingTerm', 'RisklessRate',
       'HistoricalVolatility', 'ImpliedVolatility', 'TheoreticalPrice',
       'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'DataType',
       'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
       'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
"""

PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
SAVE_ROOT_PATH = f'/home/liyu/data/hedging-option/china-market/display/'
if __name__ == '__main__':
    train_data = pd.read_csv('/home/liyu/data/hedging-option/china-market/h_sh_300/all.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])

    print(train_data.columns)
    # train_data = train_data.iloc[:100]
    train_data = train_data.sort_values(by=['TradingDate'])
    train_data_1 = train_data.loc[train_data['TradingDate'] < pd.Timestamp('2020-01-01')]
    train_data_2 = train_data.loc[train_data['TradingDate'] >= pd.Timestamp('2020-01-01')]

    # 显示各个属性的柱状图
    # test_001(train_data, train_data_1, train_data_2, 'ClosePrice', 'ClosePrice_1')
    # test_001(train_data, train_data_1, train_data_2, 'UnderlyingScrtClose', 'UnderlyingScrtClose_1')
    # test_001(train_data, train_data_1, train_data_2, 'Delta', 'Delta_1')
    # test_001(train_data, train_data_1, train_data_2, 'Gamma', 'Gamma_1')
    # test_001(train_data, train_data_1, train_data_2, 'Vega', 'Vega_1')
    # test_001(train_data, train_data_1, train_data_2, 'Theta', 'Theta_1')
    # test_001(train_data, train_data_1, train_data_2, 'Rho', 'Rho_1')
    # test_001(train_data, train_data_1, train_data_2, 'ImpliedVolatility', 'ImpliedVolatility_1')
    # 显示各个属性的线图
    # test_002(train_data, train_data_1, train_data_2, 'ClosePrice', 'ClosePrice_1')
    test_003(train_data, 'ClosePrice', 'ClosePrice_1')

    # 测试行权价
    # test_strike_price(train_data)
