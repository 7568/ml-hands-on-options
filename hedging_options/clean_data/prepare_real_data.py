# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/2
Description:
"""

import pandas as pd
import numpy as np

import sys
import os

from library import common as cm
from library import loader_aux as laux
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def get_data_by_securityids(param):
    securityids = param['securityids']
    df = param['df']
    new_datas = pd.DataFrame(columns=df.columns)
    for securityid in tqdm(securityids, total=len(securityids)):
        _option_list = df[df['SecurityID'] == securityid]
        _option_list = _option_list.sort_values(by=['TradingDate'])
        _new_datas = pd.DataFrame(columns=df.columns)
        i = 0
        for _day_option in _option_list.to_numpy():
            if pd.isna(_day_option).any():
                _new_datas.append(_option_list.iloc[:i])
                break
            i = i + 1
        if _new_datas.shape[0] > 50:
            new_datas.append(_new_datas)
    return new_datas


def run_still():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    print(df.columns)
    print(df['UnderlyingSecurityID'].unique())
    # 204000000140 沪深300 ,  204000000015 上证50
    print(len(df['SecurityID'].unique()))  # 7874

    print(f'204000000140 len : {len(df[df["UnderlyingSecurityID"] == 204000000140])}')  # 5236
    print(f'204000000015 len : {len(df[df["UnderlyingSecurityID"] == 204000000015])}')  # 5236
    print(len(df.loc[df['UnderlyingSecurityID'] == 204000000140, 'SecurityID'].unique()))  # 5236
    print(len(df.loc[df['UnderlyingSecurityID'] == 204000000015, 'SecurityID'].unique()))  # 5236

    # 找出每个期权的时间线，如果在某个时间时候，存在某些参数为NaN，那么就将这个时刻，和之后时刻的数据都删除
    df = df.loc[df['UnderlyingSecurityID'] == 204000000140]
    securityids = df.loc[df['UnderlyingSecurityID'] == 204000000140, 'SecurityID']

    cpu_num = cpu_count() - 6
    # cpu_num = 1
    if cpu_num > len(df):
        cpu_num = len(df)
    data_chunks = cm.chunks(securityids, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(securityids=data_chunk, df=df)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(get_data_by_securityids, param)
        p.close()
        p.join()
    print('run done!')

    new_df = pd.DataFrame()
    for _r in r:
        new_df = new_df.append(_r)

    new_df.to_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_2.csv')
    exit(0)


# 找出dirt_data中的数据对于的在交易基本信息表中的对应数据，产看其交易量和持仓量
def check_data_2():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/dirt_data.csv', index_col=0)
    io_quotationbas = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_QUOTATIONBAS.csv')
    df = df.reset_index()
    tmp = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        _tmp = io_quotationbas[(io_quotationbas['SecurityID'] == row['SecurityID']) & (io_quotationbas[
                                                                                           'TradingDate'] == row[
                                                                                           'TradingDate'])]
        tmp.append(_tmp.iloc[0].to_numpy())
    dirt_data = pd.DataFrame(tmp, columns=io_quotationbas.columns)
    dirt_data.to_csv('/home/liyu/data/hedging-option/china-market/dirt_data_2.csv', index=False)


# 查看一下 dirt_data 中的数据在重要参数表中，处在期权的整个持续期的哪个阶段
def check_dirt_data_period():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_3.csv', index_col=0)
    print(len(df['SecurityID'].unique()))
    dirt_data = pd.read_csv('/home/liyu/data/hedging-option/china-market/dirt_data.csv', index_col=0)
    dirt_data = dirt_data.reset_index()
    for index, row in tqdm(dirt_data.iterrows(), total=dirt_data.shape[0]):
        _options = df[df['SecurityID'] == row['SecurityID']].reset_index()
        _options = _options.sort_values(by=['TradingDate'])
        _index = _options.index[_options['TradingDate'] == row['TradingDate']]
        print(f'{_options.shape[0]} , {_index}')


# 找出那些在重要参数表中剩余的还有空值的数据
def check_data():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_4.csv', index_col=0)
    print(len(df['SecurityID'].unique()))
    # df = df.drop(columns=['ImpliedVolatility', 'DataType'])
    df = df.drop(columns=['DataType'])
    df = df.reset_index()
    i = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.isna().any():
            i = i + 1
            _options = df[df['SecurityID'] == row['SecurityID']].reset_index()
            _options = _options.sort_values(by=['TradingDate'])
            _index = _options.index[_options['TradingDate'] == row['TradingDate']]
            print(f'{_options.shape[0]} , {_index}')
    print(i)  # 我们最终发现有 25223 条数据没有 ImpliedVolatility 信息。


# 找出那些在重要参数表中剩余的还有空值的数据
def check_dirt_data():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_4.csv', index_col=0)
    print(len(df['SecurityID'].unique()))
    df = df.drop(columns=['ImpliedVolatility', 'DataType'])
    df = df.reset_index()
    i = 0
    _dirt_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.isna().any():
            i = i + 1
            _dirt_data.append(row.to_numpy())
    print(i)  # 还剩余632条数据，这些数据当天是有交易的，所以称为脏数据
    dirt_data = pd.DataFrame(_dirt_data, columns=df.columns)
    dirt_data.to_csv('/home/liyu/data/hedging-option/china-market/dirt_data.csv', index=False)


# IO_PRICINGPARAMETER 是	股指期权合约定价重要参数表 2017-06-01 至 2022-06-01
# IO_QUOTATIONBAS 是 股指期权合约日交易基础表 	2013-11-29 至 2022-06-01
# 先通过 IO_QUOTATIONBAS 找出那些没有交易量为0的期权，然后从 IO_PRICINGPARAMETER 中删去其信息
def remove_no_volume_data_precess(param):
    t_s = param['dirt_data']
    option_sensibility_df = param['option_sensibility_df']
    t_s = t_s.reset_index()
    for ii, row in tqdm(t_s.iterrows(), total=t_s.shape[0]):
        i = option_sensibility_df[((option_sensibility_df['TradingDate'] == row[0]) &
                                   (option_sensibility_df['SecurityID'] == row[1]))].index
        # print(i)
        if len(i) > 0:
            option_sensibility_df = option_sensibility_df.drop(i)
            # print(option_sensibility_df.shape)
    return option_sensibility_df


def remove_dirt_data_precess(param):
    t_s = param['dirt_data']
    option_sensibility_df = param['option_sensibility_df']
    t_s = t_s.reset_index()
    for ii, row in tqdm(t_s.iterrows(), total=t_s.shape[0]):
        i = option_sensibility_df[((option_sensibility_df['TradingDate'] == row['TradingDate']) &
                                   (option_sensibility_df['SecurityID'] == row['SecurityID']))].index
        # print(i)
        if len(i) > 0:
            option_sensibility_df = option_sensibility_df.drop(i)
            # print(option_sensibility_df.shape)
    return option_sensibility_df


def get_no_volume_data():
    volume_info_df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_QUOTATIONBAS.csv')
    # option_sensibility_df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER.csv')
    # print(f'option_sensibility_df shape : {option_sensibility_df.shape}')
    volume_info_df = volume_info_df.reset_index()
    # t_s = pd.DataFrame(columns=['TradingDate', 'SecurityID'])
    t_s = []
    for index, row in tqdm(volume_info_df.iterrows(), total=volume_info_df.shape[0]):
        # if row['Volume']==0:
        #     print(row)
        if (row['Volume'] < 1) | (row['Position'] < 1):
            t_s.append([row['TradingDate'], row['SecurityID']])
            # i = option_sensibility_df[((option_sensibility_df['TradingDate'] == row['TradingDate']) &
            #                           (option_sensibility_df['SecurityID'] == row['SecurityID']))].index
            # option_sensibility_df.drop(i)
    t_s = pd.DataFrame(t_s, columns=['TradingDate', 'SecurityID'])
    print(t_s.shape)
    t_s.to_csv('/home/liyu/data/hedging-option/china-market/t_s.csv', index=False)


def remove_no_volume_data():
    option_sensibility_df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER.csv')
    print(f'option_sensibility_df shape : {option_sensibility_df.shape}')
    t_s = pd.read_csv('/home/liyu/data/hedging-option/china-market/t_s.csv', index_col=0)
    print(f't_s shape : {t_s.shape}')
    cpu_num = cpu_count() - 6
    data_chunks = cm.chunks(option_sensibility_df, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(option_sensibility_df=data_chunk, t_s=t_s)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(remove_no_volume_data_precess, param)
        p.close()
        p.join()
    print('run done!')
    new_df = pd.DataFrame()
    for _r in r:
        new_df = new_df.append(_r)
    print(f'option_sensibility_df shape : {new_df.shape}')
    new_df.to_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_3.csv', index=False)


def remove_dirt_data():
    option_sensibility_df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_3.csv')
    print(f'option_sensibility_df shape : {option_sensibility_df.shape}')
    dirt_data = pd.read_csv('/home/liyu/data/hedging-option/china-market/dirt_data.csv', index_col=0)
    print(f't_s shape : {dirt_data.shape}')
    cpu_num = cpu_count() - 6
    data_chunks = cm.chunks(option_sensibility_df, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(option_sensibility_df=data_chunk, dirt_data=dirt_data)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(remove_dirt_data_precess, param)
        p.close()
        p.join()
    print('run done!')
    new_df = pd.DataFrame()
    for _r in r:
        new_df = new_df.append(_r)
    print(f'option_sensibility_df shape : {new_df.shape}')
    new_df.to_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_4.csv', index=False)


def assemble_next_day_features_precess(param):
    option_ids = param['option_ids']
    df = param['df']
    new_df = pd.DataFrame(columns=df.columns)
    new_df = new_df.assign(ImpliedVolatility_1=0, Delta_1=0, Gamma_1=0, Vega_1=0, Theta_1=0, Rho_1=0)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id].sort_values(by=['TradingDate']).reset_index()
        if _options.shape[0] < 16:
            continue
        _options = _options.drop(['Unnamed: 0'], axis=1)
        _options_features = _options[
            ['ClosePrice', 'UnderlyingScrtClose', 'ImpliedVolatility','Delta', 'Gamma', 'Vega', 'Theta',
             'Rho']].to_numpy()
        _options_features = np.delete(_options_features, 0, 0)
        _options_features = np.append(_options_features, [[0, 0, 0, 0, 0, 0, 0, 0]], axis=0)
        _options[['ClosePrice_1', 'UnderlyingScrtClose_1', 'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1',
                  'Theta_1', 'Rho_1']] = _options_features
        new_df = new_df.append(_options.iloc[:-1, :])  # 最后一天的数据不需要，因为既做不了训练集也做不了验证集

    return new_df


# 将每一行数据的下一个交易日的数据加到当天来
def assemble_next_day_features():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_4.csv', index_col=0, parse_dates=[
        'TradingDate', 'ExerciseDate'])
    option_ids = df['SecurityID'].unique()
    cpu_num = cpu_count() - 36
    data_chunks = cm.chunks_np(option_ids, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(option_ids=data_chunk, df=df)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(assemble_next_day_features_precess, param)
        p.close()
        p.join()
    print('run done!')
    new_df = pd.DataFrame()
    for _r in r:
        new_df = new_df.append(_r)
    print(f'option_sensibility_df shape : {new_df.shape}')
    new_df.to_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_5.csv', index=False)


def check_features():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_5.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    option_ids = df['SecurityID'].unique()
    print(len(option_ids))
    i = 0
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id].sort_values(by=['TradingDate'])
        # print(_options)
        if _options.shape[0] > 25:
            i = i + 1

    print(i)


# 将上证50和沪深300分别进行存储  204000000140 沪深300 ,  204000000015 上证50
def seperate_underlying_security():
    df = pd.read_csv('/home/liyu/data/hedging-option/china-market/IO_PRICINGPARAMETER_5.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    h_sh_300_array = []
    sh_zh_50_array = []
    df = df.reset_index()
    for ii, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['UnderlyingSecurityID'] == 204000000140:
            h_sh_300_array.append(row.to_numpy())
        elif row['UnderlyingSecurityID'] == 204000000015:
            sh_zh_50_array.append(row.to_numpy())
        else:
            print('error')
    if not (len(h_sh_300_array) + len(sh_zh_50_array) == df.shape[0]):
        print('error')
    pd.DataFrame(h_sh_300_array, columns=df.columns).to_csv(
        '/home/liyu/data/hedging-option/china-market/h_sh_300.csv', index=False)
    pd.DataFrame(sh_zh_50_array, columns=df.columns).to_csv(
        '/home/liyu/data/hedging-option/china-market/sh_zh_50.csv', index=False)
    print('done')


def check_seperate_underlying_security():
    h_sh_300 = pd.read_csv('/home/liyu/data/hedging-option/china-market/h_sh_300.csv')
    sh_zh_50 = pd.read_csv('/home/liyu/data/hedging-option/china-market/sh_zh_50.csv')
    print(h_sh_300.shape)
    print(sh_zh_50.shape)


if __name__ == '__main__':
    # run_still()
    # check_data()
    # check_data_2()
    # check_dirt_data_period()
    # get_no_volume_data()
    # remove_no_volume_data()
    # remove_dirt_data()
    assemble_next_day_features()
    check_features()
    seperate_underlying_security()
    check_seperate_underlying_security()
