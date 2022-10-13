# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/2
Description:
"""

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil
from functools import wraps

from hedging_options.library import common as cm


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
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv', parse_dates=[
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

    new_df.to_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_2.csv')
    exit(0)


# 找出dirt_data中的数据对于的在交易基本信息表中的对应数据，产看其交易量和持仓量
def check_data_2():
    df = pd.read_csv(f'{DATA_HOME_PATH}/dirt_data.csv', index_col=0)
    io_quotationbas = pd.read_csv(f'{DATA_HOME_PATH}/IO_QUOTATIONBAS.csv')
    df = df.reset_index()
    tmp = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        _tmp = io_quotationbas[(io_quotationbas['SecurityID'] == row['SecurityID']) & (io_quotationbas[
                                                                                           'TradingDate'] == row[
                                                                                           'TradingDate'])]
        tmp.append(_tmp.iloc[0].to_numpy())
    dirt_data = pd.DataFrame(tmp, columns=io_quotationbas.columns)
    dirt_data.to_csv(f'{DATA_HOME_PATH}/dirt_data_2.csv', index=False)


# 查看一下 dirt_data 中的数据在重要参数表中，处在期权的整个持续期的哪个阶段
def check_dirt_data_period():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_3.csv', index_col=0)
    print(len(df['SecurityID'].unique()))
    dirt_data = pd.read_csv(f'{DATA_HOME_PATH}/dirt_data.csv', index_col=0)
    dirt_data = dirt_data.reset_index()
    for index, row in tqdm(dirt_data.iterrows(), total=dirt_data.shape[0]):
        _options = df[df['SecurityID'] == row['SecurityID']].reset_index()
        _options = _options.sort_values(by=['TradingDate'])
        _index = _options.index[_options['TradingDate'] == row['TradingDate']]
        print(f'{_options.shape[0]} , {_index}')


# 找出那些在重要参数表中剩余的还有空值的数据
def check_data():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_4.csv', index_col=0)
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
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/dirt_data.csv')
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_clean_data.csv')
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
    dirt_data.to_csv(f'{DATA_HOME_PATH}/h_sh_300/dirt_data.csv', index=False)


# IO_PRICINGPARAMETER 是	股指期权合约定价重要参数表 2017-06-01 至 2022-06-01
# IO_QUOTATIONBAS 是 股指期权合约日交易基础表 	2013-11-29 至 2022-06-01
# 先通过 IO_QUOTATIONBAS 找出那些没有交易量为0的期权，然后从 IO_PRICINGPARAMETER 中删去其信息
def remove_no_volume_data_precess(param):
    t_s = param['t_s']
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
    t_s = param['t_s']
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
    volume_info_df = pd.read_csv(f'{DATA_HOME_PATH}/IO_QUOTATIONBAS.csv')
    # option_sensibility_df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv')
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
    t_s.to_csv(f'{DATA_HOME_PATH}/t_s.csv', index=False)


def remove_no_volume_data():
    option_sensibility_df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv')
    print(f'option_sensibility_df shape : {option_sensibility_df.shape}')
    t_s = pd.read_csv(f'{DATA_HOME_PATH}/t_s.csv', index_col=0)
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
    new_df.to_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_3.csv', index=False)


def remove_dirt_data():
    option_sensibility_df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_3.csv')
    print(f'option_sensibility_df shape : {option_sensibility_df.shape}')
    dirt_data = pd.read_csv(f'{DATA_HOME_PATH}/dirt_data.csv', index_col=0)
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
    new_df.to_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_4.csv', index=False)


def assemble_next_day_features_precess(param):
    option_ids = param['option_ids']
    df = param['df']
    new_df = pd.DataFrame(columns=df.columns)
    new_df = new_df.assign(ImpliedVolatility_1=0, Delta_1=0, Gamma_1=0, Vega_1=0, Theta_1=0, Rho_1=0)
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id].sort_values(by=['TradingDate']).reset_index()
        # _options = _options.drop(['Unnamed: 0'], axis=1)
        _options_features = _options[
            ['ClosePrice', 'UnderlyingScrtClose', 'ImpliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta',
             'Rho']].to_numpy()
        _options_features = np.delete(_options_features, 0, 0)
        _options_features = np.append(_options_features, [[0, 0, 0, 0, 0, 0, 0, 0]], axis=0)
        _options[['ClosePrice_1', 'UnderlyingScrtClose_1', 'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1',
                  'Theta_1', 'Rho_1']] = _options_features

        _options = _options.replace({'CallOrPut': {'C': 0, 'P': 1}})
        _options['on_ret'] = 1 + _options['RisklessRate'] / 100 * (1 / 253)
        _options['Mo'] = _options['UnderlyingScrtClose'] / _options['StrikePrice']
        _options['ActualDelta'] = (_options['ClosePrice_1'] - _options['on_ret'] * _options['ClosePrice']) / (
                _options['UnderlyingScrtClose_1'] - _options['on_ret'] * _options['UnderlyingScrtClose'])
        _options.drop(
            columns=['ClosePrice_1', 'UnderlyingScrtClose_1', 'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1',
                     'Theta_1', 'Rho_1'])
        new_df = new_df.append(_options.iloc[:-1, :])  # 最后一天的数据不需要，因为既做不了训练集也做不了验证集

    return new_df


# 将每一行数据的下一个交易日的数据加到当天来
# def assemble_next_day_features(file_path, file_name):
#     df = pd.read_csv(f'{DATA_HOME_PATH}/{file_path}/{file_name}', parse_dates=[
#         'TradingDate', 'ExerciseDate'])
#     option_ids = df['SecurityID'].unique()
#     cpu_num = cpu_count() - 36
#     data_chunks = cm.chunks_np(option_ids, cpu_num)
#     param = []
#     for data_chunk in data_chunks:
#         ARGS_ = dict(option_ids=data_chunk, df=df)
#         param.append(ARGS_)
#     with Pool(cpu_num) as p:
#         r = p.map(assemble_next_day_features_precess, param)
#         p.close()
#         p.join()
#     print('run done!')
#     new_df = pd.DataFrame()
#     for _r in r:
#         new_df = new_df.append(_r)
#     print(f'option_sensibility_df shape : {new_df.shape}')
#     new_df.to_csv(f'{DATA_HOME_PATH}/{file_path}/two_day_{file_name}', index=False)


# def check_features():
#     df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_5.csv', parse_dates=[
#         'TradingDate', 'ExerciseDate'])
#     option_ids = df['SecurityID'].unique()
#     print(len(option_ids))
#     i = 0
#     for option_id in tqdm(option_ids, total=len(option_ids)):
#         _options = df[df['SecurityID'] == option_id].sort_values(by=['TradingDate'])
#         # print(_options)
#         if _options.shape[0] > 25:
#             i = i + 1
#
#     print(i)


# 将上证50和沪深300分别进行存储  204000000140 沪深300 ,  204000000015 上证50
def seperate_underlying_security():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_3.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    df = df.reset_index()
    h_sh_300 = df[df['UnderlyingSecurityID'] == 204000000140]
    sh_zh_50 = df[df['UnderlyingSecurityID'] == 204000000015]
    if not (sh_zh_50.shape[0] + h_sh_300.shape[0] == df.shape[0]):
        print('error')
    h_sh_300 = h_sh_300[h_sh_300['TradingDate'] > pd.Timestamp('2020-01-01')]
    h_sh_300.to_csv(f'{DATA_HOME_PATH}/h_sh_300/raw_all.csv', index=False)
    sh_zh_50.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/raw_all.csv', index=False)
    print('done')


def check_seperate_underlying_security():
    h_sh_300 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all.csv')
    # sh_zh_50 = pd.read_csv(f'{DATA_HOME_PATH}/sh_zh_50/all.csv')
    print(h_sh_300.shape)
    # print(sh_zh_50.shape)
    # print(sh_zh_50['TradingDate'].unique())


# 显示相同的交易代码但是有不同的期权id的数据
def show_same_symbol_diff_id(param):
    df = param['df']
    symbols = param['symbols']
    for symbol in symbols:
        if len(df.loc[df['Symbol'] == symbol, 'SecurityID'].unique()) > 1:
            print(symbol)


# 查看上证50和沪深300中期权的id是否有重复
def check_options_id():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv')
    option_ids = df['SecurityID'].unique()

    h_sh_300 = df.loc[df['UnderlyingSecurityID'] == 204000000140, 'SecurityID'].unique()
    sh_zh_50 = df.loc[df['UnderlyingSecurityID'] == 204000000015, 'SecurityID'].unique()
    print(len(option_ids), len(h_sh_300), len(sh_zh_50))

    symbols = df['Symbol'].unique()

    h_sh_300 = df.loc[df['UnderlyingSecurityID'] == 204000000140, 'Symbol'].unique()
    sh_zh_50 = df.loc[df['UnderlyingSecurityID'] == 204000000015, 'Symbol'].unique()
    print(len(option_ids), len(h_sh_300), len(sh_zh_50))

    cpu_num = cpu_count() - 6
    # cpu_num = 1
    if cpu_num > len(df):
        cpu_num = len(df)
    data_chunks = cm.chunks_np(symbols, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(symbols=data_chunk, df=df)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        p.map(show_same_symbol_diff_id, param)
        p.close()
        p.join()
    print('run done!')


def check_sh_zh_50():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv')
    sh_zh_50 = df[df['UnderlyingSecurityID'] == 204000000015]
    print(sh_zh_50['TradingDate'].unique())


def check_h_sh_300():
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all.csv')
    print(df.shape)


# 通过查看数据发现这些脏数据的剩余期限为0，所以可以直接删除
def check_dirt_data_2():
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/dirt_data.csv')
    print(df.shape)


def remove_remaining_term_0_data():
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/raw_all.csv', index_col=0)
    df = df[df['RemainingTerm'] != 0]
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all.csv', index=False)


def combine_all_data():
    """
    IO_PRICINGPARAMETER:股指期权合约定价重要参数表
    证券ID[SecurityID] 交易日期[TradingDate] 交易代码[Symbol] 交易所代码[ExchangeCode] 标的证券ID[UnderlyingSecurityID]
    标的证券交易代码[UnderlyingSecuritySymbol] 合约简称[ShortName] 认购认沽[CallOrPut] 行权价[StrikePrice] 行权日[ExerciseDate]
     收盘价[ClosePrice] 标的证券收盘价[UnderlyingScrtClose] 剩余年限[RemainingTerm] 无风险利率(%)[RisklessRate]
      历史波动率[HistoricalVolatility] 隐含波动率[ImpliedVolatility] 理论价格[TheoreticalPrice] Delta[Delta] Gamma[Gamma]
      Vega[Vega] Theta[Theta] Rho[Rho] 连续股息率[DividendYeild] 数据类型[DataType]

    IO_QUOTATIONBAS: 股指期权合约日交易基础表
    证券ID[SecurityID] 交易日期[TradingDate] 交易代码[Symbol] 交易所代码[ExchangeCode] 合约简称[ShortName]
    交易日状态编码[TradingDayStatusID] 标的证券ID[UnderlyingSecurityID] 标的证券交易代码[UnderlyingSecuritySymbol]
    认购认沽[CallOrPut] 填充标识[Filling] 日开盘价[OpenPrice] 日最高价[HighPrice] 日最低价[LowPrice] 日收盘价[ClosePrice]
     日结算价[SettlePrice] 涨跌1[Change1] 涨跌2[Change2] 成交量[Volume] 持仓量[Position] 成交金额[Amount]
     数据类型[DataType]

    IO_QUOTATIONDER:股指期权合约日交易衍生表
    证券ID[SecurityID] 交易所代码[ExchangeCode] 交易代码[Symbol] 合约简称[ShortName] 认购认沽[CallOrPut]
    交易日期[TradingDate] 标的证券ID[UnderlyingSecurityID] 填充标识[Filling] 连续合约标识[ContinueSign]
    主力合约标识[MainSign] 昨收盘价[PreClosePrice] 昨结算价[PreSettlePrice] 昨持仓量[PrePosition]
    持仓量变化[PositionChange] 成交均价[AvgPrice] 涨跌幅(收盘价)(%)[ClosePriceChangeRatio]
    涨跌幅(结算价)(%)[SettlePriceChangeRatio] 振幅(%)[Amplitude] 涨停价[LimitUp] 跌停价[LimitDown]
    当日单张维持保证金[MaintainingMargin] 数据类型[DataType] 涨跌幅[ChangeRatio]
    :return:
    """
    df_1 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_PRICINGPARAMETER.csv')
    df_2 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONBAS.csv')
    df_3 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONDER.csv')

    print(f'df_1.shape : {df_1.shape} , df_2.shape : {df_2.shape} , df_3.shape : {df_3.shape}')

    fillings = df_2['Filling'].unique()  # 查看有哪些填充 0=正常交易日，填充，1=周一至周五的非节假日填充，2=周一至周五的节假日填充。
    print(f'df_2 fillings : {fillings}')

    fillings = df_3['Filling'].unique()  # 查看有哪些填充 0=正常交易日，填充，1=周一至周五的非节假日填充，2=周一至周五的节假日填充。
    print(f'df_3 fillings : {fillings}')

    # 数据类型：1=正式数据；2=仿真数据
    data_type_1 = df_1['DataType'].unique()
    print(f'data_type_1 : {data_type_1}')
    data_type_2 = df_2['DataType'].unique()
    print(f'data_type_2 : {data_type_2}')
    data_type_3 = df_3['DataType'].unique()
    print(f'data_type_3 : {data_type_3}')

    # 查看 Filling=0 的个数
    df3_filling_1 = df_3[df_3['Filling'] == 0]
    print(f'df3_filling_1.shape[0] : {df3_filling_1.shape[0]}')
    # 查看 Filling=2 的个数
    df3_filling_2 = df_3[df_3['Filling'] == 2]
    print(f'df3_filling_2.shape[0] : {df3_filling_2.shape[0]}')

    # 查看不同 DataType 的个数
    df1_data_type_1 = df_1[df_1['DataType'] == 1]
    df1_data_type_2 = df_1[df_1['DataType'] == 2]
    df2_data_type_1 = df_2[df_2['DataType'] == 1]
    df2_data_type_2 = df_2[df_2['DataType'] == 2]
    df3_data_type_1 = df_3[df_3['DataType'] == 1]
    df3_data_type_2 = df_3[df_3['DataType'] == 2]
    print(
        f'df1_data_type_1.shape[0] : {df1_data_type_1.shape[0]} , df1_data_type_2.shape[0] : {df1_data_type_2.shape[0]}')
    print(
        f'df2_data_type_1.shape[0] : {df2_data_type_1.shape[0]} , df2_data_type_2.shape[0] : {df2_data_type_2.shape[0]}')
    print(
        f'df3_data_type_1.shape[0] : {df3_data_type_1.shape[0]} , df3_data_type_2.shape[0] : {df3_data_type_2.shape[0]}')

    df_1 = df_1[['SecurityID', 'TradingDate', 'CallOrPut', 'StrikePrice', 'ClosePrice', 'UnderlyingScrtClose',
                 'RemainingTerm', 'RisklessRate', 'HistoricalVolatility', 'ImpliedVolatility', 'TheoreticalPrice',
                 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild']]

    df_2 = df_2[['SecurityID', 'TradingDate', 'TradingDayStatusID', 'Filling', 'OpenPrice', 'HighPrice', 'LowPrice',
                 'SettlePrice', 'Change1', 'Change2', 'Volume', 'Position', 'Amount']]

    df_3 = df_3[['SecurityID', 'TradingDate', 'ContinueSign', 'PreClosePrice', 'PreSettlePrice', 'PrePosition',
                 'PositionChange',
                 'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp', 'LimitDown',
                 'MaintainingMargin', 'ChangeRatio']]

    df = pd.merge(df_1, df_2, how='left', on=['TradingDate', 'SecurityID'])
    df = pd.merge(df, df_3, how='left', on=['TradingDate', 'SecurityID'])

    if os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv'):
        os.remove(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv')
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)

    if os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/head_raw_data.csv'):
        os.remove(f'{DATA_HOME_PATH}/h_sh_300/head_raw_data.csv')
    df.head(300).to_csv(f'{DATA_HOME_PATH}/h_sh_300/head_raw_data.csv', index=False)

    if os.path.exists(f'{DATA_HOME_PATH}/h_sh_300/tail_raw_data.csv'):
        os.remove(f'{DATA_HOME_PATH}/h_sh_300/tail_raw_data.csv')
    df.tail(300).to_csv(f'{DATA_HOME_PATH}/h_sh_300/tail_raw_data.csv', index=False)

    print('combine_all_data done!')


def depart_data():
    """
    将数据分成沪深300和上证50
    :return:
    """
    # 	股指期权合约定价重要参数表
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER.csv')
    sh_zh_50_0 = df[df['UnderlyingSecurityID'] == 204000000015]
    h_sh_300_0 = df[df['UnderlyingSecurityID'] == 204000000140]

    # 股指期权合约日交易基础表
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_QUOTATIONBAS.csv')
    sh_zh_50_1 = df[df['UnderlyingSecurityID'] == 204000000015]
    h_sh_300_1 = df[df['UnderlyingSecurityID'] == 204000000140]

    # 股指期权合约日交易衍生表
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_QUOTATIONDER.csv')
    sh_zh_50_2 = df[df['UnderlyingSecurityID'] == 204000000015]
    h_sh_300_2 = df[df['UnderlyingSecurityID'] == 204000000140]

    if not os.path.exists(f'{DATA_HOME_PATH}/sh_zh_50'):
        os.mkdir(f'{DATA_HOME_PATH}/sh_zh_50')
    if not os.path.exists(f'{DATA_HOME_PATH}/h_sh_300'):
        os.mkdir(f'{DATA_HOME_PATH}/h_sh_300')

    sh_zh_50_0.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_PRICINGPARAMETER.csv', index=False)
    h_sh_300_0.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_PRICINGPARAMETER.csv', index=False)

    sh_zh_50_1.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_QUOTATIONBAS.csv', index=False)
    h_sh_300_1.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONBAS.csv', index=False)

    sh_zh_50_2.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_QUOTATIONDER.csv', index=False)
    h_sh_300_2.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONDER.csv', index=False)


# def remove_no_volume_data_2():
#     df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
#     # df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
#     print(df.shape)
#     # position_index = df[(df['Position'] < 1)].index
#
#     amount_index = df[(df['Amount'] < 0)].index  # 当天可以没有成交金额，因为有可能刚开出来，没交易，后面交易多了
#     remainingterm_index = df[(df['RemainingTerm'] == 0)].index  # 剩余年限为0
#     # preposition_index = df[(df['PrePosition'] == 0)].index
#     # df['PositionChange'] = df['Position'] - df['PrePosition']
#
#     print(f', amount_index : {amount_index.shape} , remainingterm_index :'
#           f' {remainingterm_index.shape}')
#     df = df.drop(index=amount_index.append(remainingterm_index).unique())
#     # df = df.drop(index=amount_index)
#     print(df.shape)
#     df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_clean_data.csv', index=False)
#
#     print('remove_no_volume_data_2 done !')


@cm.my_log
def save_by_each_option():
    """
    将每一根合约单独保存
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    # df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    shutil.rmtree(f'{DATA_HOME_PATH}/h_sh_300/each_option')
    os.mkdir(f'{DATA_HOME_PATH}/h_sh_300/each_option')

    option_ids = df['SecurityID'].unique()

    trade_days = []
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        trade_days.append(_options.shape[0])
        sorted_options = _options.sort_values(by='TradingDate')
        sorted_options.to_csv(
            f'{DATA_HOME_PATH}/h_sh_300/each_option/{str(sorted_options.iloc[0]["TradingDate"]).split(" ")[0]}_{option_id}.csv',
            index=False)
    print(trade_days)


@cm.my_log
def remove_filling_not0_data():
    """
    删除非正常交易日的数据
    """
    print('start remove_filling_not0_data ')
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'df.shape : {df.shape}')
    df = df[df['Filling'] == 0]
    print(f'df.shape : {df.shape}')
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)
    print('remove_filling_not0_data done')


def replace_options_by_id(param):
    df = param['df']
    option_ids = param['option_ids']
    df_new = pd.DataFrame()
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        _options = _options.iloc[:-5]
        df_new = df_new.append(_options)
    return df_new


@cm.my_log
def remove_end5_trade_date_data():
    """
    删除最后交易日的数据
    """

    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'df.shape : {df.shape}')
    option_ids = df['SecurityID'].unique()

    cpu_num = 10
    if cpu_num > len(df):
        cpu_num = len(df)
    data_chunks = cm.chunks_np(option_ids, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(option_ids=data_chunk, df=df)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(replace_options_by_id, param)
        p.close()
        p.join()

    new_df = pd.DataFrame()
    for _r in tqdm(r, total=len(r)):
        new_df = new_df.append(_r)
    print(f'new_df.shape : {new_df.shape} , df.shape[0]-option_ids.size()*5 :{df.shape[0] - option_ids.size * 5}')
    new_df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)


@cm.my_log
def check_each_data_num_by_id():
    """
    检查每个表中的相同期权个数是否一致
    """
    df_1 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_PRICINGPARAMETER.csv')
    df_2 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONBAS.csv')
    df_3 = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONDER.csv')
    df_1_ids = np.sort(df_1['SecurityID'].unique())
    df_2_ids = np.sort(df_2['SecurityID'].unique())
    df_3_ids = np.sort(df_3['SecurityID'].unique())
    # df_3_ids[:df_1_ids.size]-df_2_ids
    # 删掉 中多余的SecurityID对应的期权数据
    for df_3_id in df_3_ids[df_1_ids.size:]:
        df_3 = df_3[df_3['SecurityID'] != df_3_id]
    df_3_ids = np.sort(df_3['SecurityID'].unique())
    max_val = (df_3_ids - df_2_ids).max()
    df_3.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONDER.csv')
    print(f'df_1_ids : {df_1_ids.size} , df_2_ids : {df_2_ids.size} , df_3_ids : {df_3_ids.size}')


@cm.my_log
def check_null_by_id():
    print('start check_null_by_id ')
    contain_null_ids = []
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    df.drop(columns=['ImpliedVolatility', 'ContinueSign'], axis=1, inplace=True)
    print(f'df.shape : {df.shape}')
    option_ids = df['SecurityID'].unique()
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        if _options.isnull().values.any():
            contain_null_ids.append(option_id)
    print(f'contain_null_ids : {contain_null_ids}')
    print('check_null_by_id done')


@cm.my_log
def remove_real_trade_days_less28():
    """
    将有效交易天数小于28天的期权全部删除
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'df.shape : {df.shape}')
    option_ids = df['SecurityID'].unique()
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        trade_days = _options.shape[0]
        if trade_days < 28:
            df = df[df['SecurityID'] != option_id]
    print(f'df.shape : {df.shape}')
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)


@cm.my_log
def check_volume():
    """
    在期权数据中，如果当天的成交量（Volume）为0 ， 则会导致 AvgPrice	,ClosePriceChangeRatio,	SettlePriceChangeRatio,	Amplitude
    ChangeRatio 的数据为空。
    实际上只有当Volume为0的时候才会有 AvgPrice	,ClosePriceChangeRatio,	SettlePriceChangeRatio,	Amplitude
    ChangeRatio 数据为空，所以直接将他们填充为0就可以了
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    print(f'df.shape : {df.shape}')
    avg_price = df['AvgPrice'].fillna(0)
    close_price_change_ratio = df['ClosePriceChangeRatio'].fillna(0)
    settle_price_change_ratio = df['SettlePriceChangeRatio'].fillna(0)
    amplitude = df['Amplitude'].fillna(0)
    change_ratio = df['ChangeRatio'].fillna(0)
    position_change = df['PositionChange'].fillna(0)
    df['AvgPrice'] = avg_price
    df['ClosePriceChangeRatio'] = close_price_change_ratio
    df['SettlePriceChangeRatio'] = settle_price_change_ratio
    df['Amplitude'] = amplitude
    df['ChangeRatio'] = change_ratio
    df['PositionChange'] = position_change
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)


def do_append_before4_days_data(param):
    df = param['df']
    option_ids = param['option_ids']
    b_0_c = param['before_0_column']
    b_c = param['before_columns']
    sorted_options_list = []
    for option_id in tqdm(option_ids, total=len(option_ids)):
        _options = df[df['SecurityID'] == option_id]
        sorted_options = _options.sort_values(by='TradingDate', ignore_index=True)
        for i in range(sorted_options.shape[0]):
            for j in range(4):
                if i > j:
                    sorted_options.loc[i, b_c[j]] = np.array(sorted_options.copy().iloc[i - j - 1][b_0_c])
        sorted_options_list.append(sorted_options)
    return pd.concat([_r for _r in sorted_options_list], ignore_index=True)


@cm.my_log
def append_before4_days_data():
    """
    将每条数据的前4天的数据拼接到当前，如果没有前4天的数据就用0填充
    """
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    df.drop(columns=['ImpliedVolatility', 'ContinueSign'], axis=1, inplace=True)
    option_ids = df['SecurityID'].unique()
    print('option_ids num : ', len(option_ids))
    print(f'df.shape : {df.shape}')

    before_0_column = ['StrikePrice', 'ClosePrice', 'UnderlyingScrtClose', 'RisklessRate', 'HistoricalVolatility',
                       'TheoreticalPrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'OpenPrice',
                       'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2', 'Volume', 'Position', 'Amount',
                       'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp',
                       'LimitDown', 'MaintainingMargin', 'ChangeRatio']
    before_1_column = [i + '_1' for i in before_0_column]
    before_2_column = [i + '_2' for i in before_0_column]
    before_3_column = [i + '_3' for i in before_0_column]
    before_4_column = [i + '_4' for i in before_0_column]
    df[before_1_column] = 0
    df[before_2_column] = 0
    df[before_3_column] = 0
    df[before_4_column] = 0
    before_columns = [before_1_column, before_2_column, before_3_column, before_4_column]
    cpu_num = cpu_count() - 16
    if cpu_num > len(df):
        cpu_num = len(df)
    data_chunks = cm.chunks_np(option_ids, cpu_num)
    param = []
    for data_chunk in tqdm(data_chunks, total=len(data_chunks)):
        ARGS_ = dict(option_ids=data_chunk, df=df, before_columns=before_columns, before_0_column=before_0_column)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(do_append_before4_days_data, param)
        p.close()
        p.join()

    expand_df = pd.concat(r, ignore_index=True)
    print(f'expand_df.shape : {expand_df.shape} , df.shape :{df.shape}')
    expand_df.to_csv(f'{DATA_HOME_PATH}/all_expand_df_data.csv', index=False)


@cm.my_log
def get_expand_head():
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_expand_df_data.csv', parse_dates=['TradingDate'])
    option_ids = df['SecurityID'].unique()
    print('option_ids num : ', len(option_ids))
    print(f'df.shape : {df.shape}')
    if os.path.exists(f'{DATA_HOME_PATH}/sub_expand_df_data.csv'):
        os.remove(f'{DATA_HOME_PATH}/sub_expand_df_data.csv')
    df.head(300).to_csv(f'{DATA_HOME_PATH}/sub_expand_df_data.csv', index=False)


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':
    depart_data()
    check_each_data_num_by_id()
    combine_all_data()

    remove_filling_not0_data()  # 删除原始表中节假日填充的数据
    remove_real_trade_days_less28()  # 将合约交易天数小于28天的删除
    remove_end5_trade_date_data()  # 将每份期权合约交易的最后5天的数据删除
    check_volume()  # 将成交量为0的数据中存在nan的地方填充0
    check_null_by_id()  # 查看是否还有nan数据
    save_by_each_option()  # 便于查看每份期权合约的每天交易信息

    append_before4_days_data()  # 将前4天的数据追加的当天，不够4天的用0填充
    get_expand_head()  # 查看填充效果
    # remove_no_volume_data_2()
    # #
    # check_dirt_data()  # 查看是否还有为数据项为空的数据，如果还有请分析原因
    # assemble_next_day_features('h_sh_300', 'all_clean_data.csv')
