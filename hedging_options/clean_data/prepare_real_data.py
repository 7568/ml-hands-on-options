# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/2
Description:
"""

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

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
        _options.drop(columns=['ClosePrice_1', 'UnderlyingScrtClose_1', 'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1',
                  'Theta_1', 'Rho_1'])
        new_df = new_df.append(_options.iloc[:-1, :])  # 最后一天的数据不需要，因为既做不了训练集也做不了验证集

    return new_df


# 将每一行数据的下一个交易日的数据加到当天来
def assemble_next_day_features(file_path, file_name):
    df = pd.read_csv(f'{DATA_HOME_PATH}/{file_path}/{file_name}', parse_dates=[
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
    new_df.to_csv(f'{DATA_HOME_PATH}/{file_path}/two_day_{file_name}', index=False)


def check_features():
    df = pd.read_csv(f'{DATA_HOME_PATH}/IO_PRICINGPARAMETER_5.csv', parse_dates=[
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

    df_2 = df_2[['SecurityID', 'TradingDate', 'OpenPrice', 'HighPrice', 'LowPrice',
                 'SettlePrice', 'Change1', 'Change2', 'Volume', 'Position', 'Amount']]

    df_3 = df_3[['SecurityID', 'TradingDate', 'PreClosePrice', 'PreSettlePrice', 'PrePosition', 'PositionChange',
                 'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp', 'LimitDown',
                 'MaintainingMargin', 'ChangeRatio']]

    df = pd.merge(df_1, df_2, how='left', on=['TradingDate', 'SecurityID'])
    df = pd.merge(df, df_3, how='left', on=['TradingDate', 'SecurityID'])

    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', index=False)

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

    sh_zh_50_0.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_PRICINGPARAMETER.csv', index=False)
    h_sh_300_0.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_PRICINGPARAMETER.csv', index=False)

    sh_zh_50_1.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_QUOTATIONBAS.csv', index=False)
    h_sh_300_1.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONBAS.csv', index=False)

    sh_zh_50_2.to_csv(f'{DATA_HOME_PATH}/sh_zh_50/IO_QUOTATIONDER.csv', index=False)
    h_sh_300_2.to_csv(f'{DATA_HOME_PATH}/h_sh_300/IO_QUOTATIONDER.csv', index=False)


def remove_no_volume_data_2():
    df = pd.read_csv(f'{DATA_HOME_PATH}/h_sh_300/all_raw_data.csv', parse_dates=['TradingDate'])
    df = df[df['TradingDate'] > pd.Timestamp('2020-01-01')]
    print(df.shape)
    # position_index = df[(df['Position'] < 1)].index

    amount_index = df[(df['Amount'] < 1)].index
    remainingterm_index = df[(df['RemainingTerm'] == 0)].index
    # preposition_index = df[(df['PrePosition'] == 0)].index
    df['PositionChange'] = df['Position'] - df['PrePosition']

    print(f', amount_index : {amount_index.shape} , remainingterm_index :'
          f' {remainingterm_index.shape}')
    df = df.drop(index=amount_index.append(remainingterm_index).unique())
    # df = df.drop(index=amount_index)
    print(df.shape)
    df.to_csv(f'{DATA_HOME_PATH}/h_sh_300/all_clean_data.csv', index=False)

    print('remove_no_volume_data_2 done !')


DATA_HOME_PATH = '/home/liyu/data/hedging-option/china-market'
if __name__ == '__main__':

    # depart_data()
    # combine_all_data()
    # #
    # #
    # remove_no_volume_data_2()
    # #
    # check_dirt_data()  # 查看是否还有为数据项为空的数据，如果还有请分析原因
    assemble_next_day_features('h_sh_300', 'all_clean_data.csv')
