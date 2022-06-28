# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""

import random

import numpy as np
import pandas as pd

from hedging_options.library import cleaner_aux as caux


def get_dirt_data():
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/testing.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    test_data = train_data.iloc[0:]
    del train_data



    underlying_scrt_close_rate = test_data['UnderlyingScrtClose'] / 100
    K_n = test_data['StrikePrice'] / underlying_scrt_close_rate
    test_data['UnderlyingScrtClose'] = 100
    test_data['StrikePrice'] = K_n
    test_data['delta_bs_c'] = test_data['Delta']
    # Index(['index', 'SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode',
    #        'UnderlyingSecurityID', 'UnderlyingSecuritySymbol', 'ShortName',
    #        'CallOrPut', 'StrikePrice', 'ExerciseDate', 'ClosePrice',
    #        'UnderlyingScrtClose', 'RemainingTerm', 'RisklessRate',
    #        'HistoricalVolatility', 'ImpliedVolatility', 'TheoreticalPrice',
    #        'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'DataType',
    #        'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #        'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'],
    #       dtype='object')
    # show_data(test_data['RemainingTerm'])
    test_data['ClosePrice'] = test_data['ClosePrice'] / underlying_scrt_close_rate
    test_data['ClosePrice_1'] = test_data['ClosePrice_1'] / underlying_scrt_close_rate
    test_data['UnderlyingScrtClose_1'] = test_data['UnderlyingScrtClose_1'] / underlying_scrt_close_rate
    test_data['Mo'] = test_data['UnderlyingScrtClose'] / test_data['StrikePrice']
    test_data['cp_int'] = 1
    bl_p = test_data['CallOrPut'] == 'C'
    test_data.loc[bl_p, 'cp_int'] = 0

    # bl = ((test_data['CallOrPut'] == 'P') & ((test_data['delta_bs_c'] < -1) | (test_data['delta_bs_c'] > -0.01)))
    # test_data = test_data.loc[~bl]
    # bl = ((test_data['CallOrPut'] == 'C') & ((test_data['delta_bs_c'] > 1) | (test_data['delta_bs_c'] < 0.01)))
    # test_data = test_data.loc[~bl]
    # 'V1_n' V0_n  on_ret delta_bs S1_n S0_n
    test_data['delta_bs'] = test_data['delta_bs_c']
    test_data['S0_n'] = 100
    test_data['S1_n'] = test_data['UnderlyingScrtClose_1']
    test_data['V0_n'] = test_data['ClosePrice']
    test_data['V1_n'] = test_data['ClosePrice_1']
    test_data['on_ret'] = 1 + test_data['RisklessRate'] / 100 * (1/253)
    test_data['implvol0'] = test_data['ImpliedVolatility']
    test_data['implvol1'] = test_data['ImpliedVolatility_1']
    _test_data_numpy = test_data[
        ['RisklessRate', 'CallOrPut', 'cp_int', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
         'delta_bs_c', 'delta_bs', 'Delta', 'V1_n', 'V0_n', 'on_ret', 'S1_n', 'S0_n', 'TradingDate', 'implvol0',
         'implvol1',
         'Gamma', 'Vega', 'Theta', 'Rho', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
         'UnderlyingScrtClose_1']]
    return _test_data_numpy

def get_clean_data():
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/testing.csv', parse_dates=[
        'TradingDate', 'ExerciseDate'])
    # test_data_index = pd.read_csv(f'/home/liyu/data/hedging-option/china-market/h_sh_300_test_index.csv').to_numpy()
    # bl = (train_data['SecurityID']==test_data_index[:0]) & (train_data['TradingDate']==test_data_index[:1])
    # train_data = train_data[bl]
    test_data = train_data.iloc[0:]
    del train_data

    # 处理隐含波动率
    test_data = test_data.dropna(subset=['ImpliedVolatility'])
    bl = (test_data['ImpliedVolatility'] > 1) | (test_data['ImpliedVolatility'] < 0.01)
    test_data = test_data.loc[~bl]

    # 将时间价值为负数的删除掉
    bl = (test_data['CallOrPut'] == 'C') & (test_data['UnderlyingScrtClose'] - np.exp(-test_data['RisklessRate'] / 100 *
                                                                                      test_data['RemainingTerm']) *
                                            test_data['StrikePrice'] >= test_data['ClosePrice'])
    test_data = test_data.loc[~bl]

    bl = (test_data['CallOrPut'] == 'P') & (np.exp(-test_data['RisklessRate'] / 100 * test_data['RemainingTerm']) *
                                            test_data['StrikePrice'] - test_data['UnderlyingScrtClose'] >= test_data[
                                                'ClosePrice'])
    test_data = test_data.loc[~bl]

    underlying_scrt_close_rate = test_data['UnderlyingScrtClose'] / 100
    K_n = test_data['StrikePrice'] / underlying_scrt_close_rate
    test_data['UnderlyingScrtClose'] = 100
    test_data['StrikePrice'] = K_n
    print(test_data.shape)
    delta_bs_c = caux.bs_call_delta(
        vol=test_data['ImpliedVolatility'], S=test_data['UnderlyingScrtClose'], K=K_n, tau=test_data[
            'RemainingTerm'],
        r=test_data['RisklessRate'] / 100)
    bl_p = test_data['CallOrPut'] == 'P'
    test_data['delta_bs_c'] = delta_bs_c
    test_data.loc[bl_p, 'delta_bs_c'] -= 1.
    # Index(['index', 'SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode',
    #        'UnderlyingSecurityID', 'UnderlyingSecuritySymbol', 'ShortName',
    #        'CallOrPut', 'StrikePrice', 'ExerciseDate', 'ClosePrice',
    #        'UnderlyingScrtClose', 'RemainingTerm', 'RisklessRate',
    #        'HistoricalVolatility', 'ImpliedVolatility', 'TheoreticalPrice',
    #        'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DividendYeild', 'DataType',
    #        'ImpliedVolatility_1', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #        'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'],
    #       dtype='object')
    # show_data(test_data['RemainingTerm'])
    test_data['ClosePrice'] = test_data['ClosePrice'] / underlying_scrt_close_rate
    test_data['ClosePrice_1'] = test_data['ClosePrice_1'] / underlying_scrt_close_rate
    test_data['UnderlyingScrtClose_1'] = test_data['UnderlyingScrtClose_1'] / underlying_scrt_close_rate
    test_data['Mo'] = test_data['UnderlyingScrtClose'] / test_data['StrikePrice']
    bl = (test_data['Mo'] < 0.8) & (test_data['CallOrPut'] == 'P')
    test_data = test_data[~bl]
    bl = (test_data['Mo'] > 1.5) & (test_data['CallOrPut'] == 'C')
    test_data = test_data[~bl]
    test_data['cp_int'] = 1
    bl_p = test_data['CallOrPut'] == 'C'
    test_data.loc[bl_p, 'cp_int'] = 0

    # bl = ((test_data['CallOrPut'] == 'P') & ((test_data['delta_bs_c'] < -1) | (test_data['delta_bs_c'] > -0.01)))
    # test_data = test_data.loc[~bl]
    # bl = ((test_data['CallOrPut'] == 'C') & ((test_data['delta_bs_c'] > 1) | (test_data['delta_bs_c'] < 0.01)))
    # test_data = test_data.loc[~bl]
    # 'V1_n' V0_n  on_ret delta_bs S1_n S0_n
    test_data['delta_bs'] = test_data['delta_bs_c']
    test_data['S0_n'] = 100
    test_data['S1_n'] = test_data['UnderlyingScrtClose_1']
    test_data['V0_n'] = test_data['ClosePrice']
    test_data['V1_n'] = test_data['ClosePrice_1']
    test_data['on_ret'] = 1 + test_data['RisklessRate'] / 100 * (1/253)
    test_data['implvol0'] = test_data['ImpliedVolatility']
    test_data['implvol1'] = test_data['ImpliedVolatility_1']
    _test_data_numpy = test_data[
        ['RisklessRate', 'CallOrPut', 'cp_int', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
         'delta_bs_c', 'delta_bs', 'Delta', 'V1_n', 'V0_n', 'on_ret', 'S1_n', 'S0_n', 'TradingDate','implvol0', 'implvol1',
         'Gamma', 'Vega', 'Theta', 'Rho', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
         'UnderlyingScrtClose_1']]
    return _test_data_numpy

'''
对 delta 进行回归分析和不做对冲的分析，发现在早期2020年之前，他们的 MSHE 都比较大，
有的时候高达10左右，而到了2020年之后，就变得很小，大多数时候都在1以内。
也就是说在2020年之后，期权本身的风险就已经很低，即使是不进行对冲。
'''
def test_1():
    clean_data = get_dirt_data().to_numpy()
    partial_num = 20
    _len = int(clean_data.shape[0] / partial_num)
    for i in range(partial_num):
        test_data_numpy = clean_data[i * _len:(i + 1) * _len, :]
        # print(f'======= {i} ====={test_data_numpy[0, 15].strftime("%Y-%m-%d")} to {test_data_numpy[-1, 15].strftime("%Y-%m-%d")}=========')
        call_data = test_data_numpy[test_data_numpy[:, 1] == 'C']
        put_data = test_data_numpy[test_data_numpy[:, 1] == 'P']
        _delta = call_data[:, 9]

        call_mshe = np.power((100 * (_delta * call_data[:, -1] + call_data[:, 12] * (
                call_data[:, 3] - _delta * call_data[:, 4]) - call_data[:, -2])) / call_data[:, -1], 2).mean()

        _delta = -put_data[:, 9]-1

        put_mshe = np.power((100 * (_delta * put_data[:, -1] + put_data[:, 12] * (
                put_data[:, 3] - _delta * put_data[:, 4]) - put_data[:, -2])) / put_data[:, -1], 2).mean()
        # print(round(call_mshe,3),'\t',round(put_mshe,3))
        # print(round(put_mshe,3))

        _delta = 0
        call_mshe_0_delta = np.power((100 * (_delta * call_data[:, -1] + call_data[:, 12] * (
                call_data[:, 3] - _delta * call_data[:, 4]) - call_data[:, -2])) / call_data[:, -1], 2).mean()

        put_mshe_0_delta = np.power((100 * (_delta * put_data[:, -1] + put_data[:, 12] * (
                put_data[:, 3] - _delta * put_data[:, 4]) - put_data[:, -2])) / put_data[:, -1], 2).mean()
        # print(call_mshe_0_delta)
        # print(put_mshe_0_delta)
        # print(round(call_mshe_0_delta, 3), round(put_mshe_0_delta, 3))

        # print((call_mshe_0_delta-call_mshe)/call_mshe_0_delta)
        # print((put_mshe_0_delta-put_mshe)/put_mshe_0_delta)
        print(round(call_mshe,3),'\t',round(put_mshe,3),'\t',round(call_mshe_0_delta, 3), '\t',round(put_mshe_0_delta, 3),
              '\t',
              round(round((call_mshe_0_delta-call_mshe)/call_mshe_0_delta, 3)*100,3), '\t',
              round(round((put_mshe_0_delta-put_mshe)/put_mshe_0_delta, 3)*100,3))

# 从全部数据中随机抽取1/10的数据用来做测试
def test_2():
    clean_data = get_clean_data().to_numpy()
    s = random.sample(range(clean_data.shape[0]),int(clean_data.shape[0]/10))
    test_data_numpy = clean_data[s,:]
    print(test_data_numpy.shape)
    call_data = test_data_numpy[test_data_numpy[:, 1] == 'C']
    put_data = test_data_numpy[test_data_numpy[:, 1] == 'P']
    _delta = call_data[:, 8]

    call_mshe = np.power((100 * (_delta * call_data[:, -1] + call_data[:, 12] * (
            call_data[:, 3] - _delta * call_data[:, 4]) - call_data[:, -2])) / call_data[:, -1], 2).mean()

    _delta = put_data[:, 8]

    put_mshe = np.power((100 * (_delta * put_data[:, -1] + put_data[:, 12] * (
            put_data[:, 3] - _delta * put_data[:, 4]) - put_data[:, -2])) / put_data[:, -1], 2).mean()
    print(call_mshe)
    print(put_mshe)

    _delta = 0
    call_mshe = np.power((100 * (_delta * call_data[:, -1] + call_data[:, 12] * (
            call_data[:, 3] - _delta * call_data[:, 4]) - call_data[:, -2])) / call_data[:, -1], 2).mean()

    put_mshe = np.power((100 * (_delta * put_data[:, -1] + put_data[:, 12] * (
            put_data[:, 3] - _delta * put_data[:, 4]) - put_data[:, -2])) / put_data[:, -1], 2).mean()
    print(call_mshe)
    print(put_mshe)

PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    test_1()
    test_2()
