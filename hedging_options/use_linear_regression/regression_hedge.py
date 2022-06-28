# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
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

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# torch.multiprocessing.set_sharing_strategy('file_system')


def make_predictor(df, features, delta_coeff_1=False):
    """
    If delta_coeff_1 is to be used, the target is different.
    These predictors are used for linear regressions.
    """

    if delta_coeff_1:
        y = df['V1_n'] - df['V0_n'] * df['on_ret'] - df['delta_bs'] * (df['S1_n'] - df['S0_n'] * df['on_ret'])
    else:
        y = df['V1_n'] - df['V0_n'] * df['on_ret']

    preds = df[features].copy()
    y = y.div(df['S1_n'] - df['S0_n'] * df['on_ret'], axis=0)
    return y, preds


def fit_lin_core(df, features, delta_coeff_1=False):
    """
    Fit a linear regression on the set of features,
    such that (P&L)^2 is minimized
    """
    y, preds = make_predictor(df, features, delta_coeff_1)
    lin = LinearRegression(fit_intercept=False).fit(preds, y)

    y_hat = lin.predict(preds)
    residual = (y - y_hat)

    # sigma_square_hat = residual_sum_of_square / (preds.shape[0] - preds.shape[1])
    # var_beta = (np.linalg.inv(preds.to_numpy().T @ preds.to_numpy()) * sigma_square_hat)
    # std = [var_beta[i, i] ** 0.5 for i in range(len(var_beta))]

    return {'regr': lin, 'std': (residual ** 2).mean()}


def predicted_linear_delta(
        lin, df_test, features
):
    df_delta = pd.Series(index=df_test.index)
    delta = lin.predict(df_test.loc[:, features])
    df_delta.loc[:] = delta

    return df_delta


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import numpy as np


def bs_delta_hedge_in_test(clean_data=False, batch_size=1):
    train_put_data, train_call_data, test_put_data, test_call_data = get_regression_data(clean_data)
    put_results = test_put_data
    call_results = test_call_data
    delta = -put_results['Delta'] - 1
    put_mshes = np.power((100 * (delta * put_results['S1_n'] + put_results['on_ret'] * (
            put_results['V0_n'] - delta * put_results['S0_n'])) / put_results['S1_n']), 2).mean()
    delta = call_results['Delta']
    call_mshes = np.power((100 * (delta * call_results['S1_n'] + call_results['on_ret'] * (
            call_results['V0_n'] - delta * call_results['S0_n'])) / call_results['S1_n']), 2).mean()

    print(round(call_mshes, 3), '\t', round(put_mshes, 3), '\t', round((put_mshes + call_mshes) / 2, 3))
    # put_results, call_results = get_test_data(clean_data)
    # delta = -put_results[:, 6] - 1
    # # delta_bs_c = caux.bs_call_delta(
    # #     vol=put_results[:,12], S=put_results[:,3], K=put_results[:,4], tau=put_results[ :,5], r=put_results[:,0] / 100)
    # # delta = -delta_bs_c-1
    # put_mshes = np.power((100 * (delta * put_results[:, -1] + (1 + put_results[:, 0] / 100 * put_results[:, 5]) * (
    #         put_results[:, 2] - delta * put_results[:, 3]) - put_results[:, -2])) / put_results[:, -1], 2).mean()
    # delta = call_results[:, 6]
    # # delta_bs_c = caux.bs_call_delta(
    # #     vol=call_results[:, 12], S=call_results[:, 3], K=call_results[:, 4], tau=call_results[:, 5],
    # #     r=call_results[:, 0] / 100)
    # # delta = delta_bs_c
    # call_mshes = np.power((100 * (delta * call_results[:, -1] + (1 + call_results[:, 0] / 100 * call_results[:, 5]) * (
    #         call_results[:, 2] - delta * call_results[:, 3]) - call_results[:, -2])) / call_results[:, -1], 2).mean()
    #
    # print(round(call_mshes, 3), '\t', round(put_mshes, 3), '\t', round((put_mshes + call_mshes) / 2, 3))


def get_test_data(clean_data=False, batch_size=1):
    # valid_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_validation_index.csv').to_numpy()
    test_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_test_index.csv').to_numpy()
    # train_data_index = train_data_index[:500, :]
    # valid_data_index = valid_data_index[:3000, :]
    # test_data_index = test_data_index[:500, :]
    # valid_dataset = dataset.Dataset(valid_data_index, f'{PREPARE_HOME_PATH}/parquet/validation/')
    test_dataset = dataset.Dataset(test_data_index, f'{PREPARE_HOME_PATH}/parquet/test/')

    # Create data loaders.
    # val_dataloader = DataLoader(valid_dataset, num_workers=10, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, num_workers=10, batch_size=batch_size)
    put_results = []
    call_results = []
    for ii, (datas, results) in tqdm(enumerate(test_dataloader), total=len(test_data_index) / batch_size):
        # print(results.shape)
        output_dim = datas.shape[0]
        results = results.view(output_dim, -1)
        if clean_data:
            if ((results[0, 11] > 1.5) & (results[0, 1] == 1)) | ((results[0, 11] < 0.5) & (results[0, 1] == 0)):
                continue
            if torch.isnan(results[0, 12]):
                continue
            goto_continue = False
            if results[0, 1] == 1:
                time_value = np.exp(-results[:, 0] / 100 * results[:, 5]) * results[:, 4] - results[:, 3]
                if time_value > results[:, 2]:
                    goto_continue = True
            else:
                time_value = results[:, 3] - np.exp(-results[:, 0] / 100 * results[:, 5]) * results[:, 4]
                if time_value > results[:, 2]:
                    goto_continue = True
            if (results[0, 12] < 0.01) | (results[0, 12] > 1):
                goto_continue = True
            if goto_continue:
                continue

        if results[0, 1] == 1:
            put_results.append(torch.flatten(results).detach().cpu().numpy())
        else:
            call_results.append(torch.flatten(results).detach().cpu().numpy())

    # delta[results[:, 1] == 0] = delta[results[:, 1] == 0] - 1

    _put_results = np.array(put_results)
    _call_results = np.array(call_results)
    print(_put_results.shape, _call_results.shape)
    return _put_results, _call_results




def add_extra_feature(_data):
    # ClosePrice ,StrikePrice,'Vega', 'Theta', 'Rho','Vega_1', 'Theta_1','Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'
    _scale_rate = _data['UnderlyingScrtClose'] / 100
    for _name in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
                  'UnderlyingScrtClose_1']:
        _data[_name] = _data[_name] / _scale_rate
    _data['UnderlyingScrtClose'] = 100
    _data['delta_bs'] = _data['Delta']
    _data['S0_n'] = 100
    _data['S1_n'] = _data['UnderlyingScrtClose_1']
    _data['V0_n'] = _data['ClosePrice']
    _data['V1_n'] = _data['ClosePrice_1']
    _data['on_ret'] = 1 + _data['RisklessRate'] / 100 * (1 / 253)
    return _data


def linear_regression_hedge_call_clean_data(_data):
    pos = _data['M'] < 0.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (_data['UnderlyingScrtClose'] - np.exp(-_data['RisklessRate'] / 100 *
                                                _data['RemainingTerm']) *
          _data['StrikePrice'] >= _data['ClosePrice'])
    _data = _data.loc[~bl]
    return _data


def linear_regression_hedge_put_clean_data(_data):
    pos = _data['M'] > 1.5
    _data = _data.loc[~pos]
    _data = _data.dropna(subset=['ImpliedVolatility'])
    bl = (_data['ImpliedVolatility'] > 1) | (_data['ImpliedVolatility'] < 0.01)
    _data = _data.loc[~bl]
    bl = (np.exp(-_data['RisklessRate'] / 100 * _data['RemainingTerm']) *
          _data['StrikePrice'] - _data['UnderlyingScrtClose'] >= _data[
              'ClosePrice'])
    _data = _data.loc[~bl]
    return _data


def get_regression_data(clean_data):
    # _columns = ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
    #             'Delta',
    #             'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #             'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
    train_put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/train_put_data_h_sh_300.csv',date_parser=[''])
    train_put_data['Delta'] = -train_put_data['Delta'] - 1

    train_call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/train_call_data_h_sh_300.csv')

    test_put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/test_put_data_h_sh_300.csv')
    test_put_data['Delta'] = -test_put_data['Delta'] - 1

    test_call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/test_call_data_h_sh_300.csv')

    if clean_data:
        train_put_data = linear_regression_hedge_put_clean_data(train_put_data)
        train_call_data = linear_regression_hedge_call_clean_data(train_call_data)
        test_put_data = linear_regression_hedge_put_clean_data(test_put_data)
        test_call_data = linear_regression_hedge_call_clean_data(test_call_data)
    print(test_put_data.shape, test_call_data.shape)
    train_put_data = add_extra_feature(train_put_data)
    train_call_data = add_extra_feature(train_call_data)
    test_put_data = add_extra_feature(test_put_data)
    test_call_data = add_extra_feature(test_call_data)
    return train_put_data, train_call_data, test_put_data, test_call_data


def linear_regression_hedge_in_test(clean_data):
    train_put_data, train_call_data, test_put_data, test_call_data = get_regression_data(clean_data)
    features = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'M', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    put_regs = fit_lin_core(train_put_data, features)
    call_regs = fit_lin_core(train_call_data, features)
    predicted_put = predicted_linear_delta(put_regs['regr'], test_put_data, features)
    predicted_call = predicted_linear_delta(call_regs['regr'], test_call_data, features)
    _results = test_put_data
    put_mshe = np.power((100 * (predicted_put * test_put_data['S1_n'] + test_put_data['on_ret'] * (
            test_put_data['V0_n'] - predicted_put * test_put_data['S0_n']) - test_put_data['V1_n'])) / test_put_data[
                            'S1_n'],
                        2).mean()
    _results = test_call_data
    call_mshe = np.power((100 * (predicted_call * test_call_data['S1_n'] + test_call_data['on_ret'] * (
            test_call_data['V0_n'] - predicted_call * test_call_data['S0_n']) - test_call_data['V1_n'])) /
                         test_call_data['S1_n'],
                         2).mean()

    print(round(call_mshe, 3), '\t', round(put_mshe, 3), '\t', round((put_mshe + call_mshe) / 2, 3))


def reset_features(df):
    df['M'] = df['UnderlyingScrtClose'] / df['StrikePrice']
    df = df[
        ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm', 'Delta',
         'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
         'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']]
    df = df.replace({'CallOrPut': {'C': 0, 'P': 1}})

    # underlying_scrt_close = df.iloc[:, 3]
    # underlying_scrt_close_rate = underlying_scrt_close / 1
    # df.iloc[:, -1] = df.iloc[:, -1] / underlying_scrt_close_rate
    # df.iloc[:, 2] = df.iloc[:, 2] / underlying_scrt_close_rate
    # df.iloc[:, 4] = df.iloc[:, 4] / underlying_scrt_close_rate
    # df.iloc[:, -2] = df.iloc[:, -2] / underlying_scrt_close_rate
    # # Theta Vega Rho
    # df.iloc[:, 8] = df.iloc[:, 8] / underlying_scrt_close_rate
    # df.iloc[:, 9] = df.iloc[:, 9] / underlying_scrt_close_rate
    # df.iloc[:, 10] = df.iloc[:, 10] / underlying_scrt_close_rate
    # df.iloc[:, -3] = df.iloc[:, -3] / underlying_scrt_close_rate
    # df.iloc[:, -4] = df.iloc[:, -4] / underlying_scrt_close_rate
    # df.iloc[:, -5] = df.iloc[:, -5] / underlying_scrt_close_rate
    # df.iloc[:, 3] = 1

    return df


def prepare_data():
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_training.csv')
    # valid_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_validation_index.csv').to_numpy()
    test_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_test.csv')
    # train_data = train_data.iloc[:500, :]
    # valid_data_index = valid_data_index[:3000, :]
    # test_data_index = test_data_index[:500, :]

    train_data = reset_features(train_data)
    test_data = reset_features(test_data)
    train_put_data = train_data[train_data['CallOrPut'] == 1]
    train_call_data = train_data[train_data['CallOrPut'] == 0]
    test_put_data = test_data[test_data['CallOrPut'] == 1]
    test_call_data = test_data[test_data['CallOrPut'] == 0]
    train_put_data.to_csv(f'{PREPARE_HOME_PATH}/train_put_data_h_sh_300.csv', index=False)
    train_call_data.to_csv(f'{PREPARE_HOME_PATH}/train_call_data_h_sh_300.csv', index=False)
    test_put_data.to_csv(f'{PREPARE_HOME_PATH}/test_put_data_h_sh_300.csv', index=False)
    test_call_data.to_csv(f'{PREPARE_HOME_PATH}/test_call_data_h_sh_300.csv', index=False)


DEVICE = 'cpu'

# python transformer-code-comments.py > 0.0005-log &
# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/sh_zh_50/'
if __name__ == '__main__':
    prepare_data()
    CLEAN_DATA = False
    print(f'bs_delta_hedge_in_test , clean={CLEAN_DATA}')
    bs_delta_hedge_in_test(CLEAN_DATA)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    linear_regression_hedge_in_test(CLEAN_DATA)
