# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import sys
import os

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, RidgeCV
from hedging_options.no_hedging import no_hedging


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
    # for f in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho']:
    #     if f in preds.columns:
    #         preds[f] /= 100
    lin = LinearRegression(fit_intercept=False).fit(preds, y)
    # lin = SGDRegressor(random_state=0, fit_intercept=False, shuffle=False, max_iter=90000, alpha=0.0005).fit(
    #     preds.to_numpy(), y.to_numpy())
    # lin = RidgeCV(alphas=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1,1], fit_intercept=False).fit(preds.to_numpy(), y.to_numpy())

    y_hat = lin.predict(preds)
    residual = (y - y_hat)

    # sigma_square_hat = residual_sum_of_square / (preds.shape[0] - preds.shape[1])
    # var_beta = (np.linalg.inv(preds.to_numpy().T @ preds.to_numpy()) * sigma_square_hat)
    # std = [var_beta[i, i] ** 0.5 for i in range(len(var_beta))]

    return {'regr': lin, 'std': (residual ** 2).mean()}


def predicted_linear_delta(lin, df_test, features):
    df_delta = pd.Series(index=df_test.index)
    delta = lin.predict(df_test.loc[:, features])
    df_delta.loc[:] = delta

    return df_delta


def train_linear_regression_hedge(normal_type, clean_data, features):
    train_put_data, train_call_data = get_data(normal_type, 'training', clean_data)
    put_regs = fit_lin_core(train_put_data, features)
    call_regs = fit_lin_core(train_call_data, features)
    return put_regs, call_regs


def predict_linear_regression_hedge(normal_type, features, clean_data, tag):
    put_regs, call_regs = train_linear_regression_hedge(normal_type, clean_data, features)
    test_put_data, test_call_data = get_data(normal_type,tag, clean_data)

    predicted_put = predicted_linear_delta(put_regs['regr'], test_put_data, features)
    predicted_call = predicted_linear_delta(call_regs['regr'], test_call_data, features)

    no_hedging.show_hedge_result(test_put_data, predicted_put, test_call_data, predicted_call)


def go_back_old_value(normal_type, column, _data):
    if normal_type == 'mean_norm':
        mean_normalization_info = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/mean_normalization_info.csv')
        _close_old_info = mean_normalization_info[mean_normalization_info['column'] == column]
        std = _close_old_info['std'].values[0]
        mean = _close_old_info['mean'].values[0]
        underlying_scrt_close_old = _data[column] * std + mean
        return underlying_scrt_close_old
    else:
        mean_normalization_info = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/min_max_normalization_info.csv')
        _close_old_info = mean_normalization_info[mean_normalization_info['column'] == column]
        max = _close_old_info['max'].values[0]
        min = _close_old_info['min'].values[0]
        underlying_scrt_close_old = _data[column] * (max - min) + min
        return underlying_scrt_close_old


def add_extra_feature(normal_type, _data):
    # ClosePrice ,StrikePrice,'Vega', 'Theta', 'Rho','Vega_1', 'Theta_1','Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1'
    underlying_scrt_close_old = go_back_old_value(normal_type, 'UnderlyingScrtClose', _data)
    _scale_rate = underlying_scrt_close_old / 100
    # for _name in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho', 'Vega_1', 'Theta_1', 'Rho_1', 'ClosePrice_1',
    #               'UnderlyingScrtClose_1']:
    #     _data[_name] = _data[_name] / _scale_rate
    # _data['UnderlyingScrtClose'] = 100
    _data['S0_n'] = 100
    _data['S1_n'] = go_back_old_value(normal_type, 'UnderlyingScrtClose_1', _data) / _scale_rate
    _data['V0_n'] = go_back_old_value(normal_type, 'ClosePrice', _data) / _scale_rate
    _data['V1_n'] = go_back_old_value(normal_type, 'ClosePrice_1', _data) / _scale_rate
    # _data['on_ret'] = 1 + _data['RisklessRate'] / 100 * (1 / 253)
    _data['Delta'] = go_back_old_value(normal_type, 'Delta', _data)
    return _data


def get_data(normal_type, tag, clean_data):
    # _columns = ['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose', 'StrikePrice', 'RemainingTerm',
    #             'Delta',
    #             'Gamma', 'Vega', 'Theta', 'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1',
    #             'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1']
    _put_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/{tag}_put_data.csv', date_parser=[''])
    _put_data['Delta'] = -_put_data['Delta'] - 1

    _call_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/{tag}_call_data.csv')

    if clean_data:
        _put_data = no_hedging.distill_put_clean_data(_put_data)
        _call_data = no_hedging.distill_call_clean_data(_call_data)
    _put_data = add_extra_feature(normal_type, _put_data)
    _call_data = add_extra_feature(normal_type, _call_data)
    return _put_data, _call_data

def prepare_data(normal_type):
    train_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training.csv')
    valid_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation.csv')
    test_data = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing.csv')
    # train_data = train_data.iloc[:500, :]
    # valid_data_index = valid_data_index[:3000, :]
    # test_data_index = test_data_index[:500, :]

    # train_data = reset_features(train_data)
    # valid_data = reset_features(valid_data)
    # test_data = reset_features(test_data)
    train_put_data = train_data[train_data['CallOrPut'] == 1]
    train_call_data = train_data[train_data['CallOrPut'] == 0]
    valid_put_data = valid_data[valid_data['CallOrPut'] == 1]
    valid_call_data = valid_data[valid_data['CallOrPut'] == 0]
    test_put_data = test_data[test_data['CallOrPut'] == 1]
    test_call_data = test_data[test_data['CallOrPut'] == 0]
    train_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training_put_data.csv', index=False)
    train_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/training_call_data.csv', index=False)
    valid_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation_put_data.csv', index=False)
    valid_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/validation_call_data.csv', index=False)
    test_put_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing_put_data.csv', index=False)
    test_call_data.to_csv(f'{PREPARE_HOME_PATH}/h_sh_300/{normal_type}/testing_call_data.csv', index=False)


PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    CLEAN_DATA = False
    NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'mean_norm'
    # prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']
    for i in range(8):
        print('==========')
        predict_linear_regression_hedge(NORMAL_TYPE, FEATURES[i:], CLEAN_DATA, 'validation')
        predict_linear_regression_hedge(NORMAL_TYPE, FEATURES[i:], CLEAN_DATA, 'testing')
