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
    return  preds,y


def fit_lin_core(df, features, delta_coeff_1=False):
    """
    Fit a linear regression on the set of features,
    such that (P&L)^2 is minimized
    """
    preds,y = make_predictor(df, features, delta_coeff_1)
    # a= np.mean((preds['Delta'].to_numpy().reshape(1,-1)-y.to_numpy().reshape(1,-1))**2)
    # b = np.mean((y.to_numpy().reshape(1,-1))**2)
    # for f in ['ClosePrice', 'StrikePrice', 'Vega', 'Theta', 'Rho']:
    #     if f in preds.columns:
    #         preds[f] /= 100
    lin = LinearRegression(fit_intercept=False).fit(preds, y)

    # lin = SGDRegressor(random_state=0, fit_intercept=False, shuffle=False, max_iter=90000, alpha=0.0005).fit(
    #     preds.to_numpy(), y.to_numpy())
    # lin = RidgeCV(alphas=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1,1], fit_intercept=False).fit(preds.to_numpy(), y.to_numpy())

    y_hat = lin.predict(preds)
    residual = (y - y_hat)
    # print((residual ** 2).mean())

    # sigma_square_hat = residual_sum_of_square / (preds.shape[0] - preds.shape[1])
    # var_beta = (np.linalg.inv(preds.to_numpy().T @ preds.to_numpy()) * sigma_square_hat)
    # std = [var_beta[i, i] ** 0.5 for i in range(len(var_beta))]

    return {'regr': lin, 'std': (residual ** 2).mean()}


def predicted_delta(lin, df_test, features):
    df_delta = pd.Series(index=df_test.index)
    delta = lin.predict(df_test.loc[:, features])
    df_delta.loc[:] = delta

    return df_delta


def train_linear_regression_hedge(normal_type, clean_data, features):
    train_put_data, train_call_data = no_hedging.get_data(normal_type, 'training', clean_data)
    preds, y = make_predictor(train_put_data, features, False)
    # preds2, y2 = make_predictor(train_call_data, features, False)
    # print((y.to_numpy()**2).mean())
    # print(((preds['Delta'].to_numpy()-y.to_numpy())**2).mean())
    # no_hedging.show_hedge_result(train_put_data, y, train_call_data, y2)
    put_regs = fit_lin_core(train_put_data, features)
    call_regs = fit_lin_core(train_call_data, features)
    return put_regs, call_regs


def predict_linear_regression_hedge(normal_type, features, clean_data, tag):
    put_regs, call_regs = train_linear_regression_hedge(normal_type, clean_data, features)
    test_put_data, test_call_data = no_hedging.get_data(normal_type,tag, clean_data)

    predicted_put = predicted_delta(put_regs['regr'], test_put_data, features)
    predicted_call = predicted_delta(call_regs['regr'], test_call_data, features)

    no_hedging.show_hedge_result(test_put_data, predicted_put, test_call_data, predicted_call)










PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    CLEAN_DATA = False
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']
    for i in range(8):
        print('==========')
        predict_linear_regression_hedge(NORMAL_TYPE, FEATURES[i:], CLEAN_DATA, 'training')
        predict_linear_regression_hedge(NORMAL_TYPE, FEATURES[i:], CLEAN_DATA, 'validation')
        predict_linear_regression_hedge(NORMAL_TYPE, FEATURES[i:], CLEAN_DATA, 'testing')
