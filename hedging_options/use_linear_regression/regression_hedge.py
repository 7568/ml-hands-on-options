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
from sklearn.linear_model import LinearRegression
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
    lin = LinearRegression(fit_intercept=False).fit(preds, y)

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




def train_linear_regression_hedge(clean_data, features):
    train_put_data, train_call_data = no_hedging.get_data('training', clean_data)
    put_regs = fit_lin_core(train_put_data, features)
    call_regs = fit_lin_core(train_call_data, features)
    return put_regs, call_regs


def predict_linear_regression_hedge(features, clean_data, tag):
    put_regs, call_regs = train_linear_regression_hedge(clean_data, features)
    test_put_data, test_call_data = no_hedging.get_data(tag, clean_data)

    predicted_put = predicted_linear_delta(put_regs['regr'], test_put_data, features)
    predicted_call = predicted_linear_delta(call_regs['regr'], test_call_data, features)

    predicted_delta = predicted_put
    # predicted_delta = test_put_data['delta_bs']
    put_mshe = np.power((100 * (predicted_delta * test_put_data['S1_n'] + test_put_data['on_ret'] * (
            test_put_data['V0_n'] - predicted_delta * test_put_data['S0_n']) - test_put_data['V1_n'])) / test_put_data[
                            'S0_n'], 2).mean()
    predicted_delta = predicted_call
    # predicted_delta = test_call_data['delta_bs']
    call_mshe = np.power((100 * (predicted_delta * test_call_data['S1_n'] + test_call_data['on_ret'] * (
            test_call_data['V0_n'] - predicted_delta * test_call_data['S0_n']) - test_call_data['V1_n'])) /
                         test_call_data['S0_n'], 2).mean()

    print(round(call_mshe, 3), '\t', round(put_mshe, 3), '\t', round((put_mshe + call_mshe) / 2, 3))





PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    # no_hedging.prepare_data()
    CLEAN_DATA = True

    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'M', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']
    for i in range(8):
        predict_linear_regression_hedge(FEATURES[i:], CLEAN_DATA, 'validation')
        predict_linear_regression_hedge(FEATURES[i:], CLEAN_DATA, 'testing')
