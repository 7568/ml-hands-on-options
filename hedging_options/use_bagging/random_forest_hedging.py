# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/20
Description:
"""
import sys
import os
sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
from hedging_options.use_linear_regression import regression_hedge


def make_predictor(df):
    df = df.to_numpy()[0:,:]
    underlying_scrt_close_rate = df[:, 3] / 100
    # # Theta Vega Rho
    df[:, 4] = df[:, 4] / underlying_scrt_close_rate
    df[:, 8] = df[:, 8] / underlying_scrt_close_rate
    df[:, 9] = df[:, 9] / underlying_scrt_close_rate
    df[:, 10] = df[:, 10] / underlying_scrt_close_rate
    df[:, -2] = df[:, -2] / underlying_scrt_close_rate
    df[:, -3] = df[:, -3] / underlying_scrt_close_rate
    # x = df[:, [-1, -3, -5, -6, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    x = df[:, [6]]
    y = (df[:, -2] - df[:, -1] * df[:, -3]) / (df[:, -4] - df[:, -1] * df[:, -5])
    return x, y


# Index(['RisklessRate', 'CallOrPut', 'ClosePrice', 'UnderlyingScrtClose',
#        'StrikePrice', 'RemainingTerm', 'Delta', 'Gamma', 'Vega', 'Theta',
#        'Rho', 'M', 'ImpliedVolatility', 'Delta_1', 'Gamma_1', 'Vega_1',
#        'Theta_1', 'Rho_1', 'ClosePrice_1', 'UnderlyingScrtClose_1', 'delta_bs',
#        'S0_n', 'S1_n', 'V0_n', 'V1_n', 'on_ret'],
#       dtype='object')


def fit_random_forest_core(df, features, delta_coeff_1=False):
    """
    Fit a linear regression on the set of features,
    such that (P&L)^2 is minimized
    """
    x,y = make_predictor(df)
    model = RandomForestRegressor(n_estimators=100).fit(x, y)
    return model


def random_forest_regressor(clean_data):
    features = ['Delta']
    train_put_data, train_call_data, test_put_data, test_call_data = regression_hedge.get_regression_data(clean_data)
    model_put = fit_random_forest_core(train_put_data, features, False)
    x_put,y_put = make_predictor(test_put_data)
    y_put_hat = model_put.predict(x_put)

    print(np.mean((y_put-y_put_hat)**2))

    model_call = fit_random_forest_core(train_call_data, features, False)
    x_call, y_call = make_predictor(test_call_data)
    y_call_hat = model_call.predict(x_call)
    print(np.mean((y_call - y_call_hat) ** 2))


if __name__ == '__main__':
    CLEAN_DATA = True
    print(f'random_forest_regressor , clean={CLEAN_DATA}')
    random_forest_regressor(CLEAN_DATA)
