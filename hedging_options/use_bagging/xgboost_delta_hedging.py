# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import copy
import sys
import os

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
import pandas as pd
import xgboost as xgb
from hedging_options.no_hedging import no_hedging
from hedging_options.use_linear_regression import regression_hedge
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost.sklearn import XGBRegressor
print(xgb.__version__)

def fit_lin_core2(df, features, delta_coeff_1=False):
    x, y = regression_hedge.make_predictor(df, features, delta_coeff_1)
    params = {
        'n_estimators': 303,
        'learning_rate': 0.01,
        'objective': 'reg:squarederror',
        'scale_pos_weight': 1,
        'reg_alpha': 1000,
        'colsample_bytree': 0.6,
        'subsample': 0.8,
        'gamma': 0.0,
        'max_depth': 5, 'min_child_weight': 5,
        'reg_lambda': 100,
    }

    model = XGBRegressor(**params)
    lin = model.fit(x, y)

    return {'regr': lin}



def fit_lin_core(df, features, delta_coeff_1=False):
    """
    Fit a linear regression on the set of features,
    such that (P&L)^2 is minimized
    """
    x, y = regression_hedge.make_predictor(df, features, delta_coeff_1)
    num_rounds = 3000
    params = {
        'eta': 0.001,
        'objective': 'reg:squarederror',
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 1.1,
        'max_depth': 5,
    }

    dt = xgb.DMatrix(x, label=y)
    early_stop = xgb.callback.EarlyStopping(rounds=30)
    cv = xgb.cv(params, dt, num_boost_round=num_rounds, nfold=5, early_stopping_rounds=30, metrics='rmse', callbacks=[
        copy.deepcopy(early_stop)
    ])
    num_rounds = cv.shape[0] - 1
    print('Best rounds: ', num_rounds)

    # tune max_depth & min_child_weight

    params = {
        'n_estimators': num_rounds,
        'learning_rate': 0.001,
        'objective': 'reg:squarederror',
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 1.1,
        'max_depth': 5,
    }

    model = XGBRegressor(**params)

    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 10, 2)
    }
    gridsearch_cv(model, param_test1, x, y)

    param_test1 = {
        'max_depth': range(5, 8, 1),
        'min_child_weight': range(3, 6, 1)
    }
    gridsearch_cv(model, param_test1, x, y)

    # tune gamma
    param_test2 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gridsearch_cv(model, param_test2, x, y)

    # tune subsample & colsample_bytree
    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gridsearch_cv(model, param_test3, x, y)
    # tune scale_pos_weight
    param_test4 = {
        'scale_pos_weight': [i for i in range(1, 10, 2)]
    }
    gridsearch_cv(model, param_test4, x, y)
    # tune reg_alpha
    param_test5 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000]
    }
    gridsearch_cv(model, param_test5, x, y)
    # tune reg_lambda
    param_test6 = {
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
    }
    gridsearch_cv(model, param_test6, x, y)

    print('Starting Cross Validation...')
    score = cross_val_score(model, x, y, cv=5)
    print('Score: ', score)
    print('Mean CV scores: ', np.mean(score))

    lin = model.fit(x, y)

    return {'regr': lin}


def model_cv(model, X, y, cv_folds=5, early_stopping_rounds=50, seed=0):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', seed=seed, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(early_stopping_rounds)
        ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)
    return num_round_best


def gridsearch_cv(model, test_param, X, y, cv=5):
    gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='r2', n_jobs=4, cv=cv)
    gsearch.fit(X, y)
    print('CV Results: ', gsearch.cv_results_)
    print('Best Params: ', gsearch.best_params_)
    print('Best Score: ', gsearch.best_score_)
    return gsearch.best_params_


def train_linear_regression_hedge(normal_type, clean_data, features):
    train_put_data, train_call_data = no_hedging.get_data(normal_type, 'training', clean_data)
    put_regs = fit_lin_core(train_put_data, features)
    call_regs = fit_lin_core(train_call_data, features)
    return put_regs, call_regs


def predict_xgboost_hedge(normal_type, features, clean_data, tag):
    put_regs, call_regs = train_linear_regression_hedge(normal_type, clean_data, features)
    test_put_data, test_call_data = no_hedging.get_data(normal_type, tag, clean_data)

    predicted_put = regression_hedge.predicted_delta(put_regs['regr'], test_put_data, features)
    predicted_call = regression_hedge.predicted_delta(call_regs['regr'], test_call_data, features)

    no_hedging.show_hedge_result(test_put_data, predicted_put, test_call_data, predicted_call)


PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    CLEAN_DATA = False
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # regression_hedge.prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']

    # predict_xgboost_hedge(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'validation')
    predict_xgboost_hedge(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'testing')
