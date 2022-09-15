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
        'n_estimators': 300,
        'learning_rate': 0.01,
        'objective': 'reg:squarederror',
        'scale_pos_weight': 1,
        'reg_alpha': 1000,
        'colsample_bytree': 0.8,
        'subsample': 0.6,
        'gamma': 0.4,
        'max_depth': 3,
        'min_child_weight': 5,
        'reg_lambda': 100,
    }

    model = XGBRegressor(**params)
    lin = model.fit(x, y)

    return {'regr': lin}

def fit_lin_core3(df, features, delta_coeff_1=False):
    x, y = regression_hedge.make_predictor(df, features, delta_coeff_1)
    params = {
        'n_estimators': 1,
        'learning_rate': 0.01,
        'objective': 'reg:squarederror',
        'scale_pos_weight': 1,
        'reg_alpha': 1,
        'colsample_bytree': 0.6,
        'subsample': 0.9,
        'gamma': 0.0,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_lambda': 1,
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
    num_rounds = 100


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

    param_test0 = {
        'n_estimators': range(3, 500, 50),
        'max_depth': range(3, 20, 2),
        'min_child_weight': range(3, 10, 1),
        'gamma': [i / 10.0 for i in range(0, 5)],
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        'scale_pos_weight': [i for i in range(1, 10, 2)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
    }
    gridsearch_cv(model, param_test0, x, y)

    # param_test1 = {
    #     'max_depth': range(3, 30, 1),
    #     'min_child_weight': range(3, 6, 1)
    # }
    # gridsearch_cv(model, param_test1, x, y)
    #
    # # tune gamma
    # param_test2 = {
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    # gridsearch_cv(model, param_test2, x, y)
    #
    # # tune subsample & colsample_bytree
    # param_test3 = {
    #     'subsample': [i / 10.0 for i in range(6, 10)],
    #     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    # }
    # gridsearch_cv(model, param_test3, x, y)
    # # tune scale_pos_weight
    # param_test4 = {
    #     'scale_pos_weight': [i for i in range(1, 10, 2)]
    # }
    # gridsearch_cv(model, param_test4, x, y)
    # # tune reg_alpha
    # param_test5 = {
    #     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000]
    # }
    # gridsearch_cv(model, param_test5, x, y)
    # # tune reg_lambda
    # param_test6 = {
    #     'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
    # }
    # gridsearch_cv(model, param_test6, x, y)

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
    from sklearn.metrics import SCORERS
    print(sorted(SCORERS.keys()))
    '''
    'accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
     'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 
     'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 
     'jaccard_samples', 'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_brier_score', 
     'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 
     'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 
     'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 
     'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score',
      'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo',
       'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score'
    '''
    gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='neg_mean_squared_error', n_jobs=4, cv=cv)
    gsearch.fit(X, y)
    print('CV Results: ', gsearch.cv_results_)
    print('Best Params: ', gsearch.best_params_)
    print('Best Score: ', gsearch.best_score_)
    return gsearch.best_params_


def train_linear_regression_hedge(normal_type, clean_data, features,tag):
    train_put_data, train_call_data = no_hedging.get_data(normal_type, 'training', clean_data)
    if tag == 'put':
        return fit_lin_core2(train_put_data, features)
    else:
        return fit_lin_core(train_call_data, features)


def predict_xgboost_hedge_put(normal_type, features, clean_data, tag):
    put_regs = train_linear_regression_hedge(normal_type, clean_data, features,'put')
    test_put_data, test_call_data = no_hedging.get_data(normal_type, tag, clean_data)

    predicted_put = regression_hedge.predicted_delta(put_regs['regr'], test_put_data, features)

    return no_hedging.get_hedge_result(test_put_data, predicted_put)


def predict_xgboost_hedge_call(normal_type, features, clean_data, tag):
    call_regs = train_linear_regression_hedge(normal_type, clean_data, features,'call')
    test_put_data, test_call_data = no_hedging.get_data(normal_type, tag, clean_data)

    predicted_call = regression_hedge.predicted_delta(call_regs['regr'], test_call_data, features)

    return no_hedging.get_hedge_result(test_call_data, predicted_call)


PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    CLEAN_DATA = False
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # regression_hedge.prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']

    # predict_xgboost_hedge(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'validation')
    # put_mshe = predict_xgboost_hedge_put(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'testing')
    call_mshe = predict_xgboost_hedge_call(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'testing')
    # print('put_mshe', put_mshe)
    print('call_mshe', call_mshe)
    # print('mean', (put_mshe + call_mshe) / 2)

