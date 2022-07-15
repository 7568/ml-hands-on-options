# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/7/11
Description:
"""
import sys
import os
sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from hedging_options.no_hedging import no_hedging

clf_xgb = XGBRegressor(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='reg:linear',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)
training_put_data, training_call_data = no_hedging.get_data('training', clean_data)
validation_put_data, validation_call_data = no_hedging.get_data('validation', clean_data)
testing_put_data, testing_call_data = no_hedging.get_data('testing', clean_data)
add_extra_feature(_options)
_options = _options.drop(columns=['SecurityID', 'TradingDate', 'Symbol', 'ExchangeCode', 'UnderlyingSecurityID',
                                  'UnderlyingSecuritySymbol', 'ShortName', 'DataType', 'HistoricalVolatility',
                                  'ImpliedVolatility', 'TheoreticalPrice', 'ExerciseDate',
                                  'ImpliedVolatility_1',
                                  'Delta_1', 'Gamma_1', 'Vega_1', 'Theta_1', 'Rho_1', 'index', 'ClosePrice_1',
                                  'UnderlyingScrtClose_1', ])
clf_xgb.fit(training_put_data, training_put_data['Delta'],
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)



preds = np.array(clf_xgb.predict(X_valid))
valid_auc = mean_squared_error(y_pred=preds, y_true=y_valid)
print(valid_auc)

preds = np.array(clf_xgb.predict(X_test))
test_auc = mean_squared_error(y_pred=preds, y_true=y_test)
print(test_auc)