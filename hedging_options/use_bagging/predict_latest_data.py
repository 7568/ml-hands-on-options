# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/1
Description:
"""
import util
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping
import xgboost as xgb


def xgboost_predict(x, y_true):
    model_from_file = xgb.XGBClassifier()
    model_from_file.load_model('XGBClassifier')
    y_test_hat = model_from_file.predict(x)
    util.eval_accuracy(y_true, y_test_hat)


def lgboost_predict(x, y_true):
    bst_from_file = lgb.Booster(model_file='lgboostClassifier')
    y_test_hat = bst_from_file.predict(x)
    util.eval_accuracy(y_true, y_test_hat > 0.5)


def catboost_predict(x, y_true,cat_fea):
    test_pool = Pool(x, cat_features=cat_fea)
    from_file = CatBoostClassifier()
    from_file.load_model("CatBoostClassifier")
    # make the prediction using the resulting model
    y_test_hat = from_file.predict(test_pool)
    util.eval_accuracy(y_true, y_test_hat)


LATEST_DATA_PATH = f'/home/liyu/data/hedging-option/latest-china-market/h_sh_300/'
NORMAL_TYPE = 'mean_norm'
if __name__ == '__main__':
    df = pd.read_csv(f'{LATEST_DATA_PATH}/{NORMAL_TYPE}/predict_latest.csv', parse_dates=['TradingDate'])
    no_need_columns = ['TradingDate']
    df.drop(columns=no_need_columns, axis=1, inplace=True)
    cat_features = ['CallOrPut', 'MainSign']
    for i in range(1, 5):
        cat_features.append(f'MainSign_{i}')
    df = df.astype({j: int for j in cat_features})
    target_fea = 'up_and_down'
    last_x_index = -6
    xgboost_predict(np.ascontiguousarray(df.iloc[:, :last_x_index].to_numpy()), np.array(df[target_fea]))

    lgboost_predict(df.iloc[:, :last_x_index],df[target_fea])

    catboost_predict(df.iloc[:, :last_x_index], np.array(df[target_fea]).reshape(-1, 1), cat_features)
