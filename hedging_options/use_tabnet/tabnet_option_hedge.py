# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import copy
import sys
import os

import torch
from torch import nn

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
import pandas as pd
# import xgboost as xgb
from hedging_options.no_hedging import no_hedging
from hedging_options.use_linear_regression import regression_hedge
from sklearn.model_selection import cross_val_score, GridSearchCV
# from xgboost.sklearn import XGBRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--normal_type', default='mean_norm_calls', type=str, help='mean_norm,min_max_norm')
    parser.add_argument('--device', default='cuda', type=str, help='cuda,cpu')
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--n_steps', default=6, type=int)
    parser.add_argument('--input_dim', default=[35, 100], type=list)
    parser.add_argument('--n_a', default=[35, 100], type=list)
    parser.add_argument('--n_d', default=[35, 100], type=list)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--clip_value', default=1, type=int)
    parser.add_argument('--one_day_data_numbers', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--l_r', default=0.001, type=float)
    parser.add_argument('--scheduler_step_size', default=3, type=int)
    parser.add_argument('--scheduler_gamma', default=0.95, type=float)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--drop_last', default=False, type=bool)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--redirect_sys_stderr', default=False, type=bool)
    parser.add_argument('--parquet_data_path',
                        default='/home/liyu/data/hedging-option/china-market/h_sh_300/panel_parquet', type=str)
    parser.add_argument('--next_day_features', default=['S0_n', 'S1_n', 'V0_n', 'V1_n', 'on_ret'], type=list)

    return parser


def seperates_x_y(df, features, args):
    """
    If delta_coeff_1 is to be used, the target is different.
    These predictors are used for linear regressions.
    """

    preds = df[features].copy()
    y = df[args.next_day_features].copy()
    return preds, y


def mshe(_delta, _results):
    _mshes = torch.pow((100 * (_delta * _results[:, 1] + _results[:, -1] * (
            _results[:, 2] - _delta * _results[:, 0]) - _results[:, -2])) / _results[:, 1], 2).mean()
    return _mshes


def fit_tabnet(train_df, validation_df, features, args):
    train_x, train_y = seperates_x_y(train_df, features, args)
    validation_x, validation_y = seperates_x_y(validation_df, features, args)
    cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]
    cat_idxs = 1
    # clf = TabNetRegressor(cat_dims=[], cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
    clf = TabNetRegressor(device_name=args.device,output_dim=args.output_dim,
                          lambda_sparse=1e-4,
                          momentum=0.3,
                          n_steps=args.n_steps,
                          clip_value=args.clip_value,
                          optimizer_fn=torch.optim.Adam,
                          optimizer_params=dict(lr=args.l_r),
                          scheduler_params={"gamma": args.scheduler_gamma,
                                            "step_size": args.scheduler_step_size},
                          scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)
    # aug = RegressionSMOTE(p=0.2)
    aug = None
    max_epochs = args.max_epochs
    clf.fit(
        X_train=train_x.to_numpy(), y_train=train_y.to_numpy(),
        eval_set=[(train_x.to_numpy(), train_y.to_numpy()), (validation_x.to_numpy(), validation_y.to_numpy())],
        eval_name=['train', 'valid'],
        # eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        eval_metric=['mshe'],
        loss_fn=mshe,
        max_epochs=max_epochs,
        patience=500,
        batch_size=args.batch_size, virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        augmentations=aug,  # aug
    )

    return {'regr': clf}


def train_linear_regression_hedge(normal_type, clean_data, features, args, tag):
    train_put_data, train_call_data = no_hedging.get_data(normal_type, 'training', clean_data)
    validation_put_data, validation_call_data = no_hedging.get_data(normal_type, 'validation', clean_data)
    if tag == 'puts':
        return fit_tabnet(train_put_data, validation_put_data, features, args)
    else:
        return fit_tabnet(train_call_data, validation_call_data, features, args)


def train_puts(normal_type, features, clean_data, tag, args):
    put_regs = train_linear_regression_hedge(normal_type, clean_data, features, args, 'puts')
    test_put_data, test_call_data = no_hedging.get_data(normal_type, tag, clean_data)

    predicted_put = regression_hedge.predicted_delta(put_regs['regr'], test_put_data, features)

    no_hedging.get_hedge_result(test_put_data, predicted_put)


def train_calls(normal_type, features, clean_data, tag, args):
    call_regs = train_linear_regression_hedge(normal_type, clean_data, features, args, 'calls')
    _, test_call_data = no_hedging.get_data(normal_type, tag, clean_data)

    predicted_call = regression_hedge.predicted_delta(call_regs['regr'], test_call_data, features)

    no_hedging.get_hedge_result(test_call_data, predicted_call)


PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':

    parser = argparse.ArgumentParser('TabNet training and evaluation script', parents=[get_args_parser()])
    ARGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.cuda_id)
    CLEAN_DATA = False
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # regression_hedge.prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']

    # predict_xgboost_hedge(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'validation')
    # train_puts(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'testing', ARGS)
    train_calls(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'testing', ARGS)
