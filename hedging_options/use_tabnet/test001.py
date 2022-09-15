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
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import numpy as np
import pandas as pd
import xgboost as xgb
from hedging_options.no_hedging import no_hedging
from hedging_options.use_linear_regression import regression_hedge
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost.sklearn import XGBRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':

    CLEAN_DATA = False
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    # regression_hedge.prepare_data(NORMAL_TYPE)
    print(f'linear_regression_hedge_in_test , clean={CLEAN_DATA}')
    FEATURES = ['ClosePrice', 'StrikePrice', 'RemainingTerm', 'Mo', 'Gamma', 'Vega', 'Theta', 'Rho', 'Delta']

    # predict_xgboost_hedge(NORMAL_TYPE, FEATURES, CLEAN_DATA, 'validation')

    train_put_data, train_call_data = no_hedging.get_data(NORMAL_TYPE, 'training', CLEAN_DATA)
    train_x, train_y = regression_hedge.make_predictor(train_put_data, FEATURES)

    n, bins, patches = plt.hist(train_y.to_numpy(), 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.xlim(40, 160)
    plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()