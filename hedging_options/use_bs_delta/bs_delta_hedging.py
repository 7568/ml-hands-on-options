# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/28
Description:
"""
import sys
import os

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import os
import sys

import torch

# Append the library path to PYTHONPATH, so library can be imported.
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

import random

from hedging_options.library import dataset

from tqdm import tqdm
import resource
from hedging_options.no_hedging import no_hedging


def bs_delta_hedge_in_training(clean_data):
    put_data, call_data = no_hedging.get_data('training', clean_data)
    no_hedging.show_hedge_result(put_data, put_data['delta_bs'], call_data, call_data['delta_bs'])


def bs_delta_hedge_in_validation(clean_data):
    put_data, call_data = no_hedging.get_data('validation', clean_data)
    no_hedging.show_hedge_result(put_data, put_data['delta_bs'], call_data, call_data['delta_bs'])


def bs_delta_hedge_in_test(clean_data):
    # put_results, call_results = get_test_data(clean_data)
    put_data, call_data = no_hedging.get_data('testing', clean_data)
    no_hedging.show_hedge_result(put_data, put_data['delta_bs'], call_data, call_data['delta_bs'])


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    # no_hedging.prepare_data()
    CLEAN_DATA = False
    print(f'bs_delta_hedge_in_test , clean={CLEAN_DATA}')
    # bs_delta_hedge_in_training(CLEAN_DATA)
    bs_delta_hedge_in_validation(CLEAN_DATA)
    bs_delta_hedge_in_test(CLEAN_DATA)
