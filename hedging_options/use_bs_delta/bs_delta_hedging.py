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
    bs_hedge_result(put_data, call_data)


def bs_delta_hedge_in_validation(clean_data):
    put_data, call_data = no_hedging.get_data('validation', clean_data)
    bs_hedge_result(put_data, call_data)


def bs_delta_hedge_in_test(clean_data):
    # put_results, call_results = get_test_data(clean_data)
    put_data, call_data = no_hedging.get_data('testing', clean_data)
    bs_hedge_result(put_data, call_data)


def bs_hedge_result(put_results, call_results):
    delta = put_results['delta_bs']
    put_mshes = np.power((100 * (delta * put_results['S1_n'] + put_results['on_ret'] * (
            put_results['V0_n'] - delta * put_results['S0_n']) - put_results['V1_n'])) / put_results['S0_n'], 2).mean()

    delta = call_results['delta_bs']
    call_mshes = np.power((100 * (delta * call_results['S1_n'] + call_results['on_ret'] * (
            call_results['V0_n'] - delta * call_results['S0_n']) - call_results['V1_n'])) / call_results['S0_n'],
                          2).mean()

    print(round(call_mshes, 3), '\t', round(put_mshes, 3), '\t', round((put_mshes + call_mshes) / 2, 3))

PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/china-market/'
if __name__ == '__main__':
    # no_hedging.prepare_data()
    CLEAN_DATA = True
    print(f'bs_delta_hedge_in_test , clean={CLEAN_DATA}')
    # bs_delta_hedge_in_training(CLEAN_DATA)
    bs_delta_hedge_in_validation(CLEAN_DATA)
    bs_delta_hedge_in_test(CLEAN_DATA)
